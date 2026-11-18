"""Backend-engineering prompt set with machine-verified gold answers.

Four categories, each tagged easy / hard:
  python   "what does this print?"       gold = run the snippet in a subprocess
  sql      single-value query on a toy DB gold = run it in sqlite3
  infra    reliability / capacity math    gold = Python arithmetic
  finance  ledger / interest math         gold = Python Decimal arithmetic

No gold answer is ever hand-typed: Python and SQL golds come from execution, math
golds from the formula functions below. Items whose execution fails, whose
output is not a single line, or whose rounding lands near a tie are dropped.
"""

from __future__ import annotations

import math
import random
import sqlite3
import subprocess
import sys
import textwrap
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable

ANSWER_FOOTER_NUMBER = (
    "Reason briefly. Then write the final answer on its own last line in exactly this format:\n"
    "ANSWER: <number>\n"
    "Write only the number on the ANSWER line (no units, no $ sign, no commas)."
)

# ============================================================================ python

PY_EASY = [
    "print(sorted([3, 1, 2]))",
    'print("a,b,,c".split(","))',
    'print("hello world"[::-1])',
    "print([1, 2, 3, 4, 5][1:4])",
    'print(len("backend"))',
    'print("-".join(["a", "b", "c"]))',
    "print(10 % 3)",
    "print(2 ** 10)",
    "print(list(range(2, 10, 3)))",
    'print({"a": 1, "b": 2}.get("c", 0))',
    'print("Hello".upper())',
    "print(max([4, 9, 2]))",
    "print(sum(x * x for x in range(4)))",
    'print("  pad  ".strip())',
    "print([x for x in range(10) if x % 3 == 0])",
    'print(int("42") + 1)',
    "print(type(3.0).__name__)",
    "print(len({1, 2, 2, 3}))",
    'print("abc" * 2)',
    'print(list(zip([1, 2, 3], "ab")))',
    "print(abs(-7))",
    'print("x=%d" % 5)',
    'd = {"a": 1}\nd["b"] = 2\nprint(list(d.keys()))',
    'print("a b  c".split())',
    'print(min("zebra"))',
    "print(sorted([3, 1, 2], reverse=True))",
    'print(f"{3.14159:.2f}")',
    "print(7 / 2)",
    'print("x" in "box")',
]

PY_HARD = [
    # mutable default argument
    "def f(x, acc=[]):\n    acc.append(x)\n    return acc\n\nf(1)\nf(2)\nprint(f(3))",
    # late-binding closures
    "fs = [lambda: i for i in range(3)]\nprint([f() for f in fs])",
    # floor division of negatives
    "print(-7 // 2)",
    "print(-7 % 3)",
    # generator exhaustion
    "g = (x * x for x in range(3))\nprint(sum(g), sum(g))",
    # return in finally overrides try
    "def f():\n    try:\n        return 1\n    finally:\n        return 2\n\nprint(f())",
    # dict key collisions: 1 == True == 1.0
    "print({1: 'a', True: 'b', 1.0: 'c'})",
    # float representation
    "print(round(2.675, 2))",
    # banker's rounding
    "print(round(0.5), round(1.5), round(2.5))",
    # aliasing via list multiplication
    "a = [[0] * 2] * 2\na[0][0] = 1\nprint(a)",
    "print(0.1 + 0.2 == 0.3)",
    # += mutates in place through an alias
    "x = [1, 2, 3]\ny = x\ny += [4]\nprint(x)",
    'print(bool("False"))',
    'print(sorted(["b", "A", "a", "B"]))',
    "import math\nprint(int(-3.7), math.floor(-3.7))",
    # class attribute shared across instances
    "class C:\n    items = []\n\na, b = C(), C()\na.items.append(1)\nprint(b.items)",
    # closures capture enclosing scope, not global
    "x = 10\n\ndef f():\n    x = 20\n    def g():\n        return x\n    return g()\n\nprint(f(), x)",
    # for/else runs when the loop did not break
    'for i in range(3):\n    if i == 5:\n        break\nelse:\n    print("done", i)',
    'print(list(dict.fromkeys("mississippi")))',
    "print(1 < 2 < 3 > 2 == 2)",
    "print([] == False, not [])",
    # string immutability vs list mutation in a function
    "def f(s, l):\n    s += 'x'\n    l += ['x']\n\ns, l = 'a', ['a']\nf(s, l)\nprint(s, l)",
    # is/== on small ints is impl-specific, so test tuple unpacking order instead
    "a, b = 1, 2\na, b = b, a + b\nprint(a, b)",
]


def run_python(code: str, timeout: float = 5.0) -> str:
    out = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip())
    lines = out.stdout.rstrip("\n").split("\n")
    if len(lines) != 1 or not lines[0].strip():
        raise ValueError(f"expected exactly one non-empty line, got {lines!r}")
    return lines[0].rstrip()


def python_prompt(code: str) -> str:
    return (
        "What does the following Python 3 program print?\n\n"
        f"```python\n{code}\n```\n\n"
        "Reason briefly. Then write the final answer on its own last line in exactly this format:\n"
        "ANSWER: <output>\n"
        "where <output> is exactly what the program prints (one line), with no surrounding quotes or backticks."
    )


# =============================================================================== sql

SQL_SCHEMA = textwrap.dedent(
    """\
    CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
    INSERT INTO users VALUES
      (1, 'alice', 'US', 34),
      (2, 'bob',   'UK', NULL),
      (3, 'carol', 'US', 28),
      (4, 'dave',  NULL, 45),
      (5, 'erin',  'DE', 31),
      (6, 'frank', 'UK', 52);

    CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT);
    INSERT INTO orders VALUES
      (101, 1, 120.0, 'paid'),
      (102, 1,  80.0, 'pending'),
      (103, 2, 200.0, 'paid'),
      (104, 3,  50.0, 'refunded'),
      (105, 3,  75.0, 'paid'),
      (106, 3,  NULL, 'paid'),
      (107, 5, 300.0, 'paid'),
      (108, 5,  20.0, 'paid'),
      (109, 7,  60.0, 'paid'),
      (110, 2,  40.0, NULL);"""
)

SQL_EASY = [
    "SELECT COUNT(*) FROM users;",
    "SELECT COUNT(*) FROM orders WHERE status = 'paid';",
    "SELECT name FROM users WHERE age > 40 ORDER BY age LIMIT 1;",
    "SELECT SUM(amount) FROM orders WHERE user_id = 1;",
    "SELECT MAX(amount) FROM orders;",
    "SELECT COUNT(*) FROM users WHERE country = 'US';",
    "SELECT name FROM users ORDER BY name DESC LIMIT 1;",
    "SELECT MIN(age) FROM users;",
    "SELECT COUNT(DISTINCT user_id) FROM orders;",
    "SELECT amount FROM orders WHERE id = 105;",
    "SELECT COUNT(*) FROM orders WHERE amount >= 100;",
    "SELECT name FROM users WHERE id = (SELECT user_id FROM orders WHERE id = 107);",
    "SELECT UPPER(name) FROM users WHERE id = 3;",
    "SELECT LENGTH(name) FROM users WHERE id = 6;",
    "SELECT COUNT(*) FROM orders o JOIN users u ON o.user_id = u.id WHERE u.country = 'UK';",
    "SELECT ROUND(AVG(amount), 2) FROM orders WHERE user_id = 5;",
    "SELECT COUNT(*) FROM users WHERE name LIKE '%a%';",
]

SQL_HARD = [
    # NULL is excluded by != (three-valued logic)
    "SELECT COUNT(*) FROM users WHERE country != 'US';",
    "SELECT COUNT(age) FROM users;",
    "SELECT AVG(age) FROM users;",
    "SELECT COUNT(*) FROM orders WHERE status != 'paid';",
    "SELECT SUM(amount) FROM orders WHERE user_id = 3;",
    # anti-join: users with no orders
    "SELECT COUNT(*) FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE o.id IS NULL;",
    "SELECT user_id FROM orders GROUP BY user_id HAVING COUNT(*) >= 3;",
    "SELECT COUNT(*) FROM (SELECT user_id FROM orders GROUP BY user_id HAVING SUM(amount) > 150);",
    # second-highest
    "SELECT MAX(amount) FROM orders WHERE amount < (SELECT MAX(amount) FROM orders);",
    # users whose orders are ALL paid (NULL status is not paid)
    "SELECT COUNT(*) FROM (SELECT user_id FROM orders GROUP BY user_id "
    "HAVING SUM(status = 'paid') = COUNT(*));",
    "SELECT COUNT(*) FROM orders WHERE user_id NOT IN (SELECT id FROM users);",
    # NOT IN with a NULL in the subquery matches nothing
    "SELECT COUNT(*) FROM users WHERE country NOT IN (SELECT country FROM users WHERE age > 40);",
    "SELECT COUNT(*) FROM orders o JOIN users u ON o.user_id = u.id;",
    "SELECT u.name FROM users u JOIN orders o ON o.user_id = u.id GROUP BY u.id "
    "ORDER BY SUM(o.amount) DESC LIMIT 1;",
    "SELECT COALESCE(SUM(amount), 0) FROM orders WHERE user_id = 4;",
    # AND binds tighter than OR
    "SELECT COUNT(*) FROM users WHERE age > 30 OR country = 'US' AND age < 30;",
    "SELECT COUNT(*) FROM orders WHERE amount > NULL;",
    "SELECT 7 / 2;",
    "SELECT COUNT(*) FROM users u1 JOIN users u2 ON u1.country = u2.country AND u1.id < u2.id;",
    "SELECT SUM(CASE WHEN status = 'paid' THEN amount ELSE 0 END) FROM orders;",
]


def run_sql(query: str) -> tuple[str, str]:
    """Return (gold, answer_type) for a query that yields exactly one value."""
    con = sqlite3.connect(":memory:")
    con.executescript(SQL_SCHEMA)
    rows = con.execute(query).fetchall()
    con.close()
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError(f"expected a single value, got {rows!r}")
    v = rows[0][0]
    if v is None:
        return "NULL", "exact"
    if isinstance(v, (int, float)):
        return repr(v), "number"
    return str(v), "exact"


def sql_prompt(query: str) -> str:
    return (
        "Given this SQLite database:\n\n"
        f"```sql\n{SQL_SCHEMA}\n```\n\n"
        "What single value does this query return?\n\n"
        f"```sql\n{query}\n```\n\n"
        "Reason briefly. Then write the final answer on its own last line in exactly this format:\n"
        "ANSWER: <value>\n"
        "Write just the value (no quotes); write NULL if the result is NULL."
    )


# ===================================================================== math helpers


class NearTie(ValueError):
    pass


def round_safe(x: float | Decimal, places: int) -> Decimal:
    """Round half-up, refusing values that sit on a rounding tie (ambiguous gold)."""
    d = Decimal(str(x))
    q = Decimal(1).scaleb(-places)
    scaled = d / q
    frac = abs(scaled - scaled.to_integral_value(rounding="ROUND_FLOOR"))
    if abs(frac - Decimal("0.5")) < Decimal("0.001"):
        raise NearTie(f"{x} is too close to a rounding tie at {places} places")
    return d.quantize(q, rounding=ROUND_HALF_UP)


def fmt(d: Decimal | int) -> str:
    return str(d)


# Each template: (difficulty, generator(rng) -> (question_text, gold_number))
Template = tuple[str, Callable[[random.Random], tuple[str, Decimal | int]]]


# ============================================================================= infra


def _t_logs_gb(r):
    mb = r.choice([150, 240, 375, 512, 820, 1300])
    gb = round_safe(mb * 24 / 1024, 2)
    return (f"A service writes {mb} MB of logs per hour. How many GB does it write per day? "
            f"Use 1 GB = 1024 MB and round to 2 decimal places."), gb


def _t_rpm(r):
    rps = r.choice([35, 120, 250, 840, 1500, 47])
    return f"An API receives a steady {rps} requests per second. How many requests is that per minute?", rps * 60


def _t_disk_free(r):
    size, pct = r.choice([500, 1000, 2000, 250]), r.choice([35, 62, 80, 91])
    return f"A {size} GiB volume is {pct}% full. How many GiB are free?", round_safe(size * (100 - pct) / 100, 2)


def _t_transfer(r):
    gb, mbps = r.choice([(12, 400), (90, 250), (3, 50), (250, 1000), (40, 125)])
    return (f"You copy {gb} GB over a link that sustains {mbps} MB/s. How many seconds does the copy take? "
            f"Use 1 GB = 1000 MB."), round_safe(gb * 1000 / mbps, 2)


def _t_replicas(r):
    n, rps = r.choice([3, 4, 6, 12]), r.choice([180, 450, 900, 1250])
    return (f"A deployment has {n} replicas, each able to handle {rps} requests per second. "
            f"What is the total capacity in requests per second?"), n * rps


def _t_pods(r):
    cores, reserved, req = r.choice([(4, 350, 250), (8, 600, 500), (16, 1100, 750), (2, 300, 200), (32, 1500, 1250)])
    mem_gb, mem_res, mem_req = r.choice([(16, 1.5, 1.0), (32, 2.0, 2.5), (64, 4.0, 3.0), (8, 1.0, 0.5)])
    cpu_fit = (cores * 1000 - reserved) // req
    mem_fit = math.floor((mem_gb - mem_res) / mem_req)
    return (f"A Kubernetes node has {cores} CPU cores and {mem_gb} GiB of memory. The system reserves "
            f"{reserved} millicores of CPU and {mem_res} GiB of memory. Each pod requests {req} millicores "
            f"of CPU and {mem_req} GiB of memory. How many pods can be scheduled on the node?"), min(cpu_fit, mem_fit)


def _t_error_budget(r):
    slo = r.choice(["99.9", "99.95", "99.5", "99.99", "99.0"])
    days = r.choice([30, 28])
    mins = round_safe((1 - Decimal(slo) / 100) * days * 24 * 60, 1)
    return (f"A service has an availability SLO of {slo}% measured over a {days}-day window. "
            f"How many minutes of downtime does the error budget allow? Round to 1 decimal place."), mins


def _t_backoff(r):
    base, n, cap = r.choice([(100, 5, 1000), (200, 6, 5000), (50, 8, 2000), (250, 4, 10000), (100, 7, 3000)])
    total = sum(min(base * 2**i, cap) for i in range(n))
    return (f"A client retries a failed request up to {n} times with exponential backoff: it waits {base} ms "
            f"before the first retry and doubles the wait each time, but no single wait exceeds {cap} ms. "
            f"If every attempt fails, how many milliseconds does it spend waiting in total?"), total


def _t_fanout(r):
    p, n = r.choice([(1, 10), (1, 50), (0.5, 100), (2, 20), (5, 8)])
    prob = round_safe((1 - (1 - p / 100) ** n) * 100, 2)
    return (f"Each backend independently responds slower than its p{100 - p:g} latency {p}% of the time. "
            f"A request fans out to {n} backends in parallel and must wait for all of them. "
            f"What is the probability, in percent, that the request hits at least one slow backend? "
            f"Round to 2 decimal places."), prob


def _t_little(r):
    lam, w = r.choice([(1200, 85), (450, 230), (3000, 40), (75, 1800), (800, 125)])
    return (f"A service receives {lam} requests per second and the average time a request spends in the "
            f"system is {w} ms. Using Little's law, how many requests are in the system on average? "
            f"Round to 1 decimal place."), round_safe(Decimal(lam) * Decimal(w) / 1000, 1)


def _t_kafka(r):
    parts, cons = r.choice([(12, 5), (24, 7), (10, 4), (6, 4), (32, 6)])
    return (f"A Kafka topic has {parts} partitions consumed by a consumer group of {cons} consumers. "
            f"Partitions are spread as evenly as possible. What is the largest number of partitions "
            f"any single consumer is assigned?"), math.ceil(parts / cons)


def _t_headroom(r):
    peak, per, util = r.choice([(9000, 600, 70), (25000, 1500, 60), (4200, 350, 75), (1800, 220, 65)])
    return (f"Peak traffic is {peak} requests per second. One instance can handle {per} requests per second, "
            f"but you only run instances at up to {util}% of that capacity. "
            f"What is the minimum number of instances needed?"), math.ceil(peak / (per * util / 100))


INFRA: list[Template] = [
    ("easy", _t_logs_gb), ("easy", _t_rpm), ("easy", _t_disk_free), ("easy", _t_transfer), ("easy", _t_replicas),
    ("hard", _t_pods), ("hard", _t_error_budget), ("hard", _t_backoff), ("hard", _t_fanout),
    ("hard", _t_little), ("hard", _t_kafka), ("hard", _t_headroom),
]

# =========================================================================== finance

D = Decimal


def _f_percent(r):
    pct, amt = r.choice([12, 7.5, 15, 22, 3]), r.choice([480, 1250, 86, 3999, 640])
    return f"What is {pct}% of ${amt}? Round to the nearest cent.", round_safe(D(str(pct)) * amt / 100, 2)


def _f_simple_interest(r):
    p, rate, yrs = r.choice([1000, 2500, 12000, 750]), r.choice([4, 5.5, 3.25, 6]), r.choice([2, 3, 5])
    return (f"A loan of ${p} accrues simple interest at {rate}% per year. How much interest is owed after "
            f"{yrs} years? Round to the nearest cent."), round_safe(D(p) * D(str(rate)) / 100 * yrs, 2)


def _f_tax(r):
    price, tax = r.choice([19.99, 249.0, 64.5, 1299.0, 8.75]), r.choice([8.875, 7.25, 6, 9.5])
    return (f"An item costs ${price:.2f} before sales tax. Sales tax is {tax}%. What is the total price "
            f"including tax? Round to the nearest cent."), round_safe(D(str(price)) * (1 + D(str(tax)) / 100), 2)


def _f_discount(r):
    price, off = r.choice([80, 149.99, 59.5, 420]), r.choice([15, 25, 30, 40])
    return (f"A product priced at ${price:.2f} is discounted by {off}%. What is the sale price? "
            f"Round to the nearest cent."), round_safe(D(str(price)) * (100 - off) / 100, 2)


def _f_split(r):
    total, people = r.choice([(184.6, 4), (92.0, 5), (310.5, 6), (57.3, 3)])
    return (f"A bill of ${total:.2f} is split evenly between {people} people. How much does each person pay? "
            f"Round to the nearest cent."), round_safe(D(str(total)) / people, 2)


def _f_compound(r):
    p, rate, yrs, n = r.choice([(10000, 5, 10, 12), (2500, 4.2, 3, 4), (5000, 7, 5, 12), (1200, 3.5, 8, 1),
                                (800, 6, 20, 365)])
    amt = D(p) * (1 + D(str(rate)) / 100 / n) ** (n * yrs)
    freq = {12: "monthly", 4: "quarterly", 1: "annually", 365: "daily"}[n]
    return (f"${p} is invested at {rate}% annual interest, compounded {freq}, for {yrs} years. "
            f"What is the final balance? Round to the nearest cent."), round_safe(amt, 2)


def _f_margin_call(r):
    n, price, init, maint = r.choice([(100, 50, 50, 25), (200, 80, 50, 30), (500, 12, 60, 25), (300, 40, 50, 35)])
    loan = D(n) * D(price) * (1 - D(init) / 100)
    call = loan / (D(n) * (1 - D(maint) / 100))
    return (f"You buy {n} shares at ${price} each using margin with {init}% initial margin (you borrow the rest). "
            f"The maintenance margin is {maint}%. Ignoring interest, below what share price do you get a margin "
            f"call? Round to the nearest cent."), round_safe(call, 2)


def _f_avg_cost(r):
    n1, p1, n2, p2 = r.choice([(100, 42.10, 50, 38.75), (30, 210.0, 45, 185.5), (12, 99.99, 8, 120.0),
                               (250, 7.31, 400, 6.12)])
    avg = (D(n1) * D(str(p1)) + D(n2) * D(str(p2))) / (n1 + n2)
    return (f"You buy {n1} shares at ${p1} and later {n2} more shares at ${p2}. What is your average cost per "
            f"share? Round to the nearest cent."), round_safe(avg, 2)


def _f_amortization(r):
    p, rate, yrs = r.choice([(300000, 6.5, 30), (25000, 7, 5), (18000, 4.9, 6), (450000, 5.75, 15)])
    i, n = D(str(rate)) / 100 / 12, yrs * 12
    pay = D(p) * i / (1 - (1 + i) ** -n)
    return (f"A ${p} loan at {rate}% annual interest is repaid in equal monthly payments over {yrs} years "
            f"(interest compounds monthly). What is the monthly payment? Round to the nearest cent."), round_safe(pay, 2)


def _f_up_down(r):
    up, down = r.choice([(20, 20), (50, 40), (10, 15), (25, 25), (30, 10)])
    net = ((1 + D(up) / 100) * (1 - D(down) / 100) - 1) * 100
    return (f"A stock rises {up}% one day and falls {down}% the next. What is the net percentage change over "
            f"the two days? Use a negative number for a loss and round to 2 decimal places."), round_safe(net, 2)


def _f_bps_fee(r):
    notional, bps = r.choice([(1250000, 7.5), (84300, 12), (5600000, 2.5), (330000, 18)])
    return (f"A broker charges {bps} basis points on trade notional. What is the fee on a ${notional} trade? "
            f"Round to the nearest cent."), round_safe(D(notional) * D(str(bps)) / 10000, 2)


FINANCE: list[Template] = [
    ("easy", _f_percent), ("easy", _f_simple_interest), ("easy", _f_tax), ("easy", _f_discount), ("easy", _f_split),
    ("hard", _f_compound), ("hard", _f_margin_call), ("hard", _f_avg_cost), ("hard", _f_amortization),
    ("hard", _f_up_down), ("hard", _f_bps_fee),
]


def _instantiate(templates: list[Template], per_easy: int, per_hard: int, rng: random.Random, cat: str):
    items, dropped = [], 0
    for ti, (diff, gen) in enumerate(templates):
        want, seen, tries = (per_easy if diff == "easy" else per_hard), set(), 0
        while len([1 for it in items if it["template"] == f"{cat}-{ti}"]) < want and tries < 50:
            tries += 1
            try:
                q, gold = gen(rng)
            except NearTie:
                dropped += 1
                continue
            if q in seen:
                continue
            seen.add(q)
            items.append({"template": f"{cat}-{ti}", "difficulty": diff, "question": q, "gold": fmt(gold)})
    return items, dropped


# ============================================================================ build


def build_backend_set(seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    dropped = {"python": 0, "sql": 0, "infra": 0, "finance": 0}

    for diff, codes in (("easy", PY_EASY), ("hard", PY_HARD)):
        for code in codes:
            try:
                gold = run_python(code)
            except Exception:
                dropped["python"] += 1
                continue
            items.append({"category": "python", "difficulty": diff, "question": code,
                          "prompt": python_prompt(code), "gold": gold, "answer_type": "exact"})

    for diff, queries in (("easy", SQL_EASY), ("hard", SQL_HARD)):
        for q in queries:
            try:
                gold, at = run_sql(q)
            except Exception:
                dropped["sql"] += 1
                continue
            items.append({"category": "sql", "difficulty": diff, "question": q,
                          "prompt": sql_prompt(q), "gold": gold, "answer_type": at})

    for cat, templates, pe, ph in (("infra", INFRA, 5, 3), ("finance", FINANCE, 5, 3)):
        gen, nd = _instantiate(templates, pe, ph, rng, cat)
        dropped[cat] += nd
        for it in gen:
            items.append({"category": cat, "difficulty": it["difficulty"], "question": it["question"],
                          "prompt": f"{it['question']}\n\n{ANSWER_FOOTER_NUMBER}", "gold": it["gold"],
                          "answer_type": "number", "template": it["template"]})

    for i, it in enumerate(items):
        it.update({"id": f"backend-{it['category']}-{i:03d}", "dataset": "backend", "split": "test"})
        it.setdefault("template", "")
    return items, dropped
