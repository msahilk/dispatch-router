"""Harder backend items ("stress" tier).

The first backend set turned out to be too easy: both models scored 98%, and every
classic Python/SQL gotcha was answered correctly. These items are built to need
real work instead of recall: tracking state through loops, exact big-integer and
bit arithmetic, window functions over a larger table, and multi-step math with
awkward numbers. Golds are still computed by execution / formula, never typed.
"""

from __future__ import annotations

import math
import random
import sqlite3
import textwrap
from decimal import Decimal as D

from .backend_set import NearTie, round_safe

# ============================================================== python generators
# Each returns source code; the gold is whatever it prints (verified by running it).


def _py_state_loop(r):
    xs = [r.randint(1, 30) for _ in range(9)]
    return textwrap.dedent(f"""\
        xs = {xs}
        acc = 0
        for i, x in enumerate(xs):
            if x % 3 == 0:
                acc += x * i
            elif x % 2:
                acc -= x
            else:
                acc ^= x
        print(acc)""")


def _py_sort_key(r):
    pool = ["kafka", "redis", "nginx", "envoy", "etcd", "consul", "vault", "istio", "linkerd", "jaeger",
            "grafana", "loki", "tempo", "argo", "helm"]
    words = r.sample(pool, 8)
    return f"words = {words}\nprint(sorted(words, key=lambda w: (len(w) % 3, w[-1])))"


def _py_dict_agg(r):
    pool = "alpha beta gamma delta epsilon zeta theta iota kappa lambda sigma tau omega".split()
    text = " ".join(r.choice(pool) for _ in range(12))
    return textwrap.dedent(f"""\
        text = "{text}"
        d = {{}}
        for w in text.split():
            d[w[0]] = d.get(w[0], 0) + len(w)
        print(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:3])""")


def _py_bigint(r):
    b, e = r.choice([3, 7, 11, 13]), r.randint(29, 41)
    return f"print({b} ** {e})"


def _py_modpow(r):
    b, e, m = r.randint(2, 50), r.randint(20, 60), r.choice([97, 1009, 10007, 65537])
    return f"print(pow({b}, {e}, {m}) + {b} ** 3 % {m})"


def _py_bits(r):
    x = r.randint(1000, 60000)
    s = r.randint(2, 5)
    return f"x = {x}\nprint(bin((x ^ (x >> {s})) & 0xFF), x & -x, x.bit_count())"


def _py_recursion(r):
    n, a, b = r.randint(12, 16), r.randint(1, 3), r.randint(2, 4)
    return textwrap.dedent(f"""\
        def f(n):
            return n if n < 3 else {a} * f(n - 1) - f(n - 2) + {b} * f(n - 3)

        print(f({n}))""")


def _py_collatz(r):
    n = r.randint(27, 200)
    return textwrap.dedent(f"""\
        n, steps, peak = {n}, 0, {n}
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
            peak = max(peak, n)
        print(steps, peak)""")


def _py_slices(r):
    k = r.randint(2, 4)
    return textwrap.dedent(f"""\
        a = list(range(12))
        a[2:5] = [a[-1]] * {k}
        del a[::3]
        a[1::2] = [x * 10 for x in a[1::2]]
        print(a)""")


def _py_accumulate(r):
    xs = [r.randint(1, 9) for _ in range(7)]
    return f"from itertools import accumulate\nprint(list(accumulate({xs}, lambda a, b: a * 2 - b)))"


def _py_strings(r):
    s = r.choice(["distributed-systems", "eventual_consistency", "idempotency-keys", "backpressure_signal"])
    a, b = r.randint(1, 4), r.randint(2, 3)
    return f's = "{s}"\nprint(s[{a}:-{b}:2].upper().replace("E", "3")[::-1])'


def _py_generator(r):
    n, k = r.randint(8, 15), r.randint(2, 4)
    return textwrap.dedent(f"""\
        def gen(n):
            total = 0
            for i in range(n):
                got = yield total
                total += i if got is None else got

        g = gen({n})
        out = [next(g)]
        for i in range({n} - 1):
            out.append(g.send(i * {k}) if i % 2 else next(g))
        print(out[-1], sum(out))""")


PY_STRESS = [_py_state_loop, _py_sort_key, _py_dict_agg, _py_bigint, _py_modpow, _py_bits, _py_recursion,
             _py_collatz, _py_slices, _py_accumulate, _py_strings, _py_generator]

# ===================================================================== sql: sales db


def sales_schema(seed: int = 7) -> str:
    r = random.Random(seed)
    reps = {"north": ["ana", "ben"], "south": ["cai", "dee", "eli"], "west": ["fay", "gus"]}
    products = {"widget": 12.5, "gadget": 30.0, "gizmo": 7.25, "doohickey": 55.0}
    rows = []
    for i in range(1, 31):
        region = r.choice(list(reps))
        rep = r.choice(reps[region])
        prod = r.choice(list(products))
        qty = r.randint(1, 12)
        price = products[prod] if r.random() > 0.2 else round(products[prod] * 0.9, 2)
        month, day = r.randint(1, 4), r.randint(1, 28)
        rows.append(f"  ({i}, '{region}', '{rep}', '{prod}', {qty}, {price}, '2026-{month:02d}-{day:02d}')")
    return (
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, rep TEXT, product TEXT,\n"
        "                    qty INTEGER, unit_price REAL, sold_on TEXT);\n"
        "INSERT INTO sales VALUES\n" + ",\n".join(rows) + ";"
    )


SQL_STRESS = [
    "SELECT ROUND(SUM(qty * unit_price), 2) FROM sales WHERE region = 'south';",
    "SELECT rep FROM sales GROUP BY rep ORDER BY SUM(qty * unit_price) DESC, rep LIMIT 1;",
    "SELECT COUNT(*) FROM sales WHERE qty * unit_price > (SELECT AVG(qty * unit_price) FROM sales);",
    "SELECT product FROM sales GROUP BY product ORDER BY SUM(qty) DESC, product LIMIT 1;",
    "SELECT strftime('%m', sold_on) AS m FROM sales GROUP BY m ORDER BY SUM(qty * unit_price) DESC, m LIMIT 1;",
    "SELECT COUNT(DISTINCT rep) FROM sales WHERE unit_price < 10;",
    "SELECT ROUND(AVG(qty), 2) FROM sales WHERE product = 'gadget';",
    "SELECT id FROM (SELECT id, SUM(qty) OVER (ORDER BY sold_on, id) AS running FROM sales) "
    "WHERE running >= 100 ORDER BY running LIMIT 1;",
    "SELECT ROUND(MAX(rev) - MIN(rev), 2) FROM (SELECT region, SUM(qty * unit_price) AS rev FROM sales GROUP BY region);",
    "SELECT COUNT(*) FROM (SELECT rep FROM sales GROUP BY rep HAVING COUNT(DISTINCT product) >= 3);",
    "SELECT rep FROM (SELECT rep, region, RANK() OVER (PARTITION BY region ORDER BY SUM(qty) DESC) AS rk "
    "FROM sales GROUP BY rep, region) WHERE region = 'south' AND rk = 1;",
    "SELECT MAX(gap) FROM (SELECT julianday(sold_on) - julianday(LAG(sold_on) OVER (ORDER BY sold_on, id)) AS gap "
    "FROM sales);",
    "SELECT COUNT(*) FROM sales a JOIN sales b ON a.rep = b.rep AND a.id < b.id AND a.product = b.product;",
    "SELECT ROUND(SUM(CASE WHEN unit_price < (SELECT MAX(unit_price) FROM sales s2 WHERE s2.product = s1.product) "
    "THEN qty ELSE 0 END) * 1.0 / SUM(qty), 4) FROM sales s1;",
    "SELECT region FROM sales GROUP BY region HAVING COUNT(*) = (SELECT MIN(c) FROM "
    "(SELECT COUNT(*) AS c FROM sales GROUP BY region));",
    "SELECT SUM(qty) FROM sales WHERE sold_on BETWEEN '2026-02-01' AND '2026-03-15';",
    "SELECT COUNT(*) FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY product ORDER BY qty DESC, id) AS rn "
    "FROM sales) WHERE rn <= 2;",
    "SELECT ROUND(SUM(qty * unit_price) / COUNT(DISTINCT strftime('%m', sold_on)), 2) FROM sales;",
    "SELECT id FROM sales ORDER BY qty * unit_price DESC, id LIMIT 1 OFFSET 2;",
    "SELECT COUNT(*) FROM sales WHERE rep IN (SELECT rep FROM sales GROUP BY rep HAVING SUM(qty) > 25);",
    "SELECT product || ':' || COUNT(*) FROM sales WHERE region = 'west' GROUP BY product "
    "ORDER BY COUNT(*) DESC, product LIMIT 1;",
    "SELECT ROUND(AVG(daily), 2) FROM (SELECT sold_on, SUM(qty) AS daily FROM sales GROUP BY sold_on);",
    "SELECT COUNT(*) FROM (SELECT rep, LAG(qty) OVER (PARTITION BY rep ORDER BY sold_on, id) AS prev, qty "
    "FROM sales) WHERE qty > prev;",
    "SELECT SUM(qty) FROM sales WHERE region = 'north' AND (product = 'widget' OR product = 'gizmo') AND qty > 4;",
    "SELECT MIN(id) FROM sales WHERE qty * unit_price = (SELECT MAX(qty * unit_price) FROM sales WHERE region = 'north');",
]

# ================================================================== infra (stress)


def _ti_p95(r):
    n = r.choice([20, 25, 30])
    lat = [r.randint(12, 400) for _ in range(n)]
    s = sorted(lat)
    return (f"These are {n} request latencies in milliseconds: {lat}. Using the nearest-rank method "
            f"(sort ascending and take the value at position ceil(0.95 * n), counting from 1), what is the p95 "
            f"latency?"), s[math.ceil(0.95 * n) - 1]


def _ti_chain_availability(r):
    k = r.randint(3, 5)
    avs = [r.choice(["99.9", "99.95", "99.5", "99.99", "99.8"]) for _ in range(k)]
    red = r.choice([2, 3])
    idx = r.randrange(k)
    a = D(1)
    for i, v in enumerate(avs):
        p = D(v) / 100
        a *= (1 - (1 - p) ** red) if i == idx else p
    desc = ", ".join(f"{v}%" for v in avs)
    return (f"A request passes through {k} components in series with availabilities {desc}. Component number "
            f"{idx + 1} is deployed as {red} independent redundant copies (the component is up if any copy is up). "
            f"What is the end-to-end availability in percent? Round to 4 decimal places."), round_safe(a * 100, 4)


def _ti_token_bucket(r):
    cap, rate = r.randint(5, 12), r.randint(1, 4)
    times = sorted(r.randint(0, 10) for _ in range(r.randint(18, 30)))
    tokens, last, rejected = float(cap), 0, 0
    for t in times:
        tokens = min(cap, tokens + (t - last) * rate)
        last = t
        if tokens >= 1:
            tokens -= 1
        else:
            rejected += 1
    return (f"A token-bucket rate limiter starts full with capacity {cap} tokens and refills {rate} tokens per "
            f"second continuously (never above capacity). Each request takes 1 token or is rejected if fewer than "
            f"1 token is available. Requests arrive at these times in seconds: {times}. How many requests are "
            f"rejected?"), rejected


def _ti_drain(r):
    backlog, lam, mu, c = r.randint(20000, 90000), r.randint(200, 900), r.randint(60, 150), r.randint(8, 20)
    if mu * c <= lam:
        mu = lam // c + r.randint(20, 60)
    return (f"A queue has a backlog of {backlog} messages. New messages arrive at {lam} per second. There are {c} "
            f"consumers, each processing {mu} messages per second. How many minutes until the backlog is cleared? "
            f"Round to 2 decimal places."), round_safe(D(backlog) / (mu * c - lam) / 60, 2)


def _ti_cloud_cost(r):
    n, hourly = r.randint(12, 80), r.choice(["0.192", "0.384", "0.0832", "1.008", "0.226"])
    reserved_pct, disc = r.choice([40, 60, 75]), r.choice([31, 38, 42])
    hours = 730
    n_res = math.floor(n * reserved_pct / 100)
    cost = D(n - n_res) * D(hourly) * hours + D(n_res) * D(hourly) * hours * (100 - disc) / 100
    return (f"You run {n} instances 24/7 at ${hourly} per hour on demand. {reserved_pct}% of the instances "
            f"(rounded down to a whole number) are covered by reservations that are {disc}% cheaper. Using 730 "
            f"hours per month, what is the monthly bill in dollars? Round to the nearest cent."), round_safe(cost, 2)


def _ti_weighted_latency(r):
    k = r.randint(3, 4)
    shares = [r.randint(5, 50) for _ in range(k)]
    tot = sum(shares)
    lats = [r.randint(20, 300) for _ in range(k)]
    desc = "; ".join(f"region {chr(65 + i)}: {shares[i]} req/s at {lats[i]} ms" for i in range(k))
    avg = D(sum(s * l for s, l in zip(shares, lats))) / tot
    return (f"Traffic by region: {desc}. What is the traffic-weighted mean latency in ms across all requests? "
            f"Round to 2 decimal places."), round_safe(avg, 2)


def _ti_timeout_budget(r):
    t, tries, base = r.choice([800, 1200, 1500, 2000]), r.randint(3, 5), r.choice([50, 100, 150])
    total = sum(t + (base * 2 ** i if i < tries - 1 else 0) for i in range(tries))
    return (f"A client makes up to {tries} attempts. Each attempt times out after {t} ms. Between attempts it waits "
            f"{base} ms, doubling the wait after each retry. If every attempt times out, how many milliseconds pass "
            f"from the first attempt until the client gives up?"), total


def _ti_shard_skew(r):
    keys, shards, hot = r.randint(200000, 900000), r.choice([8, 12, 16, 24]), r.choice([18, 22, 30])
    per_other = D(keys) * (100 - hot) / 100 / (shards - 1)
    hot_keys = D(keys) * hot / 100
    return (f"{keys} keys are spread over {shards} shards. One hot shard holds {hot}% of all keys and the rest are "
            f"split evenly over the other shards. How many times larger is the hot shard than a typical other shard? "
            f"Round to 3 decimal places."), round_safe(hot_keys / per_other, 3)


INFRA_STRESS = [_ti_p95, _ti_chain_availability, _ti_token_bucket, _ti_drain, _ti_cloud_cost,
                _ti_weighted_latency, _ti_timeout_budget, _ti_shard_skew]

# ================================================================ finance (stress)


def _tf_remaining_balance(r):
    p, rate, yrs = r.choice([180000, 240000, 32000, 410000, 27500]), r.choice(["4.25", "5.9", "6.75", "3.6"]), \
        r.choice([10, 15, 20, 30])
    k = r.randint(12, 60)
    i, n = D(rate) / 1200, yrs * 12
    pay = D(p) * i / (1 - (1 + i) ** -n)
    bal = D(p) * (1 + i) ** k - pay * ((1 + i) ** k - 1) / i
    return (f"A ${p} loan at {rate}% annual interest (compounded monthly) is repaid in equal monthly payments over "
            f"{yrs} years. What is the remaining balance right after payment number {k}? Round to the nearest cent."
            ), round_safe(bal, 2)


def _tf_contributions(r):
    p, c, rate, yrs = r.randint(1000, 20000), r.choice([100, 250, 500, 800]), r.choice(["5", "6.5", "7.2", "4.4"]), \
        r.randint(5, 25)
    i, n = D(rate) / 1200, yrs * 12
    fv = D(p) * (1 + i) ** n + D(c) * ((1 + i) ** n - 1) / i
    return (f"You invest ${p} today and add ${c} at the end of every month. The account earns {rate}% annual "
            f"interest compounded monthly. What is the balance after {yrs} years? Round to the nearest cent."
            ), round_safe(fv, 2)


def _tf_fifo(r):
    lots = [(r.randint(10, 60), round(r.uniform(20, 120), 2)) for _ in range(3)]
    total = sum(q for q, _ in lots)
    sell_q, sell_p = r.randint(total // 3, total - 5), round(r.uniform(40, 150), 2)
    left, basis = sell_q, D(0)
    for q, px in lots:
        take = min(q, left)
        basis += take * D(str(px))
        left -= take
    gain = sell_q * D(str(sell_p)) - basis
    desc = ", then ".join(f"{q} shares at ${px:.2f}" for q, px in lots)
    return (f"You buy {desc}. Later you sell {sell_q} shares at ${sell_p:.2f}. Using FIFO cost basis, what is the "
            f"realized gain in dollars (negative for a loss)? Round to the nearest cent."), round_safe(gain, 2)


def _tf_fx_chain(r):
    amt, r1, r2, fee = r.randint(1000, 9000), r.choice(["0.9213", "1.0847", "0.7861"]), \
        r.choice(["151.37", "7.2614", "1.3528"]), r.choice(["0.35", "0.5", "1.2"])
    a = (D(amt) * (1 - D(fee) / 100) * D(r1)).quantize(D("0.01"), rounding="ROUND_HALF_UP")
    b = a * (1 - D(fee) / 100) * D(r2)
    return (f"You convert {amt} units of currency A to currency B at rate {r1} B per A, then convert all of it to "
            f"currency C at {r2} C per B. Each conversion charges a {fee}% fee taken from the amount before "
            f"converting. The intermediate amount is rounded to 2 decimals. How much currency C do you end with? "
            f"Round to 2 decimal places."), round_safe(b, 2)


def _tf_daily_accrual(r):
    rate = r.choice(["4.5", "3.75", "5.25"])
    bal = r.randint(5000, 20000)
    events = sorted(r.sample(range(2, 30), 3))
    deltas = [r.choice([-1, 1]) * r.randint(500, 4000) for _ in events]
    interest, b, ev = D(0), D(bal), dict(zip(events, deltas))
    for day in range(1, 31):
        if day in ev:
            b += ev[day]
        interest += b * D(rate) / 100 / 365
    desc = "; ".join(f"on day {d} the balance changes by {'+' if x > 0 else '-'}${abs(x)}" for d, x in ev.items())
    return (f"An account starts the month (day 1) with ${bal}. Interest accrues daily at {rate}% / 365 on each day's "
            f"closing balance, for days 1 to 30. {desc} (changes apply before that day's accrual). What is the total "
            f"interest for the 30 days? Round to the nearest cent."), round_safe(interest, 2)


def _tf_cagr(r):
    start, yrs = r.randint(10000, 90000), r.randint(3, 12)
    end = int(start * r.uniform(1.3, 4.0))
    cagr = ((end / start) ** (1 / yrs) - 1) * 100
    return (f"An investment grows from ${start} to ${end} over {yrs} years. What is the compound annual growth rate "
            f"in percent? Round to 3 decimal places."), round_safe(cagr, 3)


def _tf_margin_interest(r):
    shares, price, init, rate, days = r.randint(100, 900), round(r.uniform(15, 220), 2), r.choice([50, 60]), \
        r.choice(["8.5", "11.25", "9.75"]), r.randint(20, 120)
    loan = D(shares) * D(str(price)) * (100 - init) / 100
    interest = loan * D(rate) / 100 * days / 360
    return (f"You buy {shares} shares at ${price:.2f} with {init}% initial margin, borrowing the rest. The broker "
            f"charges {rate}% annual simple interest on the loan using a 360-day year. How much interest do you owe "
            f"after {days} days? Round to the nearest cent."), round_safe(interest, 2)


def _tf_payroll_rounding(r):
    hours = [round(r.uniform(6, 11), 2) for _ in range(5)]
    wage, ot = round(r.uniform(18, 65), 2), r.choice(["1.5", "2"])
    reg = sum(min(h, 8) for h in hours)
    over = sum(max(0, h - 8) for h in hours)
    pay = D(str(round(reg, 2))) * D(str(wage)) + D(str(round(over, 2))) * D(str(wage)) * D(ot)
    return (f"An employee works these hours over 5 days: {hours}. Pay is ${wage:.2f} per hour; any hours beyond 8 in "
            f"a single day are paid at {ot}x. What is the gross pay for the week? Round to the nearest cent."
            ), round_safe(pay, 2)


FINANCE_STRESS = [_tf_remaining_balance, _tf_contributions, _tf_fifo, _tf_fx_chain, _tf_daily_accrual, _tf_cagr,
                  _tf_margin_interest, _tf_payroll_rounding]


def generate_math(gens, per: int, rng: random.Random):
    """Up to `per` unique instances per generator; drops rounding ties."""
    out, dropped = [], 0
    for gi, gen in enumerate(gens):
        seen, tries = set(), 0
        while len(seen) < per and tries < 60:
            tries += 1
            try:
                q, gold = gen(rng)
            except NearTie:
                dropped += 1
                continue
            if q not in seen:
                seen.add(q)
                out.append((f"{gen.__name__}", q, gold))
    return out, dropped
