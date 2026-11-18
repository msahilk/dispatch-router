# Resume notes (DRAFT — edit before using)

All numbers come from `results/summary.json`.

**Long bullet**
Built a calibrated LLM router that decides before each call whether Claude Haiku can handle a prompt or it needs
Claude Sonnet; on 2,500 held-out benchmark questions it matched 99% of Sonnet's accuracy at
54% of the cost (random routing: 71%),
evaluated with bootstrap CIs, a 329-question execution-verified backend-engineering test set, and
comparisons showing a logistic-regression router generalizes better than fine-tuned DistilBERT.

**Short bullet**
LLM cost router (Haiku vs Sonnet): 99% of Sonnet accuracy at 54% of cost on
2,500 held-out prompts; $3.05 total API spend.

**~300-character paragraph**
I built a router that sends each prompt to a cheap or an expensive Claude model before calling either. Trained on
4,816 labeled questions, it kept 99% of the expensive model's accuracy at 54%
of the cost, and I tested where it breaks on my own backend-engineering prompt set.
