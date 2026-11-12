# CRM ticket intelligence

Live demo: https://crm-ticket-intelligence.vercel.app

Personal project. The problem is the one a CRM helpdesk has (a Zoho Desk-like problem): read an incoming complaint, route it to a product line, and keep an eye on whether the mix of tickets still looks like the data the model was trained on.

This is not a Zoho product, not an internship deliverable, and not affiliated with Zoho. No Zoho data was used.

## Problem

A new ticket has a free-text narrative plus a few structured fields (where the person lives, how they tagged themselves, when it arrived). The product line is what a queue would use to assign the ticket. Issue and company are tempting features and were left out, because they mostly give the answer away.

The label is the `Product` value on the complaint. See `label_definition` in `reports/metrics.json`.

## Data

Source: the public CFPB Consumer Complaint Database, via the parquet mirror [`vivekkopthsd/cfpb-consumer-complaints`](https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints) (bulk export as of 2026-08-11, 17,021,062 rows). The file is a US federal government work (public domain). The dataset card lists CC0-1.0.

The live CFPB download dropped complaint narratives in September 2026, so this project pins that August mirror instead of the daily file at [consumerfinance.gov](https://www.consumerfinance.gov/data-research/consumer-complaints/).

The committed extract is `data/sample_complaints.csv.gz` (sha256 `db7e86ef73872d4338fcd9c93ab72a67419128400ad91e81a59ca2e04865cc0a`). Filters, from `data/universe_stats.json`:

- `date_received` from 2024-01-01 up to 2024-04-01
- narrative at least 40 characters
- product does not start with `Credit reporting` (109,166 narrative rows in this window, a different dispute workflow and most of the text)
- product is not `Debt or credit management` (311 narrative rows)

That leaves a universe of 78,200 complaints, of which 43,124 have a long enough narrative. Credit reporting aside, a narrative is optional: 34,951 universe rows have no text (missing-text rate 0.446944).

Before training, exact duplicate narratives are collapsed to the earliest complaint id. 38,758 rows remain: 24,851 in January-February (train) and 13,907 in March (test).

`scripts/build_sample.py` rebuilds the extract from the parquet (`CFPB_PARQUET=...`). It needs `duckdb`, which is not required to train or serve the model.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ticketintel.train
pytest
ruff check .
uvicorn app.main:app --port 8000
```

Training overwrites `reports/metrics.json`, `reports/data_quality.json`, `reports/monitor.json`, and `models/winner.joblib`. The numbers below are from the committed metrics file, not from a second run.

```bash
curl -s localhost:8000/health
curl -s localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"The bank charged two overdraft fees on my checking account.","state":"TX","tags":"Servicemember","date_received":"2024-03-15"}'
```

`POST /predict` takes `text` plus optional `state`, `tags`, and `date_received` (`YYYY-MM-DD`). `GET /health` reports whether the saved model loaded.

## Results

March 2024 holdout. Macro-F1 is the selection metric. ROC-AUC is one-vs-rest, macro averaged. The linear SVM cell is null because that model does not produce probabilities.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OVR macro) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.824693 | 0.764021 | 0.802239 | 0.779676 | 0.972869 |
| Linear SVM | 0.820163 | 0.768907 | 0.773052 | 0.770604 | null |
| Random forest | 0.775077 | 0.761075 | 0.683213 | 0.712295 | 0.952403 |
| Complement naive Bayes | 0.747825 | 0.783655 | 0.629042 | 0.651511 | 0.941712 |

Winner: logistic regression. Notes on the comparison are in `docs/experiment_log.md`.

Per-class F1 for the winner, from the same file:

| Product | F1 |
| --- | ---: |
| Debt collection | 0.902305 |
| Student loan | 0.901252 |
| Mortgage | 0.885154 |
| Credit card | 0.81988 |
| Checking or savings account | 0.803181 |
| Prepaid card | 0.724138 |
| Vehicle loan or lease | 0.720513 |
| Money transfer, virtual currency, or money service | 0.690871 |
| Payday loan, title loan, personal loan, or advance loan | 0.569787 |

## Data quality

From `reports/data_quality.json`:

- Missing text rate on the date-and-product universe, before the 40-character cut: 0.446944 (34,951 of 78,200).
- Complaint-id duplicate rate on the modeling extract: 0.0.
- Exact duplicate narrative rate on the modeling extract: 0.101243. Those copies are collapsed before the split.
- Class balance is uneven. Debt collection is 0.349597 of the 43,124 modeling rows. Prepaid card is 0.024812.
- Every narrative in the extract was submitted via Web, so channel is not a feature.

Weak slice: payday / title / personal / advance loan complaints in March. 514 test rows, accuracy 0.675097, overall accuracy 0.824693, gap 0.149596. On a one-product slice that accuracy is the recall of the product. The slice rule is fixed in the training script: lowest accuracy among predefined slices with at least 200 test rows.

`reports/monitor.json` compares the model's predicted mix on March with the January-February label mix. Total variation is 0.05148, under the 0.15 flag. That check is a prompt to look, not a rollback.

## Limitations

- Narratives are opt-in. They are not a random sample of complaints, and the bureau scrubs identifiers (the `XXXX` tokens are theirs).
- Credit-reporting complaints are out of scope. A model trained here will not route that queue.
- Company name would raise accuracy and would also shortcut the text. It is not in the model.
- Probabilities are not calibrated. A 0.7 is not a measured 70 percent.
- The March monitor batch is the same holdout used to pick the model, so it is not an independent drift check.
- No claim that this would pass a real desk's review bar. The payday-loan slice is the obvious place it would not.

## What is next

- Calibrate scores and send low-margin payday and money-transfer tickets to a person.
- Predict issue inside debt collection and credit card, where product routing is already decent.
- Score a later month that was not used to pick the winner.

Code in this repo is MIT. The complaint text stays under the CFPB public-domain terms.
