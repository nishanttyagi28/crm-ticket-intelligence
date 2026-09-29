# Experiment log

Task: route a consumer complaint to the product line on the form, using the narrative and a few fields that would be known when the ticket arrives. Company, issue, and sub-product were not used. They mostly restate the product.

## Data cut

Complaints received from 2024-01-01 up to 2024-04-01, narrative at least 40 characters, credit-reporting products removed, and "Debt or credit management" removed. Exact duplicate narratives were collapsed to the earliest complaint id before the split.

Holdout: train on January-February 2024 (24,851 rows), test on March 2024 (13,907 rows). The selection rule was macro-F1 on that March holdout.

## Models

All four models saw the same TF-IDF narrative (word unigrams and bigrams, `min_df=4`, 5,000 features, sublinear tf) and the same structured block: state, weekday, missing date, servicemember tag, older-American tag, and scaled length. The forest did not take the sparse TF-IDF directly. It took an 80-component truncated SVD of that TF-IDF, plus the structured block.

| Model | Why it was in the run |
| --- | --- |
| Logistic regression (`saga`, C=1, balanced class weights, 200 iterations) | Linear baseline that still gives probabilities. |
| Linear SVM (C=1, balanced) | Same features, margin classifier. No probabilities, so no multiclass ROC-AUC. |
| Complement naive Bayes (alpha 0.5) | Fast text baseline. Assumes the counts are non-negative, which they are after max-abs scaling. |
| Random forest (200 trees, depth 18, min leaf 5) | Non-linear model on the reduced text. |

## Holdout

Copied from `reports/metrics.json`.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OVR macro) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.824693 | 0.764021 | 0.802239 | 0.779676 | 0.972869 |
| Linear SVM | 0.820163 | 0.768907 | 0.773052 | 0.770604 | null |
| Random forest | 0.775077 | 0.761075 | 0.683213 | 0.712295 | 0.952403 |
| Complement naive Bayes | 0.747825 | 0.783655 | 0.629042 | 0.651511 | 0.941712 |

`roc_auc_ovr_macro` is null for the linear SVM because multiclass ROC-AUC in scikit-learn wants probabilities, and `LinearSVC` only exposes a decision function.

## Winner

Logistic regression. It has the highest macro-F1 (0.779676). The linear SVM is close on accuracy (0.820163 vs 0.824693) but weaker on macro-F1, and it does not produce probabilities the API can return. The forest and naive Bayes both lose recall on the smaller products.

Naive Bayes has the highest macro precision (0.783655) and the worst macro recall (0.629042). It is not the model I would ship for routing, because a missed product line is worse than an extra review.

The saved file `models/winner.joblib` is this logistic regression, with the fitted vectorizer and structured encoders.

## Where it is weak

On the March holdout, the payday / title / personal / advance loan product is the weakest predefined slice: 514 rows, accuracy 0.675097 against an overall accuracy of 0.824693 (gap 0.149596). For a slice that is a single true product, that accuracy is the recall of the product. Its F1 is 0.569787, against 0.902305 for debt collection and 0.901252 for student loans. Those complaints often read like a credit card or a debt-collection dispute, and the narrative is not always specific about the loan type.

## What I would try next

- Calibrate the logistic probabilities before using them as a review threshold.
- A second-stage model for issue, inside the two or three products that already route well.
- Keep a later month (April 2024 is in the source file) as a fresh monitor batch instead of reusing March.
