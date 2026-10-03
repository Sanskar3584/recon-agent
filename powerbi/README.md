# Building the Power BI dashboard

This takes about 30 minutes. Power BI Desktop is free from the Microsoft Store.

## 1. Produce the data

```
venv\Scripts\activate
python -m recon.pipeline --labeler hybrid-ollama     (or --labeler rules)
python -m recon.evaluate --months 12 --labeler rules --out output/eval_rules
python -m recon.evaluate --months 12 --labeler ollama --out output/eval_ollama
python -m recon.evaluate --months 12 --labeler hybrid-ollama --out output/eval_hybrid
```

## 2. Load the files

In Power BI Desktop, use **Get data > Text/CSV** for each of these, and click **Load**:

| File | One row per | Used for |
|---|---|---|
| `output/lines.csv` | book or bank line | match rate, status, ageing |
| `output/exceptions.csv` | exception | the review list, suggested entries |
| `output/reconciliation.csv` | statement line | the reconciliation statement |
| `evaluation.csv` in `output/eval_rules`, `output/eval_ollama` and `output/eval_hybrid` | month | rules vs LLM vs hybrid accuracy |

For the three evaluation files: in **Transform data**, select one and choose **Append queries > Three or more tables** to add the other two. Rename the result `evaluation`.

To compare the payee memory too, add the `evaluation.csv` files from `output/eval_rules_memory` and `output/eval_hybrid_memory` to the same append (the commands that produce them are in the main README).

Check the column types in **Transform data**: `amount`, `abs_amount` and `adjustment` should be Decimal number, `date` and `line_date` Date, and `age_days` Whole number.

## 3. Add the measures

Select the `lines` table, choose **New measure**, and add these one at a time:

```
Lines = COUNTROWS(lines)
Matched Lines = CALCULATE(COUNTROWS(lines), lines[status] = "MATCHED")
Auto-match % = DIVIDE([Matched Lines], [Lines])
Open Exceptions = CALCULATE(COUNTROWS(exceptions), exceptions[status] = "PENDING_APPROVAL")
Exception Value = SUM(exceptions[abs_amount])
Needs Review = CALCULATE(COUNTROWS(exceptions), exceptions[category] = "NEEDS_REVIEW")
Difference = CALCULATE(SUM(reconciliation[amount]), reconciliation[line] = "Difference")
Label Accuracy % = DIVIDE(SUMX(evaluation, evaluation[exceptions] * evaluation[entry_accuracy_pct] / 100), SUM(evaluation[exceptions]))
```

Format `Auto-match %` and `Label Accuracy %` as percentages, and the amounts as currency with 2 decimals.

## 4. Build the pages

**Page 1, Overview**
- Four **cards**: Auto-match %, Open Exceptions, Exception Value, Difference
- **Clustered bar chart**: `exceptions[category]` on the axis, count of `exception_id` as the value
- **Donut chart**: `lines[status]`, count of `line_id`
- **Table**: `reconciliation[line]` and `reconciliation[amount]`, sorted by `reconciliation[order]`

**Page 2, Exceptions**
- **Slicers**: `exceptions[category]`, `exceptions[status]`, `exceptions[age_bucket]`
- **Table**: exception_id, category, description, amount, journal_entry, explanation, confidence, status
- **Stacked column chart**: `exceptions[age_bucket]` on the axis, Exception Value as the value, `exceptions[category]` as the legend

**Page 3, Rules vs LLM vs hybrid**
- **Line chart**: `evaluation[month]` on the axis, Label Accuracy % as the value, `evaluation[labeler]` as the legend
- Three **cards** filtered by labeler, showing the 12-month accuracy of each

## 5. Save it

Save the report as `powerbi/recon_dashboard.pbix`. Then export a screenshot of page 1 to `docs/dashboard.png` and add it to the main README.
