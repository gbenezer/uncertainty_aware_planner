# Model Card: Demand and Inventory Planning

## 4.1 Model Details

| Field | Detail |
| --- | --- |
| **Project / Team** | Demand and Inventory Planning (Uncertain-Team) |
| **Date** | 10 Nov 2026 |
| **Version** | 0.0 |
| **Architecture** | 1 imputation model (DLinear or TimesNet) + 1 forecasting model (DLinear or TFT) |
| **Baseline commit** | TBD |

## 4.2 Intended Use

**Primary users:** Demand planners and inventory stockers.

**Primary uses:** Helping to plan sales, stocking activities, and other business actions based on inventory demand. Formally, the imputation model outputs a 95% confidence interval and point estimate for demand during historical stockout hours, while the forecasting model outputs a 95% confidence interval on a per-day demand basis for 7 days in the future.

**Out-of-scope uses:**

- Dynamic pricing: The model has no pricing information, and this use would be unethical.

## 4.3 Factors

| Factor | Description | Category |
|---|---|---|
| Cyclicality | Weekend vs. weekday | Environment |
| Weather | Precipitation, humidity, and wind speed | Environment |
| Holidays | On-holiday vs. off-holiday | Environment / Groups of people |
| Volume | High-volume vs. low-volume products | Groups of people |
| Demographic data | Not measurable; would be based on socioeconomic demographics of store locations | Groups of people |

## 4.4 Metrics

- Weighted Absolute Percent Error (WAPE)
  - $\frac{\sum_{i=1}^{N} |A_i - F_i|}{\sum_{i=1}^{N} A_i}$
- Weighted Percent Error (WPE)
  - $\frac{\sum_{i=1}^{N} A_i - F_i}{\sum_{i=1}^{N} A_i}$
- Mean Absolute Scaled Error (MASE)
  - $\frac{\sum_{i=1}^{N} |A_i - F_i|}{\sum_{i=2}^{N} |A_i - A_{i-1}|}$
- Prediction Interval Coverage Percent (PICP)
  - $\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{l}_i \leq y_i \leq \hat{u}_i]$
- Mean Prediction Interval Width (MPIW)
  - $\frac{1}{N} \sum_{i=1}^{N} \hat{u}_i - \hat{l}_i$

## 4.5 Evaluation and Training Data

- **Training/validation:** Training is conducted on the 4 weeks prior to a split, with validation on the 4 weeks after the split. There are 6 splits on the full training set.
- **Training set:** The full training set comes from the dataset authors.
- **Conformal Prediction Split** The last week of the training dataset is explicitly held out for conformal prediction calibration.
- **Final evaluation:** Performed on the authors' test split (the last week), enforced by HuggingFace.

## 4.6 Quantitative Analysis

- Ongoing. As of 10Nov2026, we only have aggregate imputation results for WAPE and WPE gathered in an ad-hoc manner.

## 4.7 Ethical Considerations

- Anonymized data limits our ability to evaluate disparate impact.
- Poor performance on low-volume products may impact minorities with respect to the availability of niche foods.

## 4.8 Caveats and Recommendations

- We still need to add mechanisms to separate out relevant factors within our data pipeline.