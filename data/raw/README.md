# data/raw/

This folder contains raw data files for use in this project.

Raw data should not be modified.
Keep original files as received so analysis can be reproduced from the source.

## hours_scores_case.csv

A small synthetic dataset used in the example project.
Ten students with study hours, quiz scores, attendance, sleep, and a final score.
Used to verify the project environment is working correctly.

## cereal.csv (My Custom Project Dataset, Phase 5)

Nutrition facts and Consumer Reports ratings for 77 breakfast cereals.
Used in `notebooks/ml_07_teja_p5.ipynb` and `src/mlstudio/app_cereal_teja.py`
to predict `rating` from the nine nutrition columns.

- 77 rows, 16 columns
- Features used: calories, protein, fat, sodium, fiber, carbo, sugars, potass, vitamins
- Target: rating (0-100)
- Downloaded from Kaggle

### The -1 sentinel (important)

This file has **no blank cells**, so a plain missing-value check reports the
data is clean. It is not. Four values across three cereals are recorded as
`-1`, meaning "not measured":

| cereal | column(s) with -1 |
| --- | --- |
| Almond Delight | potass |
| Cream of Wheat (Quick) | potass |
| Quaker Oatmeal | carbo, sugars |

`make_clean_view()` converts `-1` to `NaN` before dropping, leaving 74 rows.

Those three rows are dropped rather than imputed, for a reason established in
`notebooks/ml_07_teja_p5.ipynb` Section 3b: the published `rating` for each of
them is reproduced **exactly** when `-1` is fed to the model as a real value,
and missed when `-1` is replaced with `0`. The rating column was therefore
computed without cleaning the sentinel first, so these rows have unreliable
**targets**, not just unreliable features - imputing the feature would only
teach the model to reproduce a data-entry artifact.

### Citation

Crawford, C. (2017). 80 Cereals [Data set].
Retrieved from <https://www.kaggle.com/datasets/crawford/80-cereals>

Original source: Carnegie Mellon University StatLib.

## Option 1. Recommended: diabetes_case.csv (Custom Project Dataset)

A sample from the CDC Diabetes Health Indicators dataset derived from the
Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.

This sample contains 10,000 rows and 22 columns including a binary target
indicating whether a respondent has diabetes or prediabetes.

- ⭐diabetes.csv, 6MB, balanced 50/50 split, roughly 70,000 rows, included in data/raw.
- diabetes_binary_health_indicators_BRFSS2015.csv, 22 MB, binary target: 0=no diabetes, 1=diabetes/prediabetes
- diabetes_012_health_indicators_BRFSS2015.csv, 22 MB, 3 classes: 0=no diabetes, 1=prediabetes, 2=diabetes

### Features include

- Demographics: age group, sex, income, education
- Health indicators: BMI, blood pressure, cholesterol
- Lifestyle: physical activity, smoking, alcohol consumption, diet
- Healthcare access: health coverage, doctor visits, cost barriers

### Example Use By Module

- ml-01: Characterize the problem. Is this classification or regression and why?
- ml-02: Explore features, handle class imbalance, encode categorical columns.
- ml-03: Classify diabetes status, compare precision and recall across models.
- ml-04: Predict BMI as a continuous target using regression.
- ml-05: Compare random forest and gradient boosting on this dataset.

### Citation

Teboul, A. (2021). Diabetes Health Indicators Dataset [Data set].
Derived from CDC BRFSS 2015.
Retrieved from
<https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset>

Original source: Centers for Disease Control and Prevention (CDC).
Behavioral Risk Factor Surveillance System Survey Data.
Atlanta, Georgia: U.S. Department of Health and Human Services, 2015.
<https://www.cdc.gov/brfss/>

UCI ML Repository:
<https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators>

### License

Public domain (U.S. Government data, CDC).

## Option 2. Choose Another Seaborn Dataset

For this exercise, you can choose another Seaborn dataset.
See the example notebook for a command to list them.

## Option 3. Choose Another CSV Real-World Dataset

Possible places to look:

- <https://github.com/awesomedata/awesome-public-datasets>
- <https://www.dataquest.io/blog/free-datasets-for-projects/>

## Option 4: Explore your Own CSV Dataset

Use your own data from work or home and explore possible relationships.
