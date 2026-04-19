# Predicting Powerlifter Churn

## Overview

Mention raw lifting, and IPF affiliates.


The goal of this project is to identify powerlifters at risk of churning (i.e. not competing in the following calendar year), enabling powerlifting federations to target their retention interventions. A HistGradientBoostingClassifier trained on data from Open Powerlifting achieved an accuracy of X%, compared to a majority class baseline of Y%. In practice, if richer business data were available, a profit-based evaluation metric taking into account customer liftertime value, cost of intervention retentions, and effectiveness of interventions could be used to better align model performance with real-world impact.


## Dataset
The original dataset was transformed to a panel structure as detailed below.

Source| https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html
Original Structure| A history of powerlifting meet performances. Each record corresponds to a lifter's performance at a particular powerlifting meet.
Transformed panel structure| Panel data where each record represents a unique combination of lifter and year. Features are constructed using only information available up to the end of that year. Contains X records and Y unique lifters. 


## Data Cleaning

See `notebooks/01_cleaning.ipynb` for full implementation.

### Initial Cleaning
A reduced set of 26 columns was retained covering lifter demographics, 
competition metadata, individual attempts, and best lift totals. Routine cleaning was applied including type standardisation, whitespace 
normalisation, and removal of rows with missing `WeightClassKg`. Entries sharing the same name, sex, event, date, total, and meet name were 
treated as duplicate records. Duplicates were removed on this subset of columns to 
account for minor inconsistencies in other fields.

### Filtering
- **Equipment**: restricted to raw powerlifting only, as equipped lifting is 
  considered a distinct discipline
- **Sanctioned**: unsanctioned meets excluded
- **Division**: dropped due to high cardinality (1,135 unique values) caused 
  by inconsistent data entry
- **Weight class**: rows with missing `WeightClassKg` dropped
- **Event**: restricted to full power (SBD) entries, excluding single-lift 
  competitions such as bench-only


### World Record Validation
Entries exceeding current official IPF world records by weight class were 
flagged and reviewed. These were retained after cross-referencing against 
available meet results, as records can only be set at international 
competitions and higher totals may occur at national or local level without 
official recognition.

### Weight Class Restructuring
The IPF revised its weight class structure during the dataset's time span. 
Participation trends were analysed annually to identify the transition point 
and the classes in use before and after:

- **Female**: the 72kg class was present from 2015–2020 and replaced by 
  69kg and 76kg from 2021 onwards. Entries were retained only where the 
  weight class matched the structure in use that year.
- **Male**: weight classes were stable across the 2015–2025 window.
- **Mx entries**: retained where `WeightClassKg` matched a class used in 
  either the men's or women's divisions, consistent with current IPF rules.

### Bomb-Out Flagging & Attempt Imputation
Lifters who recorded no successful attempts across all three lifts in a 
discipline were flagged as having bombed out (`BombOut = True`). For 
non-bombing lifters where individual attempt data was absent, attempt count 
was imputed using the mean of comparable non-bombing lifters.

## Feature Engineering

See `notebooks/02_data_transformation.ipynb` for full implementation.

Predictions are made at the end of each calendar year, as most powerlifting 
federation memberships operate on an annual basis. All features are constructed 
using only information available up to the end of that year to prevent leakage.

### Final Feature Set

| Feature | Description |
|---|---|
| `TimeSinceLastPBYearEnd` | Days since the lifter last set a personal best, measured at year end |
| `BestGoodliftOfYear` | Best Goodlift points achieved across all meets that year |
| `ImprovementGradientWithinYear` | Rate of improvement in total between first and last meet of the year |
| `Age` | Lifter age at time of last meet of the year |
| `AvgMeetsPerYear` | Expanding mean of meets per year up to and including the current year |
| `Sex` | Encoded as 0 (F), 1 (M), 2 (Mx) |

### Feature Construction Notes
- **Improvement features**: both absolute and percentage-based versions were 
  constructed; absolute versions were retained after validation performance 
  showed negligible difference between the two
- **Attempt imputation**: where individual attempt data was absent but the 
  lifter did not bomb out, `AttemptsMade` was imputed using the mean of 
  non-bombing lifters
- **Age imputation**: a binary `AgeMissing` indicator was added before imputing 
  missing `Age` values with the median for that year, to allow the model to 
  learn from the missingness pattern while avoiding leakage
- **Weight class and federation** were considered but did not survive feature 
  selection

### Target Variable
`Churns` is set to 1 where a lifter does not appear in the dataset in the 
following calendar year. The final year (2025) is excluded as future 
competition status is unknown.

## Methodology

### Train / Validation / Test Split
A temporal split was used to prevent data leakage:

| Split      | Years     | Share |
|---|---|---|
| Train      | up to 2022 | ~75% |
| Validation | 2023       | ~12% |
| Test       | 2024       | ~13% |

Feature selection decisions were made on the validation set. For hyperparameter 
tuning, train and validation years were combined with time-based cross-validation 
applied within the combined set. Final evaluation was performed on the held-out 
2024 test set.

### Handling Covid Years
Excluding 2020 and 2021 from training was considered given atypical churn 
patterns during the pandemic. Performance differences on the validation set 
were negligible between models trained with and without these years, so they 
were retained. This is expected to make the model more robust to future shocks.

### Model Selection
`HistGradientBoostingClassifier` was selected as it natively handles 
categorical features and missing values.

### Evaluation Metric
Accuracy was chosen as the primary evaluation metric. The test set is 
approximately balanced (52:48), making accuracy an interpretable metric. 
In the absence of business data (intervention cost, conversion rate, customer 
lifetime value), equal misclassification costs were assumed. A profit-based 
metric could be substituted with this information.

### Feature Selection
1. Permutation importance was computed on the validation set and features 
   with negative importance were removed
2. Remaining features were added incrementally in order of importance and 
   validation accuracy was tracked
3. The feature count maximising validation accuracy was selected (6 features)

## Results
Key metrics and findings.

## How to Run
Steps to reproduce.

## Project Structure
Folder layout.

## Requirements
Dependencies.

## Future Work
Planned improvements.

## Author
Your name + links.