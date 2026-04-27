import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config



CUTOFF_DATE = pd.Timestamp('2015-01-01')


def add_time_competing(cleaned, full_history):
    """
    Compute TimeCompeting and TimeCompetingYearEnd for each entry in cleaned.
    First competition date is taken from full_history so pre-2015 history is accounted for.
    """
    full_history = full_history.sort_values(['Name', 'Date'])
    full_history['FirstCompDate'] = full_history.groupby('Name')['Date'].transform('min')
    full_history['TimeCompeting'] = (full_history['Date'] - full_history['FirstCompDate']).dt.days

    df_sorted = cleaned.sort_values(['Name', 'Year', 'Date']).copy()
    df_sorted['NumberOfMeets'] = df_sorted.groupby(['Name', 'Year']).transform('size')

    true_first_comp = full_history[['Name', 'FirstCompDate']].drop_duplicates()
    df_sorted = df_sorted.merge(true_first_comp, on='Name', how='left')

    df_sorted['TimeCompetingYearEnd'] = (
        pd.to_datetime(df_sorted['Year'].astype(str) + '-12-31') - df_sorted['FirstCompDate']
    ).dt.days

    df_sorted['TimeCompeting'] = (df_sorted['Date'] - df_sorted['FirstCompDate']).dt.days

    return df_sorted, full_history


def add_improvement_features(df_sorted):
    """
    Compute ImprovementGradientWithinYear: rate of improvement between
    first and last meet of the year.
    """
    df_sorted = df_sorted.sort_values(['Name', 'Year', 'Date'])
    grouped = df_sorted.groupby(['Name', 'Year'])

    df_sorted['ImprovementGradientWithinYear'] = (
        (grouped['TotalKg'].transform('last') -
         grouped['TotalKg'].transform('first')) /
        (grouped['TimeCompeting'].transform('last') -
         grouped['TimeCompeting'].transform('first')).replace(0, np.nan)
    )

    return df_sorted


def add_time_since_last_pb(df_sorted, full_history):
    """
    Compute TimeSinceLastPBYearEnd accounting for pre-2015 PBs.
    Pre-2015 PBs are looked up from full_history and used to correctly
    identify whether a post-2015 meet constitutes a PB.
    """
    df_sorted = df_sorted.sort_values(['Name', 'Year', 'Date'])
    full_history = full_history.sort_values(['Name', 'Date'])

    lifters_in_dataset = df_sorted['Name'].unique()
    pre_2015 = full_history[full_history['Date'] < CUTOFF_DATE]
    pre_2015 = pre_2015[pre_2015['Name'].isin(lifters_in_dataset)]
    pre_2015_sorted = pre_2015.sort_values(['Name', 'TotalKg', 'Date'], ascending=[True, False, True])

    best_per_lifter = pre_2015_sorted.groupby('Name').head(1)
    best_per_lifter = best_per_lifter[['Name', 'TotalKg', 'TimeCompeting']]
    historical_pb = best_per_lifter.rename(columns={
        'TotalKg': 'Before2015PB',
        'TimeCompeting': 'Before2015PBTimeCompeting'
    })
    df_sorted = df_sorted.merge(historical_pb, on='Name', how='left')

    df_sorted['PB'] = df_sorted.groupby('Name')['TotalKg'].cummax()
    df_sorted['PB'] = df_sorted[['PB', 'Before2015PB']].max(axis=1)

    df_sorted['PBBeforeMeet'] = df_sorted.groupby('Name')['PB'].shift(1)
    df_sorted['IsPB'] = df_sorted['TotalKg'] > (df_sorted['PBBeforeMeet'].fillna(df_sorted['Before2015PB']))
    df_sorted['IsPB'] = (df_sorted['PBBeforeMeet'].isna() & df_sorted['Before2015PB'].isna()) | df_sorted['IsPB']

    df_sorted['PBTimeCompeting'] = df_sorted['TimeCompeting'].where(df_sorted['IsPB'])
    df_sorted['LastPBTimeCompeting'] = df_sorted.groupby('Name')['PBTimeCompeting'].ffill()
    df_sorted['LastPBTimeCompeting'] = df_sorted[['LastPBTimeCompeting', 'Before2015PBTimeCompeting']].max(axis=1)

    df_sorted['TimeSinceLastPBYearEnd'] = df_sorted['TimeCompetingYearEnd'] - df_sorted['LastPBTimeCompeting']

    df_sorted = df_sorted.drop(columns=['PB', 'PBBeforeMeet', 'IsPB', 'PBTimeCompeting',
                                        'LastPBTimeCompeting', 'Before2015PB', 'Before2015PBTimeCompeting'])
    return df_sorted


def add_goodlift_features(df_sorted):
    """Compute BestGoodliftOfYear."""
    df_sorted = df_sorted.sort_values(['Name', 'Year', 'Date'])
    df_sorted['BestGoodliftOfYear'] = df_sorted.groupby(['Name', 'Year'])['Goodlift'].transform('max')
    return df_sorted


def to_panel_data(df_sorted):
    """
    Transform meet-level data to panel data (one row per lifter per year).
    Takes the last meet of each year for each lifter.
    Computes AvgMeetsPerYear as cumulative meets divided by years since first competition.
    """
    panel_data = df_sorted.groupby(['Name', 'Year']).tail(1).reset_index(drop=True)
    panel_data = panel_data.sort_values(['Name', 'Year'])

    panel_data['CumulativeMeets'] = panel_data.groupby('Name')['NumberOfMeets'].cumsum()
    panel_data['YearsSinceFirst'] = panel_data['Year'] - panel_data.groupby('Name')['Year'].transform('min') + 1
    panel_data['AvgMeetsPerYear'] = panel_data['CumulativeMeets'] / panel_data['YearsSinceFirst']
    panel_data = panel_data.drop(columns=['CumulativeMeets', 'YearsSinceFirst'])

    return panel_data


def add_churn_target(panel_data, last_complete_year = None):
    """
    Add Churns target column. A lifter churns if they do not compete the following year.
    Drops the final year of data as churn cannot be determined for it.
    """
    panel_data = panel_data.sort_values(['Name', 'Year'])
    panel_data['CompetesNextYear'] = (
        (panel_data['Name'] == panel_data['Name'].shift(-1)) &
        (panel_data['Year'] + 1 == panel_data['Year'].shift(-1))
    )
    panel_data['Churns'] = (~panel_data['CompetesNextYear']).astype(int)
    panel_data = panel_data.drop(columns='CompetesNextYear')
    

    if last_complete_year is None:
        now = datetime.now()
        last_complete_year = now.year if (now.month == 12 and now.day >= 7) else now.year - 1

    panel_data = panel_data[panel_data['Year'] <= last_complete_year]

    return panel_data, last_complete_year


def impute_age(panel_data, age_median):
    """Impute missing Age values with global training median."""
    panel_data['Age'] = panel_data['Age'].fillna(age_median)
    return panel_data


def select_columns(panel_data):
    """Reduce panel data to final column set."""
    cols = ['Name', 'Year'] + config.FEATURES + ['Churns']
    return panel_data.loc[:, cols]


def build_panel_data(cleaned, full_history):
    """
    Full feature engineering pipeline. Takes cleaned competition history and full history
    dataframes and returns panel data ready for modelling.

    Args:
        cleaned: Cleaned SBD-only dataframe from cleaning.py (2015-2025).
        full_history: Broader SBD dataframe across all years for lookups.
        age_median: median age computed from training set for imputation. or in deployment compute from all available data.

    Returns:
        panel_data: One row per lifter per year with engineered features and Churns target.
    """
    df_sorted, full_history = add_time_competing(cleaned, full_history)
    df_sorted = add_improvement_features(df_sorted)
    df_sorted = add_time_since_last_pb(df_sorted, full_history)
    df_sorted = add_goodlift_features(df_sorted)

    panel_data = to_panel_data(df_sorted)
    panel_data, last_complete_year = add_churn_target(panel_data)
    panel_data = impute_age(panel_data, panel_data['Age'].median())
    panel_data = select_columns(panel_data)

    return panel_data, last_complete_year