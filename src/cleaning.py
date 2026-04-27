import pandas as pd
import numpy as np
from datetime import datetime


COLUMNS = [
    'Name', 'Date', 'Sex', 'Age', 'MeetName', 'Federation', 'Division',
    'BodyweightKg', 'WeightClassKg', 'Best3SquatKg', 'Best3BenchKg',
    'Best3DeadliftKg', 'Equipment', 'Event', 'Squat1Kg', 'Squat2Kg',
    'Squat3Kg', 'Bench1Kg', 'Bench2Kg', 'Bench3Kg', 'Deadlift1Kg',
    'Deadlift2Kg', 'Deadlift3Kg', 'TotalKg', 'Goodlift', 'Sanctioned'
]

current_year = datetime.now().year
OLD_STRUCT_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
NEW_STRUCT_YEARS = list(range(2021, current_year + 1))
ALL_YEARS = OLD_STRUCT_YEARS + NEW_STRUCT_YEARS

F_OLD_CLASSES = ['43', '47', '52', '57', '63', '72', '84', '84+']
F_NEW_CLASSES = ['43', '47', '52', '57', '63', '69', '76', '84', '84+']
M_CLASSES = ['53', '59', '66', '74', '83', '93', '105', '120', '120+']


def reduce_columns(df):
    """reduce to relevant columns."""
    return df[COLUMNS].copy()


def normalise_strings(df):
    """strip whitespace and lowercase string columns."""
    df = df.copy()
    for col in ['Name', 'MeetName', 'Federation', 'Equipment']:
        df[col] = df[col].str.strip().str.lower()
    return df


def filter_entries(df):
    """
    Filter for sanctioned, raw, full power (SBD) entries with valid weight classes.
    Removes duplicates based on subset of columns to capture duplicates despite minor inconsistencies in other fields.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    df['Year'] = df['Date'].dt.year
    df = df.dropna(subset=['WeightClassKg'])
    df = df.loc[df['Sanctioned'] != 'No']
    df = df.loc[df['Equipment'] == 'raw']
    df = df.drop_duplicates(['Name', 'Sex', 'Event', 'Date', 'TotalKg', 'MeetName'])
    df = df.loc[df['Event'] == 'SBD'].copy()
    return df


def filter_years(df):
    """Filter for 2015-2025 (inclusive)"""
    return df.loc[df['Year'].isin(ALL_YEARS)].copy()


def clean_female_weight_classes(df):
    """
    filter female entries to valid weight classes for relevant year.
    IPF restructured female weight classes between 2020 and 2021:
        Pre-2021: includes 72kg, excludes 69kg and 76kg
        Post-2021: includes 69kg and 76kg, excludes 72kg
    """
    f = df.loc[df['Sex'] == 'F'].copy()
    f_cleaned = f.loc[
        (f['Year'].isin(OLD_STRUCT_YEARS) & f['WeightClassKg'].isin(F_OLD_CLASSES)) |
        (f['Year'].isin(NEW_STRUCT_YEARS) & f['WeightClassKg'].isin(F_NEW_CLASSES))
    ]
    return f_cleaned


def clean_male_weight_classes(df):
    """
    Filter male entries to valid IPF weight classes.
    Male weight classes were the same 2015 onwards.
    """
    m = df.loc[df['Sex'] == 'M'].copy()
    m_cleaned = m.loc[m['WeightClassKg'].isin(M_CLASSES)]
    return m_cleaned


def clean_mx_weight_classes(df, f_cleaned, m_cleaned):
    """
    The IPF does not define a non-binary weight class structure, so Mx entries are retained only where their weight class matches an existing or 
    historical class. Therefore filter Mx entries to weight classes used in the male and female divisions.
    """
    mx = df.loc[df['Sex'] == 'Mx'].copy()
    all_classes = (
        list(f_cleaned['WeightClassKg'].unique()) +
        list(m_cleaned['WeightClassKg'].unique())
    )
    mx_cleaned = mx.loc[mx['WeightClassKg'].isin(all_classes)]
    return mx_cleaned



def clean(df):
    """
    Full cleaning pipeline. Takes OpenPowerlifting dataframe and returns
    cleaned dataframe of sanctioned, raw, full power (SBD) entries from 2015-2025
    with valid IPF weight classes, and a broader full history dataframe used
    for feature engineering lookups e.g. pre-2015 PBs, TimeCompeting.

    Args:
        df: Raw OpenPowerlifting dataframe.

    Returns:
        cleaned: Filtered dataframe for model training.
        full_history: Dataframe across all years for feature engineering lookups.
    """
    df = reduce_columns(df)
    df = normalise_strings(df)
    full_history = filter_entries(df)
    filtered = filter_years(full_history)

    f_cleaned = clean_female_weight_classes(filtered)
    m_cleaned = clean_male_weight_classes(filtered)
    mx_cleaned = clean_mx_weight_classes(filtered, f_cleaned, m_cleaned)

    cleaned = pd.concat([f_cleaned, m_cleaned, mx_cleaned], ignore_index=True)
    return cleaned, full_history




