#testing churn target as only unit test because it's so central to project want to verify correctness

import pandas as pd
from datetime import datetime
from feature_engineering import add_churn_target

def test_churn_target():
    panel_data = pd.DataFrame([
        {'Name': 'daniel chen', 'Year': 2021},
        {'Name': 'daniel chen', 'Year': 2022},
        {'Name': 'daniel chen', 'Year': 2023},  # competes all 3 years. no churn in 2021 or 2022
        {'Name': 'jane doe', 'Year': 2021},
        {'Name': 'jane doe', 'Year': 2022},    # stops after 2022. churns in 2022, no churn in 2021
        {'Name': 'bob jones', 'Year': 2021},   # takes 2022 off. churns in 2021
        {'Name': 'bob jones', 'Year': 2023},   # returns in 2023. churns in 2023
        {'Name': 'ria patel', 'Year': 2023}, # competes in 2023 and 2024. no churn in 2023
        {'Name': 'ria patel', 'Year': 2024}, # 2024 should be dropped as specified by last_complete_year
    ])

    result = add_churn_target(panel_data, last_complete_year=2023)

    assert result.loc[(result['Name'] == 'daniel chen') & (result['Year'] == 2021), 'Churns'].item() == 0
    assert result.loc[(result['Name'] == 'daniel chen') & (result['Year'] == 2022), 'Churns'].item() == 0
    assert result.loc[(result['Name'] == 'jane doe') & (result['Year'] == 2021), 'Churns'].item() == 0
    assert result.loc[(result['Name'] == 'jane doe') & (result['Year'] == 2022), 'Churns'].item() == 1
    assert result.loc[(result['Name'] == 'bob jones') & (result['Year'] == 2021), 'Churns'].item() == 1
    assert result.loc[(result['Name'] == 'bob jones') & (result['Year'] == 2023), 'Churns'].item() == 1
    assert result.loc[(result['Name'] == 'ria patel') & (result['Year'] == 2023), 'Churns'].item() == 0
    assert 2024 not in result['Year'].values
    assert len(result) == 8


