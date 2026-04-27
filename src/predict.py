import joblib
import pandas as pd
import config
import json

#model designed to be retrained annually (end of calendar year which counts as December 7th onwards)


def predict(names):
    """
    Predicts churn for a list of lifters using the most recent year's panel data.

    Loads panel data, model and metadata from disk. Predictions are made for
    the last complete year as determined at training time. Lifters not found
    in the panel data are excluded from predictions and flagged.

    Args:
        names: List of lifter names. Names are normalised to lowercase and
               stripped of whitespace before lookup.

    Returns:
        predictions: Dictionary mapping lifter name to churn prediction (0 or 1).
        last_complete_year: The most recent year used for predictions.
        name_not_found: List of names not found in the panel data.
    """
    panel = pd.read_csv(config.PRODUCTION_PANEL)
    model = joblib.load(config.MODEL_PATH)
    with open(config.METADATA_PATH, 'r') as metadata_file:
        metadata = json.load(metadata_file)
        last_complete_year = metadata['last_complete_year']
        
    names = [name.strip().lower() for name in names]
    panel_rows = panel.loc[(panel['Name'].isin(names)) & (panel['Year'] == last_complete_year)]

    name_not_found = [name for name in names if name not in panel_rows['Name'].values]
    if name_not_found:
        print(f'The following lifters were not found in panel data: {name_not_found}')

    if len(panel_rows) == 0:
        return {'predictions': {}, 'last_complete_year': last_complete_year, 'name_not_found': name_not_found}


        
    X = panel_rows[config.FEATURES]
    pred = model.predict(X)
    predictions_df = panel_rows[['Name', 'Year']].copy()
    predictions_df['ChurnPrediction'] = pred
    predictions_dict = predictions_df.set_index('Name')['ChurnPrediction'].to_dict()
    return {'predictions': predictions_dict, 'last_complete_year': last_complete_year, 'name_not_found': name_not_found}
    
    
    
    
    

