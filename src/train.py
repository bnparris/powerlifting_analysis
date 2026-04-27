import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from cleaning import clean
from feature_engineering import build_panel_data
from datetime import datetime
import json


def train():
    df = pd.read_csv(config.DB_LOCATION)
    print('Building panel data')
    cleaned, full_history = clean(df)
    panel_data, last_complete_year = build_panel_data(cleaned, full_history)
    panel_data.to_csv(config.PRODUCTION_PANEL, index = False)

    #train on all available data
    print('Training model')
    train_X = panel_data[config.FEATURES]
    train_y = panel_data['Churns']

    clf = HistGradientBoostingClassifier(
        learning_rate=config.FINAL_PARAMS['learning_rate'],
        max_iter=config.FINAL_PARAMS['max_iter'],
        max_depth=config.FINAL_PARAMS['max_depth'],
        min_samples_leaf=config.FINAL_PARAMS['min_samples_leaf'],
        l2_regularization=config.FINAL_PARAMS['l2_regularization'],
        random_state=42
    )

    clf.fit(train_X, train_y)

    joblib.dump(clf, config.MODEL_PATH)


    metadata = {
    'trained_on': datetime.now().strftime('%d-%m-%Y %H:%M:%S'),
    'last_complete_year': last_complete_year,
    'n_training_samples': len(train_y),
    'features': config.FEATURES,
    'params': config.FINAL_PARAMS
    }
    
    with open(config.METADATA_PATH, 'w') as file:
        json.dump(metadata, file, indent=2)
        
    print('Model saved.')


if __name__ == '__main__':
    train()