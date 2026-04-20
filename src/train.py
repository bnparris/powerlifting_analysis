import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from cleaning import clean
from feature_engineering import build_panel_data


def train():
    df = pd.read_csv(config.DB_LOCATION)
    # clean and build panel data
    cleaned, full_history = clean(df)

    panel_data = build_panel_data(cleaned, full_history)

    # train on all available data
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

    joblib.dump(clf, config.PROJECT_ROOT / 'models' / 'model.pkl')
    print('Model saved.')


if __name__ == '__main__':
    train()