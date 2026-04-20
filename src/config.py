from pathlib import Path
import os


#Path makes it a path object
#.resolve makes it an absolute file path
#parent gets the parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "dataset.csv"

#tries to read the environment variable "PROJECT_DB_LOCATION". if it does not exist then it uses DEFAULT_DB
DB_LOCATION = Path(os.getenv("PROJECT_DB_LOCATION", DEFAULT_DB))

#can write to the environment variable from powershell using 
#setx PROJECT_DB_LOCATION "C:\absolute\path\to\dataset.csv"


CAP_MEETS_SINCE_BOMBOUT = 999

FEATURES = [
    'TimeSinceLastPBYearEnd',
    'ImprovementGradientWithinYear',
    'BestGoodliftOfYear',
    'Age',
    'AvgMeetsPerYear'
]

FINAL_PARAMS = {
    'learning_rate': 0.01,
    'max_iter': 750,
    'max_depth': 2,
    'min_samples_leaf': 5,
    'l2_regularization': 0.05
}
