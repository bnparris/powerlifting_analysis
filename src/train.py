
import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent / "src"))
import config




model_path = config.PROJECT_ROOT/ "model.pkl"
joblib.dump(clf, model_path)