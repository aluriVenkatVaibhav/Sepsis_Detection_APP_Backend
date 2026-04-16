import numpy as np
import logging
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
POP_IF_PATH = os.path.join(MODEL_DIR, "population_if.pkl")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

def build_population_if() -> IsolationForest:
    """Load a population Isolation Forest if available; otherwise fallback to synthetic."""
    if os.path.exists(POP_IF_PATH):
        try:
            loaded = joblib.load(POP_IF_PATH)
            logger.info("Loaded population IsolationForest from %s", POP_IF_PATH)
            return loaded
        except Exception as e:
            logger.warning("Failed to load population_if.pkl (%s). Using fallback synthetic model.", e)

    # Fallback (demo-only): synthetic "healthy" distribution.
    X_healthy = np.random.normal(0, 1, (1000, 7))
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_healthy)
    logger.warning("Using synthetic population IsolationForest fallback.")
    return model

def build_random_forest() -> RandomForestClassifier:
    """Load a trained Random Forest if available; otherwise fallback to synthetic."""
    if os.path.exists(RF_MODEL_PATH):
        try:
            loaded = joblib.load(RF_MODEL_PATH)
            logger.info("Loaded RF model from %s", RF_MODEL_PATH)
            return loaded
        except Exception as e:
            logger.warning("Failed to load rf_model.pkl (%s). Using fallback synthetic model.", e)

    # Fallback (demo-only): synthetic training set.
    X = np.random.normal(0, 1, (100, 11))
    y = np.random.randint(0, 3, 100)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    logger.warning("Using synthetic RandomForest fallback.")
    return model
