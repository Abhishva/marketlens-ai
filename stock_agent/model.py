import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from .features import FEATURE_COLUMNS

def train_model(data: pd.DataFrame):
    """Train on older rows and evaluate on the latest chronological holdout."""
    clean = data.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    if len(clean) < 80:
        raise ValueError("Not enough usable rows after indicator calculation.")

    X = clean[FEATURE_COLUMNS]
    y = clean["Target"].astype(int)

    split = int(len(clean) * 0.8)
    if split < 40 or len(clean) - split < 15:
        raise ValueError("Not enough data for a chronological train/test split.")

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, accuracy, FEATURE_COLUMNS

def predict_next_day(model, data: pd.DataFrame, feature_columns=None):
    feature_columns = feature_columns or FEATURE_COLUMNS
    latest = data.dropna(subset=feature_columns).iloc[[-1]]
    probabilities = model.predict_proba(latest[feature_columns])[0]
    classes = list(model.classes_)
    up_index = classes.index(1)
    up_probability = float(probabilities[up_index])

    if up_probability >= 0.5:
        direction = "Upward bias"
        confidence = up_probability * 100
    else:
        direction = "Downward bias"
        confidence = (1 - up_probability) * 100

    return {"direction": direction, "confidence": confidence}
