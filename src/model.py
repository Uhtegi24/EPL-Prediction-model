import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def load_processed_matches(filepath: str) -> pd.DataFrame:
    """
    Load preprocessed match-level data from matches_processed.csv.
    """
    print(f"Loading processed matches from {filepath}...")
    return pd.read_csv(filepath)


def prepare_features_from_processed(df: pd.DataFrame):
    """
    Prepare feature matrix X, target y and feature names from a preprocessed
    matches DataFrame (like matches_processed.csv).
    """
    if "outcome" not in df.columns:
        raise ValueError("Input data must contain an 'outcome' column.")

    y = df["outcome"]
    X = df.drop(columns=["outcome"])

    # Use only numeric features
    X = X.select_dtypes(include=["number"]).fillna(0)

    feature_names = X.columns.tolist()
    return X, y, feature_names


class MatchOutcomeModel:
    """
    Wrapper around RandomForest / XGBoost for multi-class
    match outcome prediction (home / draw / away).
    """

    def __init__(self, model_type: str = "rf", random_state: int = 42, **model_kwargs):
        self.model_type = model_type
        self.random_state = random_state
        self.le = LabelEncoder()
        self.model = self._build_model(**model_kwargs)

    def _build_model(self, **kwargs):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs,
            )
        elif self.model_type == "xgb":
            if not XGB_AVAILABLE:
                raise ImportError(
                    "xgboost is not installed, cannot use model_type='xgb'."
                )
            return XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                learning_rate=0.05,
                max_depth=6,
                n_estimators=400,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                n_jobs=-1,
                num_class=3,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit model on training data.
        y_train uses string labels ('home', 'draw', 'away').
        """
        y_enc = self.le.fit_transform(y_train)
        self.model.fit(X_train, y_enc)

    def predict(self, X: pd.DataFrame):
        """
        Predict class labels (home/draw/away) using predict_proba + argmax
        so behaviour is consistent between RF and XGB.
        """
        proba = self.model.predict_proba(X)
        y_pred_enc = proba.argmax(axis=1)
        return self.le.inverse_transform(y_pred_enc)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return probability-based predictions as a DataFrame with
        columns like ['prob_home', 'prob_draw', 'prob_away'].
        """
        proba = self.model.predict_proba(X)
        class_names = list(self.le.classes_)
        proba_cols = [f"prob_{c}" for c in class_names]
        return pd.DataFrame(proba, columns=proba_cols, index=X.index)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Compute evaluation metrics (accuracy, log_loss, classification report).
        """
        y_true_enc = self.le.transform(y_test)
        proba = self.model.predict_proba(X_test)
        y_pred_enc = proba.argmax(axis=1)
        y_pred_labels = self.le.inverse_transform(y_pred_enc)

        acc = accuracy_score(y_test, y_pred_labels)
        ll = log_loss(y_true_enc, proba)
        report = classification_report(y_test, y_pred_labels, digits=4)

        return {
            "accuracy": acc,
            "log_loss": ll,
            "classification_report": report,
        }

    def get_feature_importances(self, feature_names, top_n=None) -> pd.DataFrame:
        """
        Returns a DataFrame of feature importances sorted descending.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError(
                "Underlying model has no feature_importances_ attribute."
            )

        importances = self.model.feature_importances_
        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )

        if top_n is not None:
            fi = fi.head(top_n)

        return fi


def train_match_outcome_model(
    data_path: str,
    model_type: str = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    High-level helper that:
      1) loads matches_processed.csv
      2) prepares features/labels
      3) splits train/test
      4) trains model
      5) evaluates and returns metrics + feature importances
    """
    df = load_processed_matches(data_path)
    X, y, feature_names = prepare_features_from_processed(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = MatchOutcomeModel(model_type=model_type, random_state=random_state)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    feature_importances = model.get_feature_importances(feature_names)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "feature_names": feature_names,
    }
