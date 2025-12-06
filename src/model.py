import argparse
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

from data import load_data, preprocess_data


def prepare_features(df):
    """
    Takes raw events-level dataframe, calls preprocess_data(df)
    and returns (X, y, feature_names) ready for modeling.
    """
    processed = preprocess_data(df)

    # Target = match outcome (home/draw/away)
    y = processed["outcome"]

    # Features: drop target and keep only numeric columns
    X = processed.drop(columns=["outcome"])
    X = X.select_dtypes(include=["number"]).fillna(0)

    return X, y, X.columns.tolist()


class MatchOutcomeModel:
    """
    Wrapper around RandomForest / XGBoost for multi-class
    match outcome prediction (home / draw / away).
    """

    def __init__(self, model_type="rf", random_state=42, **model_kwargs):
        self.model_type = model_type
        self.random_state = random_state
        self.le = LabelEncoder()
        self.model = self._build_model(**model_kwargs)

    def _build_model(self, **kwargs):
        if self.model_type == "rf":
            # Baseline RF defaults; tune later if needed
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )
        elif self.model_type == "xgb":
            if not XGB_AVAILABLE:
                raise ImportError("xgboost is not installed, cannot use model_type='xgb'.")

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
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit(self, X_train, y_train):
        """
        Fit model on training data.
        y_train is string labels ('home', 'draw', 'away').
        """
        y_enc = self.le.fit_transform(y_train)
        self.model.fit(X_train, y_enc)

    def predict(self, X):
        """
        Predict class labels (home/draw/away).
        """
        class_indices = self.model.predict(X)
        # For XGB, predict returns class indices; for RF, we convert from probs
        if class_indices.ndim > 1:
            class_indices = class_indices.argmax(axis=1)
        return self.le.inverse_transform(class_indices)

    def predict_proba(self, X) -> pd.DataFrame:
        """
        Return probability-based predictions as a DataFrame with
        columns like ['prob_draw', 'prob_home', 'prob_away'].
        """
        proba = self.model.predict_proba(X)
        class_names = list(self.le.classes_)
        proba_cols = [f"prob_{c}" for c in class_names]
        return pd.DataFrame(proba, columns=proba_cols, index=X.index)

    def evaluate(self, X_test, y_test):
        """
        Compute evaluation metrics (accuracy, log_loss, classification report).
        Returns a dict + text report string for logging if needed.
        """
        y_true_enc = self.le.transform(y_test)
        proba = self.model.predict_proba(X_test)
        y_pred_enc = proba.argmax(axis=1)
        y_pred_labels = self.le.inverse_transform(y_pred_enc)

        acc = accuracy_score(y_test, y_pred_labels)
        ll = log_loss(y_true_enc, proba)
        report = classification_report(y_test, y_pred_labels, digits=4)

        metrics = {
            "accuracy": acc,
            "log_loss": ll,
            "classification_report": report,
        }
        return metrics

    def get_feature_importances(self, feature_names, top_n=None):
        """
        Returns a DataFrame of feature importances sorted descending.
        This is purely data; actual visualisation is done elsewhere.
        """
        if not hasattr(self.model, "feature_importances_"):
            raise AttributeError("Underlying model has no feature_importances_ attribute.")

        importances = self.model.feature_importances_
        fi = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

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
      1) loads data
      2) preprocesses into features/labels
      3) splits train/test
      4) trains model
      5) evaluates and returns metrics + feature importances

    Returns:
      {
        "model": MatchOutcomeModel instance,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": {...},
        "feature_importances": DataFrame,
        "feature_names": [...],
      }
    """
    # 1. Load raw events-level data
    df = load_data(data_path)

    # 2. Preprocess into match-level with rolling features
    X, y, feature_names = prepare_features(df)

    # 3. Train/test split (stratified by outcome)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 4. Init and train model
    model = MatchOutcomeModel(model_type=model_type, random_state=random_state)
    model.fit(X_train, y_train)

    # 5. Evaluate
    metrics = model.evaluate(X_test, y_test)

    # 6. Feature importances (for later visualisation elsewhere)
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


def main():
    """
    Optional CLI entry point.
    This only prints text summaries; all visualisation is expected
    to be handled in a separate module by whoever you’ve bullied into doing it.
    """
    parser = argparse.ArgumentParser(description="Train match outcome prediction model.")
    parser.add_argument("--data", type=str, required=True, help="Path to raw events CSV.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["rf", "xgb"],
        default="rf",
        help="Model type: 'rf' for RandomForest, 'xgb' for XGBoost.",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Test set size fraction (default: 0.2)."
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility."
    )

    args = parser.parse_args()

    result = train_match_outcome_model(
        data_path=args.data,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("=== Metrics ===")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Log loss: {result['metrics']['log_loss']:.4f}")
    print("\nClassification report:")
    print(result["metrics"]["classification_report"])

    print("\n=== Top 20 feature importances ===")
    print(result["feature_importances"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()