"""ResearchGPT Classical Classifier Service.

Implements TF-IDF + Classical Machine Learning models:
- Multinomial Naive Bayes (MultinomialNB)
- Logistic Regression (LogisticRegression)
- Support Vector Machine (LinearSVC)

Provides training, inference, probability/confidence estimation,
comprehensive metric evaluation, and artifact serialization.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from backend.app.models.classification import (
    ClassificationEvaluationMetrics,
    ClassificationResponse,
)

logger = logging.getLogger("classical_classifier_service")


class ClassicalClassifierService:
    """Service for building, training, evaluating, and serving TF-IDF classical classifiers."""

    SUPPORTED_MODELS = {
        "naive_bayes": MultinomialNB,
        "logistic_regression": LogisticRegression,
        "svm": LinearSVC,
    }

    def __init__(
        self,
        model_type: str = "logistic_regression",
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 20000,
        sublinear_tf: bool = True,
        stop_words: Optional[str] = "english",
        c_param: float = 1.0,
        alpha: float = 1.0,
        random_state: int = 42,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. Choose from {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.stop_words = stop_words
        self.c_param = c_param
        self.alpha = alpha
        self.random_state = random_state

        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[List[Any]] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        """Construct the Scikit-Learn TF-IDF + Classifier Pipeline."""
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
        )

        if self.model_type == "naive_bayes":
            clf = MultinomialNB(alpha=self.alpha)
        elif self.model_type == "logistic_regression":
            clf = LogisticRegression(
                C=self.c_param,
                max_iter=1000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        elif self.model_type == "svm":
            clf = LinearSVC(
                C=self.c_param,
                max_iter=2000,
                random_state=self.random_state,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unknown model_type {self.model_type}")

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])

    def train(self, texts: List[str], labels: List[Any]) -> "ClassicalClassifierService":
        """Fit the TF-IDF vectorizer and classifier on training data."""
        if not texts or not labels:
            raise ValueError("Training texts and labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(labels)} labels")

        self.pipeline.fit(texts, labels)
        clf = self.pipeline.named_steps["clf"]
        self.classes_ = list(clf.classes_)
        logger.info(
            "Trained [%s] on %d samples. Classes: %s",
            self.model_type,
            len(texts),
            self.classes_,
        )
        return self

    def predict(self, texts: List[str]) -> List[Any]:
        """Predict class labels for given texts."""
        if self.pipeline is None or self.classes_ is None:
            raise RuntimeError("Model has not been trained or loaded yet")
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> List[Dict[str, float]]:
        """Return probability or confidence score distribution per class for each text."""
        if self.pipeline is None or self.classes_ is None:
            raise RuntimeError("Model has not been trained or loaded yet")

        clf = self.pipeline.named_steps["clf"]
        tfidf = self.pipeline.named_steps["tfidf"]

        if hasattr(self.pipeline, "predict_proba") and hasattr(clf, "predict_proba"):
            probs = self.pipeline.predict_proba(texts)
        else:
            # For LinearSVC, apply sigmoid/softmax over decision_function to obtain pseudo-probabilities
            X_vec = tfidf.transform(texts)
            decision = clf.decision_function(X_vec)
            if decision.ndim == 1:
                # Binary classification: decision function is (n_samples,)
                prob_pos = 1.0 / (1.0 + np.exp(-decision))
                probs = np.column_stack([1.0 - prob_pos, prob_pos])
            else:
                # Multiclass: softmax
                exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        results = []
        for row in probs:
            prob_dict = {str(c): float(p) for c, p in zip(self.classes_, row)}
            results.append(prob_dict)
        return results

    def predict_single(self, text: str) -> ClassificationResponse:
        """Convenience method to classify a single text snippet."""
        pred = self.predict([text])[0]
        probs = self.predict_proba([text])[0]
        confidence = probs[str(pred)] if str(pred) in probs else max(probs.values())

        return ClassificationResponse(
            text=text,
            predicted_label=pred,
            probabilities=probs,
            confidence=confidence,
            model_type=self.model_type,
        )

    def evaluate(
        self,
        texts: List[str],
        true_labels: List[Any],
    ) -> ClassificationEvaluationMetrics:
        """Compute standard evaluation metrics on a test or validation set."""
        preds = self.predict(texts)
        classes_str = [str(c) for c in self.classes_]

        acc = float(accuracy_score(true_labels, preds))
        p_macro = float(precision_score(true_labels, preds, average="macro", zero_division=0))
        r_macro = float(recall_score(true_labels, preds, average="macro", zero_division=0))
        f1_macro = float(f1_score(true_labels, preds, average="macro", zero_division=0))

        p_weighted = float(precision_score(true_labels, preds, average="weighted", zero_division=0))
        r_weighted = float(recall_score(true_labels, preds, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(true_labels, preds, average="weighted", zero_division=0))

        cm = confusion_matrix(true_labels, preds, labels=self.classes_).tolist()
        report = classification_report(
            true_labels,
            preds,
            labels=self.classes_,
            target_names=classes_str,
            output_dict=True,
            zero_division=0,
        )

        return ClassificationEvaluationMetrics(
            accuracy=acc,
            precision_macro=p_macro,
            recall_macro=r_macro,
            f1_macro=f1_macro,
            precision_weighted=p_weighted,
            recall_weighted=r_weighted,
            f1_weighted=f1_weighted,
            confusion_matrix=cm,
            classes=classes_str,
            sample_count=len(texts),
            classification_report=report,
        )

    def save(
        self,
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialize trained pipeline and companion metadata."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save an uninitialized pipeline")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        name = model_name or f"{self.model_type}_model"
        model_file = out_path / f"{name}.joblib"
        meta_file = out_path / f"{name}_meta.json"

        joblib.dump(self.pipeline, model_file)

        meta = {
            "model_type": self.model_type,
            "classes": [str(c) for c in (self.classes_ or [])],
            "ngram_range": list(self.ngram_range),
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "stop_words": self.stop_words,
            "c_param": self.c_param,
            "alpha": self.alpha,
            "random_state": self.random_state,
        }
        if extra_metadata:
            meta["extra"] = extra_metadata

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved classifier artifact to %s and metadata to %s", model_file, meta_file)
        return model_file

    @classmethod
    def load(cls, model_file: Union[str, Path]) -> "ClassicalClassifierService":
        """Load a trained pipeline and companion metadata from disk."""
        path = Path(model_file)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        meta_file = path.parent / f"{path.stem}_meta.json"
        meta = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        service = cls(
            model_type=meta.get("model_type", "logistic_regression"),
            ngram_range=tuple(meta.get("ngram_range", [1, 2])),
            max_features=meta.get("max_features", 20000),
            sublinear_tf=meta.get("sublinear_tf", True),
            stop_words=meta.get("stop_words", "english"),
            c_param=meta.get("c_param", 1.0),
            alpha=meta.get("alpha", 1.0),
            random_state=meta.get("random_state", 42),
        )

        service.pipeline = joblib.load(path)
        clf = service.pipeline.named_steps["clf"]
        service.classes_ = list(clf.classes_)
        return service
