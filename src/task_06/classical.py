"""Task 6 - non-CNN character recognition with HOG and SVM/nearest-centroid.

An RBF SVM is used when scikit-learn is available. A deterministic nearest
centroid classifier is retained as a dependency-free fallback so the project
can still run in a minimal environment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from common.datasets import discover_classification_dataset, stratified_split
from common.features import extract_features
from common.image_io import load_grayscale, require_numpy


def _make_classifier(seed: int = 42):
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed)), "rbf_svm"
    except ImportError:
        return None, "nearest_centroid"


class NearestCentroid:
    def fit(self, features, labels):
        np = require_numpy()
        self.classes_ = np.unique(labels)
        self.centroids_ = np.stack([features[labels == label].mean(axis=0) for label in self.classes_])
        return self

    def predict(self, features):
        np = require_numpy()
        distances = ((features[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(distances, axis=1)]


def train(root, output, test_fraction: float = 0.2, seed: int = 42):
    images, labels, class_names = discover_classification_dataset(root)
    features = extract_features(images)
    train_indices, test_indices = stratified_split(labels, test_fraction, seed)
    classifier, method = _make_classifier(seed)
    if classifier is None:
        classifier = NearestCentroid()
    start = time.perf_counter()
    classifier.fit(features[train_indices], labels[train_indices])
    train_seconds = time.perf_counter() - start
    predictions = classifier.predict(features[test_indices])
    accuracy = float((predictions == labels[test_indices]).mean())
    model = {"classifier": classifier, "classes": class_names, "method": method, "feature": "32x32 HOG, cell=4, bins=9", "seed": seed}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(model, handle)
    return {"method": method, "classes": class_names, "train_samples": len(train_indices), "test_samples": len(test_indices), "accuracy": accuracy, "train_seconds": train_seconds}


def predict(model_path, image_path):
    with Path(model_path).open("rb") as handle:
        model = pickle.load(handle)
    prediction = int(model["classifier"].predict(extract_features([load_grayscale(image_path)]))[0])
    return model["classes"][prediction]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--model", required=True)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--metrics")
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    args = parser.parse_args()
    if args.command == "train":
        metrics = train(args.data, args.model, args.test_fraction, args.seed)
        if args.metrics:
            Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
            Path(args.metrics).write_text(json.dumps(metrics, indent=2) + "\n")
        else:
            print(json.dumps(metrics, indent=2))
    else:
        print(predict(args.model, args.input))


if __name__ == "__main__":
    main()
