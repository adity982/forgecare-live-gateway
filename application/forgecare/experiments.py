"""Runnable curriculum comparisons with explicit synthetic/research provenance."""
from __future__ import annotations

import json
import math
import uuid
import time
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, make_regression
from sklearn.decomposition import FastICA, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .db import audit, connect, now
from .vision import candidates, scores
from .rag_eval import evaluate as evaluate_rag


def classification_experiment() -> dict:
    digits = load_digits()
    X = digits.data.astype(np.float32) / 16
    y = (digits.target >= 5).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
    findings = {}
    for name, model in candidates().items():
        try:
            model.fit(X_train, y_train)
            start = time.perf_counter()
            predicted = model.predict(X_test)
            latency = (time.perf_counter() - start) * 1000 / len(X_test)
            raw_scores = scores(model, X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
            findings[name] = {"test_accuracy": round(float(accuracy_score(y_test, predicted)), 4),
                              "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
                              "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
                              "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
                              "roc_auc": round(float(roc_auc_score(y_test, raw_scores)), 4),
                              "average_precision": round(float(average_precision_score(y_test, raw_scores)), 4),
                              "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                              "latency_ms_per_image": round(latency, 4)}
        except Exception as exc:
            findings[name] = {"error": str(exc)}
    return {"dataset": "sklearn_digits_research_not_phone_data", "task": "binary_digit_classification",
            "split": "stratified_75_25_seed_42", "train": len(y_train), "test": len(y_test), "results": findings}


def regression_experiment() -> dict:
    X, y = make_regression(n_samples=500, n_features=8, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)
    models = {"median": None, "linear": LinearRegression(),
              "polynomial": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
              "knn": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7))}
    results = {}
    for name, model in models.items():
        if model is None:
            predicted = np.full_like(y_test, np.median(y_train))
        else:
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
        results[name] = {"mae": round(float(mean_absolute_error(y_test, predicted)), 4),
                         "rmse": round(float(np.sqrt(mean_squared_error(y_test, predicted))), 4)}
    return {"dataset": "synthetic_learning_fixture", "operational_duration_model": "insufficient_real_duration_labels", "results": results}


def unsupervised_experiment() -> dict:
    digits = load_digits()
    X = digits.data.astype(np.float32)
    train, test = train_test_split(np.arange(len(X)), test_size=.25, random_state=42,
                                   stratify=digits.target)
    scaler = StandardScaler().fit(X[train])
    X_train, X_test = scaler.transform(X[train]), scaler.transform(X[test])
    pca = PCA(n_components=12, random_state=42).fit(X_train)
    reduced = pca.transform(X_train)
    ica = FastICA(n_components=12, random_state=42, max_iter=300, tol=.01)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            separated = ica.fit_transform(X_train)
        ica_status = {"components": int(separated.shape[1]), "finite": bool(np.isfinite(separated).all()),
                      "converged": not any(issubclass(w.category, ConvergenceWarning) for w in captured)}
    except Exception as exc:
        ica_status = {"error": str(exc)}
    clusters = KMeans(n_clusters=10, random_state=42, n_init=5).fit(reduced)
    full_classifier = LogisticRegression(max_iter=500).fit(X_train, digits.target[train])
    pca_classifier = LogisticRegression(max_iter=500).fit(reduced, digits.target[train])
    return {"dataset": "sklearn_digits_research_not_equipment_data", "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "preprocessing": "train_fitted_scaler_and_pca", "split": "stratified_75_25_seed_42",
            "downstream_accuracy": {"full_features": round(float(accuracy_score(digits.target[test], full_classifier.predict(X_test))), 4),
                                    "pca_12": round(float(accuracy_score(digits.target[test], pca_classifier.predict(pca.transform(X_test)))), 4)},
            "ica": ica_status, "kmeans_inertia": round(float(clusters.inertia_), 4),
            "silhouette": round(float(silhouette_score(reduced, clusters.labels_, sample_size=500, random_state=42)), 4),
            "clusters_are_not_defect_labels": True}


def time_series_experiment() -> dict:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.stattools import acf, adfuller, pacf
    from statsmodels.tsa.seasonal import MSTL, seasonal_decompose

    t = np.arange(160)
    rng = np.random.default_rng(42)
    known_calendar = np.cos(2*np.pi*t/7)
    series = 40 + .05*t + 3*np.sin(2*np.pi*t/12) + 1.5*known_calendar + rng.normal(0, .4, len(t))
    train, test = series[:136], series[136:]
    naive = np.full_like(test, train[-1])
    candidates = {}
    for order in ((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = ARIMA(train, order=order).fit()
            candidates[order] = fitted.aic
        except Exception:
            pass
    selected_order = min(candidates, key=candidates.get)
    arima = ARIMA(train, order=selected_order).fit().forecast(len(test))
    ets_candidates = {}
    for trend in (None, "add"):
        fitted = ExponentialSmoothing(train, trend=trend, seasonal="add", seasonal_periods=12).fit()
        ets_candidates[trend or "none"] = fitted.aic
    selected_trend = min(ets_candidates, key=ets_candidates.get)
    ets = ExponentialSmoothing(train, trend=None if selected_trend == "none" else selected_trend,
                               seasonal="add", seasonal_periods=12).fit().forecast(len(test))
    arimax = ARIMA(train, order=(1, 1, 1), exog=known_calendar[:136]).fit().forecast(len(test), exog=known_calendar[136:])
    mstl = MSTL(train, periods=(12,)).fit()
    seasonal = np.asarray(mstl.seasonal).reshape(-1)
    mstl_forecast = np.asarray(mstl.trend)[-1] + np.resize(seasonal[-12:], len(test))
    decomposition = seasonal_decompose(train, period=12)
    predictions = {"naive": naive, "auto_arima_grid": arima, "auto_ets_grid": ets,
                   "arimax_known_calendar": arimax, "mstl_seasonal_naive": mstl_forecast}
    walk_forward = []
    for end in (88, 100, 112, 124):
        actual = series[end:end+12]
        baseline = np.full_like(actual, series[end-1])
        fitted = ARIMA(series[:end], order=selected_order).fit()
        forecast = fitted.forecast(12)
        walk_forward.append({"train_end": end, "horizon": 12,
                             "naive_mae": round(float(mean_absolute_error(actual, baseline)), 4),
                             "arima_mae": round(float(mean_absolute_error(actual, forecast)), 4)})
    return {"dataset": "synthetic_learning_fixture", "operational_forecast": "insufficient_real_regular_history",
            "horizon": len(test), "adf_pvalue": round(float(adfuller(train, result_object=False)[1]), 5),
            "acf_lag_1": round(float(acf(train, nlags=1)[1]), 4), "pacf_lag_1": round(float(pacf(train, nlags=1)[1]), 4),
            "seasonal_amplitude": round(float(np.ptp(decomposition.seasonal)), 4),
            "auto_arima_order": selected_order, "auto_ets_trend": selected_trend,
            "arimax_exog": "known_future_calendar_only", "walk_forward": walk_forward,
            "mae": {k: round(float(mean_absolute_error(test, v)), 4) for k, v in predictions.items()},
            "rmse": {k: round(float(np.sqrt(mean_squared_error(test, v))), 4) for k, v in predictions.items()}}


def deep_learning_experiment() -> dict:
    import torch
    from torch import nn
    torch.manual_seed(42)
    digits = load_digits()
    X = torch.tensor(digits.images[:, None] / 16, dtype=torch.float32)
    y = torch.tensor((digits.target >= 5).astype(np.int64))
    train, test = train_test_split(np.arange(len(y)), test_size=.2, stratify=y.numpy(), random_state=42)
    architectures = {
        "mlp": nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)),
        "cnn": nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(128, 2)),
    }
    findings = {}
    for name, model in architectures.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        losses = []
        for _ in range(4):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(X[train]), y[train])
            loss.backward()
            optimizer.step()
            losses.append(round(float(loss.detach()), 4))
        with torch.no_grad():
            accuracy = (model(X[test]).argmax(1) == y[test]).float().mean().item()
        findings[name] = {"test_accuracy": round(accuracy, 4), "loss_curve": losses,
                          "parameters": sum(p.numel() for p in model.parameters())}
    # A small sequence task demonstrates RNN forward/backward and optimizer behavior.
    sequence = torch.sin(torch.linspace(0, 8*math.pi, 120)).reshape(-1, 1)
    windows = torch.stack([sequence[i:i+8] for i in range(100)])
    targets = torch.stack([sequence[i+8] for i in range(100)])
    class RNNForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.RNN(1, 8, batch_first=True)
            self.head = nn.Linear(8, 1)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1])
    recurrent = RNNForecast()
    optimizer = torch.optim.RMSprop(recurrent.parameters(), lr=.01)
    for _ in range(8):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(recurrent(windows[:80]), targets[:80])
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        rnn_mae = torch.abs(recurrent(windows[80:]) - targets[80:]).mean().item()
        naive_mae = torch.abs(windows[80:, -1] - targets[80:]).mean().item()
    return {"dataset": "sklearn_digits_and_synthetic_sequence_research", "image_models": findings,
            "rnn_synthetic_mae": round(rnn_mae, 4), "naive_synthetic_mae": round(naive_mae, 4),
            "operational_deep_model": "needs_reviewed_equipment_data"}


def optimization_experiment() -> dict:
    import torch
    from torch import nn
    digits = load_digits()
    X = torch.tensor(digits.data / 16, dtype=torch.float32)
    y = torch.tensor((digits.target >= 5).astype(np.int64))
    train, test = train_test_split(np.arange(len(y)), test_size=.25, stratify=y.numpy(), random_state=42)
    train = train[:600]
    results = {}
    activations = {"sigmoid": nn.Sigmoid, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}
    optimizers = {"sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop, "adam": torch.optim.Adam}
    for activation_name, activation in activations.items():
        for optimizer_name, optimizer_class in optimizers.items():
            torch.manual_seed(42)
            model = nn.Sequential(nn.Linear(64, 24), activation(), nn.Linear(24, 2))
            rate = .05 if optimizer_name == "sgd" else .005
            optimizer = optimizer_class(model.parameters(), lr=rate)
            losses, gradients = [], []
            for _ in range(4):
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(X[train]), y[train])
                loss.backward()
                gradients.append(round(float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))), 4))
                optimizer.step()
                losses.append(round(float(loss.detach()), 4))
            with torch.no_grad():
                accuracy = float((model(X[test]).argmax(1) == y[test]).float().mean())
            results[f"{activation_name}_{optimizer_name}"] = {"learning_rate": rate, "epochs": 4,
                "loss_curve": losses, "gradient_norms": gradients, "test_accuracy": round(accuracy, 4)}
    return {"dataset": "sklearn_digits_research_not_phone_data", "split": "stratified_75_25_seed_42",
            "task": "activation_and_optimizer_comparison", "output": "two_class_logits_cross_entropy",
            "results": results}


TASKS = {"classification": classification_experiment, "regression": regression_experiment,
         "unsupervised": unsupervised_experiment, "time_series": time_series_experiment,
         "deep_learning": deep_learning_experiment, "optimization": optimization_experiment,
         "rag_retrieval": evaluate_rag}


def run_suite(site_id: str, experiment_ids: dict[str, str]) -> None:
    for task, experiment_id in experiment_ids.items():
        try:
            result = TASKS[task](site_id) if task == "rag_retrieval" else TASKS[task]()
            with connect() as db:
                db.execute("UPDATE experiments SET status=?,results_json=?,ended_at=? WHERE id=?",
                           (result.get("status", "complete"), json.dumps(result), now(), experiment_id))
                audit(db, None, site_id, "experiment_complete", "experiment", experiment_id)
        except Exception as exc:
            with connect() as db:
                db.execute("UPDATE experiments SET status='failed',results_json=?,ended_at=? WHERE id=?",
                           (json.dumps({"error": f"{type(exc).__name__}: {exc}"}), now(), experiment_id))
                audit(db, None, site_id, "experiment_failed", "experiment", experiment_id, json.dumps({"error": str(exc)}))
