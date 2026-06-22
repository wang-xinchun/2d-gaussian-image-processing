#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_source(raw: str, *, need_label: bool) -> tuple[str, Path, str]:
    parts = raw.split(":")
    if need_label:
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("expected LABEL:JSON_PATH:RECORD_KEY")
        label, path, key = parts
    else:
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("expected JSON_PATH:RECORD_KEY")
        path, key = parts
        label = Path(path).stem
    return label, Path(path), key


def load_records(source: tuple[str, Path, str]) -> tuple[str, list[dict[str, Any]]]:
    label, path, key = source
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data[key]
    if not isinstance(records, list) or not records:
        raise ValueError(f"empty record list: {path}:{key}")
    return label, records


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 50.0), -50.0)))


def psnr_value(record: dict[str, Any], choose_b: bool) -> float:
    return float(record["psnr_b"] if choose_b else record["psnr_a"])


def metric_value(record: dict[str, Any], choose_b: bool, metric: str) -> float | None:
    key = "metrics_b" if choose_b else "metrics_a"
    metrics = record.get(key)
    if isinstance(metrics, dict) and metric in metrics:
        return float(metrics[metric])
    if metric == "psnr":
        return psnr_value(record, choose_b)
    return None


def summarize_choices(records: list[dict[str, Any]], choices: list[bool]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(records),
        "choose_b_count": sum(choices),
        "oracle_correct": 0,
    }
    metric_values: dict[str, list[float]] = {}
    for record, choose_b in zip(records, choices):
        if (choose_b and record["psnr_b"] > record["psnr_a"]) or (
            (not choose_b) and record["psnr_a"] >= record["psnr_b"]
        ):
            result["oracle_correct"] += 1
        metric_names = {"psnr"}
        for key in ("metrics_a", "metrics_b"):
            metrics = record.get(key)
            if isinstance(metrics, dict):
                metric_names.update(metrics)
        for metric in sorted(metric_names):
            value = metric_value(record, choose_b, metric)
            if value is not None:
                metric_values.setdefault(metric, []).append(value)
    for metric, values in metric_values.items():
        result[f"{metric}_mean"] = sum(values) / len(values)
    return result


def constant_choices(records: list[dict[str, Any]], choose_b: bool) -> list[bool]:
    return [choose_b] * len(records)


def oracle_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    choices = [record["psnr_b"] > record["psnr_a"] for record in records]
    result = summarize_choices(records, choices)
    result["choose_b_count"] = sum(choices)
    return result


def fixed_threshold_choices(
    records: list[dict[str, Any]], feature: str, direction: str, threshold: float
) -> list[bool]:
    choices: list[bool] = []
    for record in records:
        value = float(record["features"][feature])
        choices.append(value >= threshold if direction == "gte" else value <= threshold)
    return choices


def label(record: dict[str, Any]) -> float:
    return 1.0 if record["psnr_b"] > record["psnr_a"] else 0.0


def sample_weight(record: dict[str, Any]) -> float:
    return abs(float(record["psnr_b"]) - float(record["psnr_a"])) + 0.01


def standardizer(records: list[dict[str, Any]], features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    means = {
        feature: sum(float(record["features"][feature]) for record in records) / len(records)
        for feature in features
    }
    stds: dict[str, float] = {}
    for feature in features:
        var = sum((float(record["features"][feature]) - means[feature]) ** 2 for record in records) / len(records)
        stds[feature] = math.sqrt(var) or 1.0
    return means, stds


def vector(record: dict[str, Any], features: list[str], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [1.0] + [
        (float(record["features"][feature]) - means[feature]) / stds[feature]
        for feature in features
    ]


def train_logistic(
    records: list[dict[str, Any]],
    features: list[str],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> dict[str, Any]:
    means, stds = standardizer(records, features)
    weights = [0.0] * (len(features) + 1)
    for _ in range(epochs):
        grad = [0.0] * len(weights)
        for record in records:
            x_vec = vector(record, features, means, stds)
            probability = sigmoid(sum(w * x for w, x in zip(weights, x_vec)))
            error = probability - label(record)
            weight = sample_weight(record)
            for index, value in enumerate(x_vec):
                grad[index] += weight * error * value
        for index in range(len(weights)):
            grad[index] = grad[index] / len(records) + l2 * weights[index]
            weights[index] -= lr * grad[index]
    return {"features": features, "means": means, "stds": stds, "weights": weights}


def logistic_probabilities(records: list[dict[str, Any]], model: dict[str, Any]) -> list[float]:
    features = list(model["features"])
    means = dict(model["means"])
    stds = dict(model["stds"])
    weights = list(model["weights"])
    return [
        sigmoid(sum(w * x for w, x in zip(weights, vector(record, features, means, stds))))
        for record in records
    ]


def choices_from_probabilities(probabilities: list[float], threshold: float) -> list[bool]:
    return [probability >= threshold for probability in probabilities]


def tune_probability_threshold(records: list[dict[str, Any]], probabilities: list[float]) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_summary = summarize_choices(records, choices_from_probabilities(probabilities, best_threshold))
    for index in range(1, 100):
        threshold = index / 100.0
        summary = summarize_choices(records, choices_from_probabilities(probabilities, threshold))
        if summary["psnr_mean"] > best_summary["psnr_mean"]:
            best_threshold = threshold
            best_summary = summary
    return best_threshold, best_summary


def evaluate_record_set(
    records: list[dict[str, Any]],
    *,
    model: dict[str, Any],
    tuned_threshold: float,
    fixed_feature: str | None,
    fixed_direction: str | None,
    fixed_threshold: float | None,
) -> dict[str, Any]:
    probabilities = logistic_probabilities(records, model)
    result = {
        "baseline_a": summarize_choices(records, constant_choices(records, False)),
        "baseline_b": summarize_choices(records, constant_choices(records, True)),
        "binary_oracle": oracle_summary(records),
        "logistic_default": summarize_choices(records, choices_from_probabilities(probabilities, 0.5)),
        "logistic_tuned": summarize_choices(records, choices_from_probabilities(probabilities, tuned_threshold)),
    }
    if fixed_feature is not None and fixed_direction is not None and fixed_threshold is not None:
        result["fixed_rule"] = {
            "feature": fixed_feature,
            "direction": fixed_direction,
            "threshold": fixed_threshold,
            **summarize_choices(
                records,
                fixed_threshold_choices(records, fixed_feature, fixed_direction, fixed_threshold),
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lightweight routers from saved A/B router records.")
    parser.add_argument("--train-source", action="append", required=True, help="JSON_PATH:RECORD_KEY")
    parser.add_argument("--eval-source", action="append", required=True, help="LABEL:JSON_PATH:RECORD_KEY")
    parser.add_argument("--features", nargs="+", default=["input_mean", "input_std"])
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--fixed-rule-feature", default=None)
    parser.add_argument("--fixed-rule-direction", choices=("gte", "lte"), default=None)
    parser.add_argument("--fixed-rule-threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    fixed_rule_args = (args.fixed_rule_feature, args.fixed_rule_direction, args.fixed_rule_threshold)
    if any(value is not None for value in fixed_rule_args) and not all(
        value is not None for value in fixed_rule_args
    ):
        parser.error("--fixed-rule-feature, --fixed-rule-direction, and --fixed-rule-threshold must be provided together")

    train_sources = [parse_source(raw, need_label=False) for raw in args.train_source]
    eval_sources = [parse_source(raw, need_label=True) for raw in args.eval_source]

    train_records: list[dict[str, Any]] = []
    train_source_counts: dict[str, int] = {}
    for source in train_sources:
        label_text, records = load_records(source)
        train_records.extend(records)
        train_source_counts[label_text] = len(records)

    model = train_logistic(train_records, args.features, epochs=args.epochs, lr=args.lr, l2=args.l2)
    train_probabilities = logistic_probabilities(train_records, model)
    tuned_threshold, tuned_train_summary = tune_probability_threshold(train_records, train_probabilities)

    result = {
        "features": args.features,
        "train_source_counts": train_source_counts,
        "train_samples": len(train_records),
        "model": model,
        "tuned_probability_threshold": tuned_threshold,
        "train": evaluate_record_set(
            train_records,
            model=model,
            tuned_threshold=tuned_threshold,
            fixed_feature=args.fixed_rule_feature,
            fixed_direction=args.fixed_rule_direction,
            fixed_threshold=args.fixed_rule_threshold,
        ),
        "tuned_train_summary": tuned_train_summary,
        "eval": {},
    }
    for source in eval_sources:
        label_text, records = load_records(source)
        result["eval"][label_text] = evaluate_record_set(
            records,
            model=model,
            tuned_threshold=tuned_threshold,
            fixed_feature=args.fixed_rule_feature,
            fixed_direction=args.fixed_rule_direction,
            fixed_threshold=args.fixed_rule_threshold,
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
