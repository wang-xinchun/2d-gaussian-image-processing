#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_FIXED_FEATURE = "input_std"
DEFAULT_FIXED_DIRECTION = "lte"
DEFAULT_FIXED_THRESHOLD = 0.07026463374495506


def parse_source(raw: str) -> tuple[str, Path]:
    parts = raw.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("expected LABEL:JSON_PATH")
    return parts[0], Path(parts[1])


def load_source(source: tuple[str, Path]) -> dict[str, Any]:
    label, path = source
    data = json.loads(path.read_text(encoding="utf-8"))
    val_records = data.get("val_records")
    test_records = data.get("test_records")
    if not isinstance(val_records, list) or not val_records:
        raise ValueError(f"missing non-empty val_records: {path}")
    if not isinstance(test_records, list) or not test_records:
        raise ValueError(f"missing non-empty test_records: {path}")
    return {"label": label, "path": str(path), "val_records": val_records, "test_records": test_records}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 50.0), -50.0)))


def choose_oracle_b(record: dict[str, Any]) -> bool:
    return float(record["psnr_b"]) > float(record["psnr_a"])


def metric_value(record: dict[str, Any], choose_b: bool, metric: str) -> float | None:
    if metric == "psnr":
        return float(record["psnr_b"] if choose_b else record["psnr_a"])
    metrics = record.get("metrics_b" if choose_b else "metrics_a")
    if isinstance(metrics, dict) and metric in metrics:
        return float(metrics[metric])
    return None


def summarize_choices(records: list[dict[str, Any]], choices: list[bool]) -> dict[str, Any]:
    if len(records) != len(choices):
        raise ValueError("records and choices length mismatch")
    result: dict[str, Any] = {
        "samples": len(records),
        "choose_b_count": sum(choices),
        "oracle_correct": 0,
    }
    metric_values: dict[str, list[float]] = {}
    oracle_psnr = []
    for record, choose_b in zip(records, choices):
        oracle_choice = choose_oracle_b(record)
        if choose_b == oracle_choice:
            result["oracle_correct"] += 1
        oracle_psnr.append(float(record["psnr_b"] if oracle_choice else record["psnr_a"]))
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
        result[f"{metric}_samples"] = len(values)
    result["oracle_correct_fraction"] = result["oracle_correct"] / len(records)
    result["oracle_psnr_mean"] = sum(oracle_psnr) / len(oracle_psnr)
    result["psnr_regret_to_oracle"] = result["oracle_psnr_mean"] - result["psnr_mean"]
    return result


def constant_choices(records: list[dict[str, Any]], choose_b: bool) -> list[bool]:
    return [choose_b] * len(records)


def fixed_threshold_choices(
    records: list[dict[str, Any]],
    *,
    feature: str,
    direction: str,
    threshold: float,
) -> list[bool]:
    choices: list[bool] = []
    for record in records:
        value = float(record["features"][feature])
        choices.append(value <= threshold if direction == "lte" else value >= threshold)
    return choices


def standardizer(
    records: list[dict[str, Any]],
    features: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    means = {
        feature: sum(float(record["features"][feature]) for record in records) / len(records)
        for feature in features
    }
    stds: dict[str, float] = {}
    for feature in features:
        var = sum((float(record["features"][feature]) - means[feature]) ** 2 for record in records) / len(records)
        stds[feature] = math.sqrt(var) or 1.0
    return means, stds


def vector(
    record: dict[str, Any],
    features: list[str],
    means: dict[str, float],
    stds: dict[str, float],
) -> list[float]:
    return [1.0] + [
        (float(record["features"][feature]) - means[feature]) / stds[feature]
        for feature in features
    ]


def sample_weight(record: dict[str, Any]) -> float:
    return abs(float(record["psnr_b"]) - float(record["psnr_a"])) + 0.01


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
            error = probability - (1.0 if choose_oracle_b(record) else 0.0)
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


def tune_probability_threshold(
    records: list[dict[str, Any]],
    probabilities: list[float],
) -> tuple[float, dict[str, Any]]:
    candidates = sorted({0.5, *[index / 100.0 for index in range(1, 100)], *probabilities})
    best_threshold = 0.5
    best_summary = summarize_choices(records, choices_from_probabilities(probabilities, best_threshold))
    for threshold in candidates:
        summary = summarize_choices(records, choices_from_probabilities(probabilities, threshold))
        if summary["psnr_mean"] > best_summary["psnr_mean"]:
            best_threshold = threshold
            best_summary = summary
    return best_threshold, best_summary


def feature_sets(common_features: set[str]) -> dict[str, list[str]]:
    candidates = {
        "input_std": ["input_std"],
        "input_mean_std": ["input_mean", "input_std"],
        "input_residual": [
            "input_mean",
            "input_std",
            "input_a_l1",
            "input_b_l1",
            "input_a_mse",
            "input_b_mse",
            "input_residual_l1_delta_b_minus_a",
        ],
        "prediction_stats": [
            "input_mean",
            "input_std",
            "pred_a_mean",
            "pred_a_std",
            "pred_b_mean",
            "pred_b_std",
            "pred_ab_l1",
            "pred_ab_mse",
        ],
        "control_stats": [
            "input_mean",
            "input_std",
            "a_control_canvas_mean",
            "a_control_canvas_std",
            "a_control_feedback_mean",
            "a_control_feedback_std",
            "a_guidance_blend_mean",
            "a_guidance_blend_std",
            "b_control_canvas_mean",
            "b_control_canvas_std",
            "b_control_feedback_mean",
            "b_control_feedback_std",
            "b_guidance_blend_mean",
            "b_guidance_blend_std",
        ],
        "all_common": sorted(common_features),
    }
    return {
        name: features
        for name, features in candidates.items()
        if all(feature in common_features for feature in features)
    }


def evaluate_fixed(
    records: list[dict[str, Any]],
    *,
    feature: str,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    return {
        "feature": feature,
        "direction": direction,
        "threshold": threshold,
        **summarize_choices(
            records,
            fixed_threshold_choices(records, feature=feature, direction=direction, threshold=threshold),
        ),
    }


def evaluate_model(records: list[dict[str, Any]], model: dict[str, Any], threshold: float) -> dict[str, Any]:
    probabilities = logistic_probabilities(records, model)
    return summarize_choices(records, choices_from_probabilities(probabilities, threshold))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-block validation for saved A08/A15 router records.")
    parser.add_argument("--source", action="append", type=parse_source, required=True, help="LABEL:JSON_PATH")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--fixed-rule-feature", default=DEFAULT_FIXED_FEATURE)
    parser.add_argument("--fixed-rule-direction", choices=("gte", "lte"), default=DEFAULT_FIXED_DIRECTION)
    parser.add_argument("--fixed-rule-threshold", type=float, default=DEFAULT_FIXED_THRESHOLD)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sources = [load_source(source) for source in args.source]
    common_features: set[str] | None = None
    for source in sources:
        for split in ("val_records", "test_records"):
            for record in source[split]:
                keys = set(record["features"])
                common_features = keys if common_features is None else common_features & keys
    if not common_features:
        raise ValueError("no common features found")
    sets = feature_sets(common_features)

    result: dict[str, Any] = {
        "sources": [{"label": source["label"], "path": source["path"]} for source in sources],
        "common_features": sorted(common_features),
        "feature_sets": sets,
        "epochs": args.epochs,
        "lr": args.lr,
        "l2": args.l2,
        "fixed_rule": {
            "feature": args.fixed_rule_feature,
            "direction": args.fixed_rule_direction,
            "threshold": args.fixed_rule_threshold,
        },
        "leave_one_source_out": {},
        "aggregate_loso_test": {},
        "all_val_to_all_test": {},
    }

    aggregate_records: dict[str, list[dict[str, Any]]] = {"fixed_rule": []}
    aggregate_choices: dict[str, list[bool]] = {"fixed_rule": []}
    for name in sets:
        aggregate_records[f"logistic_{name}"] = []
        aggregate_choices[f"logistic_{name}"] = []

    for holdout in sources:
        holdout_label = holdout["label"]
        train_records = [
            record
            for source in sources
            if source["label"] != holdout_label
            for record in source["val_records"]
        ]
        test_records = list(holdout["test_records"])
        fold: dict[str, Any] = {
            "train_sources": [source["label"] for source in sources if source["label"] != holdout_label],
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "baseline_a": summarize_choices(test_records, constant_choices(test_records, False)),
            "baseline_b": summarize_choices(test_records, constant_choices(test_records, True)),
            "binary_oracle": summarize_choices(test_records, [choose_oracle_b(record) for record in test_records]),
            "fixed_rule": evaluate_fixed(
                test_records,
                feature=args.fixed_rule_feature,
                direction=args.fixed_rule_direction,
                threshold=args.fixed_rule_threshold,
            ),
            "logistic": {},
        }
        fixed_choices = fixed_threshold_choices(
            test_records,
            feature=args.fixed_rule_feature,
            direction=args.fixed_rule_direction,
            threshold=args.fixed_rule_threshold,
        )
        aggregate_records["fixed_rule"].extend(test_records)
        aggregate_choices["fixed_rule"].extend(fixed_choices)
        for name, features in sets.items():
            model = train_logistic(train_records, features, epochs=args.epochs, lr=args.lr, l2=args.l2)
            train_probabilities = logistic_probabilities(train_records, model)
            tuned_threshold, tuned_train = tune_probability_threshold(train_records, train_probabilities)
            test_probabilities = logistic_probabilities(test_records, model)
            test_choices = choices_from_probabilities(test_probabilities, tuned_threshold)
            fold["logistic"][name] = {
                "features": features,
                "threshold": tuned_threshold,
                "tuned_train": tuned_train,
                "test": summarize_choices(test_records, test_choices),
                "weights": model["weights"],
            }
            aggregate_records[f"logistic_{name}"].extend(test_records)
            aggregate_choices[f"logistic_{name}"].extend(test_choices)
        result["leave_one_source_out"][holdout_label] = fold

    for method, records in aggregate_records.items():
        result["aggregate_loso_test"][method] = summarize_choices(records, aggregate_choices[method])

    all_val_records = [record for source in sources for record in source["val_records"]]
    all_test_records = [record for source in sources for record in source["test_records"]]
    result["all_val_to_all_test"]["baseline_a"] = summarize_choices(all_test_records, constant_choices(all_test_records, False))
    result["all_val_to_all_test"]["baseline_b"] = summarize_choices(all_test_records, constant_choices(all_test_records, True))
    result["all_val_to_all_test"]["binary_oracle"] = summarize_choices(
        all_test_records,
        [choose_oracle_b(record) for record in all_test_records],
    )
    result["all_val_to_all_test"]["fixed_rule"] = evaluate_fixed(
        all_test_records,
        feature=args.fixed_rule_feature,
        direction=args.fixed_rule_direction,
        threshold=args.fixed_rule_threshold,
    )
    result["all_val_to_all_test"]["logistic"] = {}
    for name, features in sets.items():
        model = train_logistic(all_val_records, features, epochs=args.epochs, lr=args.lr, l2=args.l2)
        train_probabilities = logistic_probabilities(all_val_records, model)
        tuned_threshold, tuned_train = tune_probability_threshold(all_val_records, train_probabilities)
        result["all_val_to_all_test"]["logistic"][name] = {
            "features": features,
            "threshold": tuned_threshold,
            "tuned_train": tuned_train,
            "test": evaluate_model(all_test_records, model, tuned_threshold),
            "weights": model["weights"],
        }

    leaderboard = []
    for method, summary in result["aggregate_loso_test"].items():
        leaderboard.append(
            {
                "method": method,
                "psnr_mean": summary["psnr_mean"],
                "oracle_correct_fraction": summary["oracle_correct_fraction"],
                "choose_b_count": summary["choose_b_count"],
                "samples": summary["samples"],
            }
        )
    leaderboard.sort(key=lambda item: item["psnr_mean"], reverse=True)
    result["aggregate_loso_leaderboard"] = leaderboard

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
