from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def safe_division(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> Dict[str, object]:
    label_list = list(labels or sorted(set(y_true) | set(y_pred)))
    total = len(y_true)
    accuracy = safe_division(sum(true == pred for true, pred in zip(y_true, y_pred)), total)
    per_label = {}
    precisions = []
    recalls = []
    f1s = []
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    for label in label_list:
        tp = sum(true == label and pred == label for true, pred in zip(y_true, y_pred))
        fp = sum(true != label and pred == label for true, pred in zip(y_true, y_pred))
        fn = sum(true == label and pred != label for true, pred in zip(y_true, y_pred))
        precision = safe_division(tp, tp + fp)
        recall = safe_division(tp, tp + fn)
        f1 = safe_division(2 * precision * recall, precision + recall)
        support = sum(true == label for true in y_true)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "weighted_precision": safe_division(weighted_precision, total),
        "weighted_recall": safe_division(weighted_recall, total),
        "weighted_f1": safe_division(weighted_f1, total),
        "per_label": per_label,
        "labels": label_list,
    }


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> List[List[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true, pred in zip(y_true, y_pred):
        matrix[index[true]][index[pred]] += 1
    return matrix


def bio_to_spans(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag == "O" or tag == "":
            index += 1
            continue
        if "-" not in tag:
            index += 1
            continue
        prefix, label = tag.split("-", 1)
        if prefix not in {"B", "I"}:
            index += 1
            continue
        start = index
        index += 1
        while index < len(tags):
            next_tag = tags[index]
            if next_tag == f"I-{label}":
                index += 1
                continue
            break
        spans.append((label, start, index - 1))
    return spans


def span_overlap(a: Tuple[str, int, int], b: Tuple[str, int, int]) -> bool:
    return a[0] == b[0] and not (a[2] < b[1] or b[2] < a[1])


def entity_metrics(gold_sequences: Sequence[Sequence[str]], pred_sequences: Sequence[Sequence[str]]) -> Dict[str, object]:
    strict_true_positive = 0
    partial_true_positive = 0
    gold_total = 0
    pred_total = 0
    per_entity: Dict[str, Dict[str, float]] = {}

    def ensure_bucket(label: str) -> Dict[str, float]:
        if label not in per_entity:
            per_entity[label] = {
                "strict_tp": 0.0,
                "strict_fp": 0.0,
                "strict_fn": 0.0,
                "partial_tp": 0.0,
                "partial_fp": 0.0,
                "partial_fn": 0.0,
                "support": 0.0,
            }
        return per_entity[label]

    for gold_tags, pred_tags in zip(gold_sequences, pred_sequences):
        gold_spans = bio_to_spans(gold_tags)
        pred_spans = bio_to_spans(pred_tags)
        gold_total += len(gold_spans)
        pred_total += len(pred_spans)

        for label, _, _ in gold_spans:
            ensure_bucket(label)["support"] += 1

        gold_set = set(gold_spans)
        pred_set = set(pred_spans)
        strict_true_positive += len(gold_set & pred_set)

        for label in {label for label, _, _ in gold_spans} | {label for label, _, _ in pred_spans}:
            bucket = ensure_bucket(label)
            gold_label_spans = [span for span in gold_spans if span[0] == label]
            pred_label_spans = [span for span in pred_spans if span[0] == label]
            gold_label_set = set(gold_label_spans)
            pred_label_set = set(pred_label_spans)
            bucket["strict_tp"] += len(gold_label_set & pred_label_set)
            bucket["strict_fp"] += len(pred_label_set - gold_label_set)
            bucket["strict_fn"] += len(gold_label_set - pred_label_set)

        matched_gold = set()
        for pred_span in pred_spans:
            for gold_index, gold_span in enumerate(gold_spans):
                if gold_index in matched_gold:
                    continue
                if span_overlap(pred_span, gold_span):
                    matched_gold.add(gold_index)
                    partial_true_positive += 1
                    ensure_bucket(pred_span[0])["partial_tp"] += 1
                    break

        for label in {label for label, _, _ in gold_spans} | {label for label, _, _ in pred_spans}:
            bucket = ensure_bucket(label)
            gold_label_count = sum(1 for span in gold_spans if span[0] == label)
            pred_label_count = sum(1 for span in pred_spans if span[0] == label)
            bucket["partial_fn"] += max(0, gold_label_count - sum(1 for idx in matched_gold if gold_spans[idx][0] == label))
            bucket["partial_fp"] += max(0, pred_label_count - sum(1 for span in pred_spans if span[0] == label and any(span_overlap(span, gold_span) for gold_span in gold_spans if gold_span[0] == label)))

    strict_precision = safe_division(strict_true_positive, pred_total)
    strict_recall = safe_division(strict_true_positive, gold_total)
    strict_f1 = safe_division(2 * strict_precision * strict_recall, strict_precision + strict_recall)

    partial_precision = safe_division(partial_true_positive, pred_total)
    partial_recall = safe_division(partial_true_positive, gold_total)
    partial_f1 = safe_division(2 * partial_precision * partial_recall, partial_precision + partial_recall)

    per_entity_report: Dict[str, Dict[str, float]] = {}
    for label, bucket in per_entity.items():
        strict_precision_label = safe_division(bucket["strict_tp"], bucket["strict_tp"] + bucket["strict_fp"])
        strict_recall_label = safe_division(bucket["strict_tp"], bucket["strict_tp"] + bucket["strict_fn"])
        strict_f1_label = safe_division(2 * strict_precision_label * strict_recall_label, strict_precision_label + strict_recall_label)
        partial_precision_label = safe_division(bucket["partial_tp"], bucket["partial_tp"] + bucket["partial_fp"])
        partial_recall_label = safe_division(bucket["partial_tp"], bucket["partial_tp"] + bucket["partial_fn"])
        partial_f1_label = safe_division(2 * partial_precision_label * partial_recall_label, partial_precision_label + partial_recall_label)
        per_entity_report[label] = {
            "strict_precision": strict_precision_label,
            "strict_recall": strict_recall_label,
            "strict_f1": strict_f1_label,
            "partial_precision": partial_precision_label,
            "partial_recall": partial_recall_label,
            "partial_f1": partial_f1_label,
            "support": bucket["support"],
        }

    return {
        "strict_precision": strict_precision,
        "strict_recall": strict_recall,
        "strict_f1": strict_f1,
        "partial_precision": partial_precision,
        "partial_recall": partial_recall,
        "partial_f1": partial_f1,
        "per_entity": per_entity_report,
    }


def model_size_mb(model) -> float:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return parameters * 4 / (1024 ** 2)


def summarize_latencies(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "throughput": 0.0}
    ordered = sorted(latencies)
    count = len(latencies)
    p95_index = min(count - 1, max(0, int(round(count * 0.95)) - 1))
    total_time = sum(latencies)
    return {
        "mean": total_time / count,
        "median": ordered[count // 2],
        "p95": ordered[p95_index],
        "throughput": count / total_time if total_time > 0 else 0.0,
    }