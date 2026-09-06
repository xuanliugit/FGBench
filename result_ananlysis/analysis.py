#!/usr/bin/env python3
"""Evaluate predictions against the Hugging Face FGBench test split and save as CSV.

Use --models to select models and --output-dir to choose the CSV destination.

e.g., python result_ananlysis/analysis.py --models llama-3.1-8b
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HF_DATASET = "xuan-liu/FGBench"
# Preserve Hugging Face row order: batch custom_id suffixes are global indices.
DATASETS = ("esol", "lipo", "freesolv", "qm9", "hiv", "bace", "bbbp",
            "tox21", "sider", "clintox")
TASK_COUNTS = dict(zip(
    ("single_bool", "single_value", "interaction_bool", "interaction_value",
     "comparison_bool", "comparison_value"),
    (1423, 375, 1425, 350, 2848, 725),
))
# key: display name, source directory, filename, answer parser
MODELS = {
    "gpt-4o": ("GPT-4o", "gpt", "benchmark_result_4o.jsonl", "standard"),
    "o3-mini": ("o3 mini", "gpt", "benchmark_result_o3_mini.jsonl", "standard"),
    "llama-3.1-8b": ("Llama-3.1 8B", "results", "llama3.1-8b.jsonl", "standard"),
    "llama-3.1-70b": ("Llama-3.1 70B", "results", "Llama-3.1-70B-Instruct.jsonl", "standard"),
    "qwen": ("Qwen2.5-7B", "results", "qwen-7b-Instruct.jsonl", "standard"),
    "chemllm": ("ChemLLM-7B", "results", "ChemLLM-7B-Chat.jsonl", "standard"),
    "nach0": ("nach0-base", "results", "nach0_base.jsonl", "nach0"),
    "molinst": ("Llama-3-8B-MolInst", "results", "molinst_results_cl.jsonl", "molinst"),
    "llasmol": ("LlaSMol-Mistral-7B", "results", "llasmol_results_2.jsonl", "llasmol"),
}
COLUMNS = [f"{task}_{metric}" for task in TASK_COUNTS
           for metric in ("acc" if task.endswith("bool") else "rmse", "valid")]
BOXED = re.compile(r"oxed\{([^\}]+)\}")
SENTENCE = re.compile(
    r"The answer is therefore .*?"
    r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|True|False|true|false).*?\."
)


def scalar(text, yes_no=False):
    """Preserve the notebooks' case-sensitive substring and numeric rules."""
    try:
        return float(text)
    except ValueError:
        if "rue" in text:
            return True
        if "alse" in text:
            return False
        if yes_no:
            if "yes" in text or "Yes" in text:
                return True
            if ("no" in text or "No" in text) and "one" not in text:
                return False
    return None


def extract_answer(text, parser="standard"):
    """Apply eval_all_results cell 1 or nach cells 8, 12, 16 (zero-based)."""
    if not isinstance(text, str):
        raise ValueError("Prediction must be a string")
    match = BOXED.search(text)
    if match:
        return scalar(match.group(1).strip().replace("[", "").replace("]", ""))
    match = SENTENCE.search(text)
    if match:
        return scalar(match.group(1))
    if parser == "molinst":
        text = text.replace("[", "").replace("]", "")
    elif parser == "llasmol":
        for pattern in (r"BOOLEAN>(.*?)<", r"NUMBER>(.*?)<", r"\[(.*?)\]"):
            match = re.search(pattern, text)
            if match:
                text = match.group(1)
                break
    return scalar(text, yes_no=parser == "nach0")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def question_key(question):
    """Remove prompt wrappers and normalize the corrected 'postion' typo."""
    start = question.find("For a ")
    if start < 0:
        raise ValueError("Cannot find benchmark question in prompt")
    question = re.split(r"<\|(?:eot_id|im_end)\|>", question[start:])[0].strip()
    # Saved model prompts predate this spelling correction on Hugging Face.
    return question.replace(" at postion ", " at position ")


def task_name(row):
    task = row["type"].rsplit("_", 1)[0]
    if task not in TASK_COUNTS:
        raise ValueError(f"Unknown question type: {row['type']}")
    return task


def load_ground_truth():
    """Load and validate the Hugging Face test split without reordering its rows."""
    
    rows = list(load_dataset(HF_DATASET, split="test"))
    for index, row in enumerate(rows):
        if row["dataset"] not in DATASETS or row["split"] != "test":
            raise ValueError(f"{HF_DATASET}: row {index}: expected an FGBench test sample")
        gt = scalar(str(row["answer"]))
        if gt is None or not math.isfinite(gt):
            raise ValueError(f"{HF_DATASET}: row {index}: invalid ground-truth answer")
    counts = Counter(task_name(row) for row in rows)
    if counts != TASK_COUNTS:
        raise ValueError(f"Test-split counts differ from Table 4: {dict(counts)}")
    lookup = {question_key(row["question"]): i for i, row in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("Ground truth contains duplicate questions")
    return rows, lookup


def read_llama70b(path):
    """
    Read the multiline export.
    """
    content = path.read_text(encoding="utf-8")
    blocks = re.split(r'(?m)^(?=\{"question":)', content)
    for number, block in enumerate((b for b in blocks if b.strip()), 1):
        try:
            question, tail = block.split('", "output": "', 1)
            _, label = tail.rsplit('", "answer": ', 1)
            label = json.loads(label.strip()[:-1])  # Drop closing object brace.
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{path}: malformed 70B record {number}") from exc
        lines = [line for line in block.splitlines(keepends=True) if '"answer":' in line]
        if len(lines) != 1:
            raise ValueError(f"{path}: record {number} has {len(lines)} answer lines")
        yield {"question": question, "label": label,
               "prediction": lines[0]}


def align_predictions(path, model, truth, lookup):
    """Join by question or batch custom_id; reject missing/duplicate/unknown rows."""
    predictions = {}
    records = (read_llama70b(path) if model == "llama-3.1-70b"
               else read_jsonl(path))
    for number, row in enumerate(records, 1):
        try:
            if MODELS[model][1] == "gpt":
                dataset, index = row["custom_id"].rsplit("_", 1)
                index = int(index)
                if not 0 <= index < len(truth) or truth[index]["dataset"] != dataset:
                    raise ValueError(f"Unknown custom_id: {row['custom_id']}")
                response = row["response"]
                if row.get("error") or response["status_code"] != 200:
                    raise ValueError(f"Failed batch response: {row['custom_id']}")
                prediction = response["body"]["choices"][0]["message"]["content"]
            else:
                index = lookup[question_key(row["question"])]
                field = ("prediction" if model == "llama-3.1-70b" else
                         "output" if model == "chemllm" else "answer")
                prediction = row[field]
                label = row.get("answer") if model == "chemllm" else row.get("label")
                if label is not None:
                    embedded, gt = scalar(str(label)), scalar(str(truth[index]["answer"]))
                    if embedded is None or not math.isclose(embedded, gt, rel_tol=0, abs_tol=1e-12):
                        raise ValueError("Embedded ground-truth label disagrees with test data")
            if index in predictions:
                raise ValueError(f"Duplicate prediction for test index {index}")
            if not isinstance(prediction, str):
                raise ValueError("Prediction must be a string")
            predictions[index] = prediction
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            raise ValueError(f"{path}: record {number}: {exc}") from exc
    if len(predictions) != len(truth):
        raise ValueError(f"{path}: expected {len(truth)} unique predictions, got {len(predictions)}")
    return [predictions[i] for i in range(len(truth))]


def evaluate(truth, predictions, parser):
    groups = {task: [] for task in TASK_COUNTS}
    for index, (row, text) in enumerate(zip(truth, predictions)):
        gt = scalar(str(row["answer"]))
        pred = extract_answer(text, parser)
        if pred is not None and not math.isfinite(pred):
            raise ValueError(f"Non-finite prediction at test index {index}")
        groups[task_name(row)].append((gt, pred))
    metrics = {}
    for task, pairs in groups.items():
        total = len(pairs)
        if task.endswith("bool"):
            valid = sum(pred is not None for _, pred in pairs)
            correct = sum(gt == pred for gt, pred in pairs)
            metrics[task] = {"n": total, "n_valid": valid, "n_correct": correct,
                             "acc": correct / total, "valid": valid / total}
        else:
            # Historical rule: zero gt/pred is excluded; bool is a Python number.
            valid_pairs = [(gt, pred) for gt, pred in pairs
                           if pred and gt and isinstance(pred, (int, float))
                           and isinstance(gt, (int, float))]
            squared = [(gt - pred) ** 2 for gt, pred in valid_pairs]
            rmse = math.sqrt(math.fsum(squared) / len(squared)) if squared else None
            metrics[task] = {"n": total, "n_valid": len(valid_pairs), "rmse": rmse,
                             "valid": len(valid_pairs) / total}
    return metrics


def rounded(value):
    return "NA" if value is None else f"{value:.3f}"


def flatten(metrics):
    return {f"{task}_{metric}": rounded(result[metric])
            for task, result in metrics.items()
            for metric in ("acc" if task.endswith("bool") else "rmse", "valid")}


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "data/results")
    parser.add_argument("--gpt-results-dir", type=Path, default=ROOT / "data/gpt_request")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args(argv)
    selected = [model for model in MODELS if model in args.models]
    try:
        truth, lookup = load_ground_truth()
        table = []
        for model in selected:
            name, source, filename, answer_parser = MODELS[model]
            directory = args.gpt_results_dir if source == "gpt" else args.results_dir
            path = directory / filename
            predictions = align_predictions(path, model, truth, lookup)
            metrics = evaluate(truth, predictions, answer_parser)
            flat = flatten(metrics)
            table.append({"model": name, **flat})
    except (OSError, ValueError, KeyError, OverflowError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_dir / "result.csv", ["model"] + COLUMNS, table)
    except OSError as exc:
        parser.exit(2, f"Error writing report: {exc}\n")
    print(f"CSV file written to {(args.output_dir / 'result.csv').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
