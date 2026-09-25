import csv
import os
import re
import time
from statistics import mean


FIELDNAMES = [
    "run_id",
    "section",
    "method",
    "model_arg",
    "dataset",
    "feature",
    "client",
    "client_mae",
    "client_mse",
    "client_rmse",
    "client_mape",
    "mae",
    "mse",
    "rmse",
    "mape",
    "elements",
    "abs_error_sum",
    "sq_error_sum",
    "mape_error_sum",
    "mape_elements",
    "sample_mae",
    "sample_mse",
    "sample_rmse",
    "sample_mape",
    "sample_elements",
    "seed",
]


_RUN_ID = os.environ.get("BENCHMARK_RUN_ID") or os.environ.get("FCGCN_RUN_ID") or time.strftime("%Y%m%d_%H%M%S")


def current_run_id():
    return _RUN_ID


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def display_dataset_name(dataset_name):
    if dataset_name in ("PeMS07", "PeMSD7"):
        return "PeMSD7"
    return dataset_name


def _write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    existing_rows = []
    if file_exists:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and reader.fieldnames != FIELDNAMES:
                existing_rows = list(reader)
                file_exists = False

    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({name: row.get(name, "") for name in FIELDNAMES})
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDNAMES})


def _baseline_dirname(method):
    name = str(method or "unknown").strip()
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", name)
    return name or "unknown"


def output_paths(paradigm, method):
    custom_root = os.environ.get("BENCHMARK_OUTPUT_ROOT", "").strip()
    if custom_root:
        log_root = os.path.abspath(custom_root)
    else:
        root = project_root()
        log_root = os.path.join(root, f"{paradigm}_logs", _baseline_dirname(method))
    return (
        os.path.join(log_root, f"training_paradigm_{paradigm}_raw.csv"),
        os.path.join(log_root, f"training_paradigm_{paradigm}_summary.csv"),
    )


def write_raw_result(paradigm, row):
    row = dict(row)
    row.setdefault("run_id", current_run_id())
    row["dataset"] = display_dataset_name(row["dataset"])
    raw_path, _ = output_paths(paradigm, row.get("method"))
    _write_rows(raw_path, [row])


def write_summary_results(paradigm, raw_rows):
    grouped = {}
    for row in raw_rows:
        key = (
            row.get("run_id", current_run_id()),
            row["section"],
            row["method"],
            row["model_arg"],
            display_dataset_name(row["dataset"]),
            row["feature"],
            row["seed"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (run_id, section, method, model_arg, dataset, feature, seed), rows in grouped.items():
        sample_metrics = _sample_level_metrics(rows)
        summary_rows.append({
            "run_id": run_id,
            "section": section,
            "method": method,
            "model_arg": model_arg,
            "dataset": dataset,
            "feature": feature,
            "client": "mean" if paradigm == "local" else "global",
            "client_mae": round(mean(float(r["mae"]) for r in rows), 4),
            "client_mse": round(mean(float(r["mse"]) for r in rows), 4),
            "client_rmse": round(mean(float(r["rmse"]) for r in rows), 4),
            "client_mape": round(mean(float(r["mape"]) for r in rows), 4),
            "mae": round(mean(float(r["mae"]) for r in rows), 4),
            "mse": round(mean(float(r["mse"]) for r in rows), 4),
            "rmse": round(mean(float(r["rmse"]) for r in rows), 4),
            "mape": round(mean(float(r["mape"]) for r in rows), 4),
            "sample_mae": sample_metrics["sample_mae"],
            "sample_mse": sample_metrics["sample_mse"],
            "sample_rmse": sample_metrics["sample_rmse"],
            "sample_mape": sample_metrics["sample_mape"],
            "sample_elements": sample_metrics["sample_elements"],
            "seed": seed,
        })

    for method, rows in _group_by_method(summary_rows).items():
        _, summary_path = output_paths(paradigm, method)
        _write_rows(summary_path, rows)
    return summary_rows


def _to_float(value, default=0.0):
    if value in ("", None):
        return default
    return float(value)


def _to_int(value, default=0):
    if value in ("", None):
        return default
    return int(float(value))


def _sample_level_metrics(rows):
    total_elements = sum(_to_int(r.get("elements")) for r in rows)
    total_abs = sum(_to_float(r.get("abs_error_sum")) for r in rows)
    total_sq = sum(_to_float(r.get("sq_error_sum")) for r in rows)
    total_mape = sum(_to_float(r.get("mape_error_sum")) for r in rows)
    total_mape_elements = sum(_to_int(r.get("mape_elements")) for r in rows)

    if total_elements == 0:
        total_elements = len(rows)
        total_abs = sum(float(r["mae"]) for r in rows)
        total_sq = sum(float(r["mse"]) for r in rows)
        total_mape = sum(float(r["mape"]) for r in rows)
        total_mape_elements = len(rows)

    sample_mse = total_sq / max(total_elements, 1)
    return {
        "sample_mae": round(total_abs / max(total_elements, 1), 4),
        "sample_mse": round(sample_mse, 4),
        "sample_rmse": round(sample_mse ** 0.5, 4),
        "sample_mape": round(total_mape / total_mape_elements, 4) if total_mape_elements else 0.0,
        "sample_elements": total_elements,
    }


def row_sample_metrics(row):
    total_elements = _to_int(row.get("elements"))
    total_abs = _to_float(row.get("abs_error_sum"))
    total_sq = _to_float(row.get("sq_error_sum"))
    total_mape = _to_float(row.get("mape_error_sum"))
    total_mape_elements = _to_int(row.get("mape_elements"))

    if total_elements <= 0:
        return {
            "sample_mae": row.get("mae", ""),
            "sample_mse": row.get("mse", ""),
            "sample_rmse": row.get("rmse", ""),
            "sample_mape": row.get("mape", ""),
            "sample_elements": "",
        }

    sample_mse = total_sq / max(total_elements, 1)
    return {
        "sample_mae": round(total_abs / max(total_elements, 1), 4),
        "sample_mse": round(sample_mse, 4),
        "sample_rmse": round(sample_mse ** 0.5, 4),
        "sample_mape": round(total_mape / total_mape_elements, 4) if total_mape_elements else 0.0,
        "sample_elements": total_elements,
    }


def _group_by_method(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("method"), []).append(row)
    return grouped
