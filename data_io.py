from typing import List, Tuple, Optional

import math
import os
import csv
from pathlib import Path
import io
from datetime import datetime, date

import pandas as pd
from features import analyze_preprocessing


def _normalize_trailing_separators(raw_bytes: bytes, sep_hint: Optional[str]) -> Tuple[bytes, Optional[str], dict]:
    """
    Ensure rows that are missing a trailing empty field (i.e., missing the final
    separator when the last value is blank) are normalized by appending the
    separator at the end of such lines.

    Returns potentially modified bytes and the (possibly unchanged) separator.
    """
    try:
        text = raw_bytes.decode(errors="ignore")
    except Exception:
        return raw_bytes, sep_hint, {"lines_fixed": 0}

    # Determine separator to use for normalization
    sep = sep_hint if sep_hint in {",", ";", "\t", "|"} else None
    if sep is None:
        # Heuristic: pick the first likely delimiter present in the header line
        first_line = text.splitlines()[0] if text else ""
        for cand in [",", ";", "\t", "|"]:
            if cand in first_line:
                sep = cand
                break
    if not sep:
        return raw_bytes, sep_hint, {"lines_fixed": 0}

    # Keep line endings while processing
    lines = text.splitlines(True)
    if not lines:
        return raw_bytes, sep, {"lines_fixed": 0}

    # Compute expected number of separators per line by majority vote
    from collections import Counter

    def count_sep(s: str) -> int:
        core = s.rstrip("\r\n")
        return core.count(sep)

    counts = [count_sep(l) for l in lines if sep in l]
    if not counts:
        return raw_bytes, sep, {"lines_fixed": 0}
    expected = Counter(counts).most_common(1)[0][0]

    # Normalize lines that are short by exactly one trailing separator
    changed = False
    lines_fixed = 0
    normalized_parts: List[str] = []
    for l in lines:
        # Preserve original line ending
        line_ending = ""
        if l.endswith("\r\n"):
            line_ending = "\r\n"
            core = l[:-2]
        elif l.endswith("\n"):
            line_ending = "\n"
            core = l[:-1]
        elif l.endswith("\r"):
            line_ending = "\r"
            core = l[:-1]
        else:
            core = l

        current_count = core.count(sep)
        if core and (current_count == expected - 1) and (not core.endswith(sep)):
            core = core + sep
            changed = True
            lines_fixed += 1
        normalized_parts.append(core + line_ending)

    if not changed:
        return raw_bytes, sep, {"lines_fixed": 0}

    normalized_text = "".join(normalized_parts)
    try:
        return normalized_text.encode(), sep, {"lines_fixed": lines_fixed}
    except Exception:
        return raw_bytes, sep, {"lines_fixed": 0}

def read_table_any(file_obj_or_path):
    """
    Read an uploaded file strictly as CSV with robust delimiter and header detection.

    - Only CSV is accepted (Excel and other formats are rejected)
    - Detect delimiter using csv.Sniffer with fallback heuristics
    - Read with header=None first, then detect if the first row is a header and
      adjust the DataFrame accordingly

    Returns: (df, info)
      df: pandas.DataFrame (data rows only; header row removed if detected)
      info: dict with keys: file_type, size_mb, separator, header_detected, n_rows, n_cols, error
    """
    info = {
        "file_type": None,
        "size_mb": None,
        "separator": None,
        "header_detected": None,
        "n_rows": None,
        "n_cols": None,
        "error": None,
        # Extra header metadata for UI hints
        "header_renamed_count": 0,
        "header_names": None,
    }

    name = None
    size_bytes = None
    buf_for_sniff = None
    raw_bytes = None
    is_file_like = hasattr(file_obj_or_path, "read") or hasattr(file_obj_or_path, "getvalue")
    if is_file_like:
        name = getattr(file_obj_or_path, "name", None)
        size_bytes = getattr(file_obj_or_path, "size", None)
        # Read a small head sample without consuming the stream permanently
        head_bytes = b""
        try:
            if hasattr(file_obj_or_path, "getvalue"):
                head_bytes = file_obj_or_path.getvalue()[:4096]
            else:
                head_bytes = file_obj_or_path.read(4096)
                try:
                    file_obj_or_path.seek(0)
                except Exception:
                    pass
        except Exception:
            head_bytes = b""
        buf_for_sniff = head_bytes
        # Capture full bytes for normalization
        try:
            if hasattr(file_obj_or_path, "getvalue"):
                raw_bytes = file_obj_or_path.getvalue()
            else:
                raw_bytes = file_obj_or_path.read()
        except Exception:
            raw_bytes = None
        finally:
            try:
                file_obj_or_path.seek(0)
            except Exception:
                pass
    else:
        # Treat as path-like
        path_str = str(file_obj_or_path)
        name = path_str
        try:
            size_bytes = os.path.getsize(path_str)
        except Exception:
            size_bytes = None
        try:
            with open(path_str, "rb") as f:
                buf_for_sniff = f.read(4096)
            with open(path_str, "rb") as f:
                raw_bytes = f.read()
        except Exception:
            buf_for_sniff = b""
            raw_bytes = None

    ext = (Path(name).suffix or "").lower() if name else ""
    info["file_type"] = (ext.replace(".", "") or "unknown")
    info["size_mb"] = (size_bytes / (1024 * 1024)) if size_bytes is not None else None

    if info["size_mb"] is not None and info["size_mb"] > 10.0:
        info["error"] = "File larger than 10 MB"
        return pd.DataFrame(), info

    # We only accept CSV according to the new rules
    if ext not in [".csv", "csv", ""]:
        info["error"] = "Only CSV files are accepted"
        return pd.DataFrame(), info

    # Detect delimiter
    detected_sep = None
    try:
        sample_text = (buf_for_sniff or b"").decode(errors="ignore")
        if sample_text:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=[",", ";", "\t", "|"])
            detected_sep = dialect.delimiter
    except Exception:
        line = sample_text.splitlines()[0] if sample_text else ""
        for cand in [",", ";", "\t", "|"]:
            if cand in line:
                detected_sep = cand
                break
    info["separator"] = detected_sep or "auto"

    # Prepare a BytesIO target, normalizing trailing separators if needed
    if raw_bytes is None:
        read_target = file_obj_or_path
    else:
        normalized_bytes, _, norm_stats = _normalize_trailing_separators(
            raw_bytes if isinstance(raw_bytes, (bytes, bytearray)) else bytes(str(raw_bytes), "utf-8"),
            detected_sep,
        )
        read_target = io.BytesIO(normalized_bytes)
        # Surface normalization stats to info after it's created

    # Read with header=None first; we will detect header manually
    normalization_lines_fixed = 0
    try:
        if detected_sep is None:
            raw_df = pd.read_csv(read_target, sep=None, engine="python", header=None, dtype=str)
        else:
            raw_df = pd.read_csv(read_target, sep=detected_sep, engine="python", header=None, dtype=str)
    except Exception as e:
        info["error"] = f"Read failed: {e}"
        return pd.DataFrame(), info

    # Header detection: if the first cell cannot be parsed as date but almost all following cells in first column can,
    # assume the first row is header
    header_detected = False
    try:
        first_col = raw_df.iloc[:, 0]
        parsed_all = pd.to_datetime(first_col, errors="coerce")
        if len(parsed_all) >= 2:
            first_is_not_date = pd.isna(parsed_all.iloc[0])
            after_ratio = parsed_all.iloc[1:].notna().mean() if len(parsed_all) > 1 else 0.0
            if first_is_not_date and after_ratio >= 0.95:
                header_detected = True
    except Exception:
        header_detected = False

    if header_detected:
        # Use the first row as header names; drop it from data
        header_row = raw_df.iloc[0].tolist()
        df = raw_df.iloc[1:].copy().reset_index(drop=True)
        try:
            # Ensure unique string column names
            dedup = []
            seen = set()
            for idx, val in enumerate(header_row):
                name = str(val).strip() if (val is not None and str(val).strip() != "") else f"col_{idx+1}"
                if name in seen:
                    k = 2
                    new_name = f"{name}_{k}"
                    while new_name in seen:
                        k += 1
                        new_name = f"{name}_{k}"
                    name = new_name
                seen.add(name)
                dedup.append(name)
            df.columns = dedup
            # Track if any renames were needed compared to original header strings (after strip)
            original = [str(v).strip() if v is not None else "" for v in header_row]
            info["header_renamed_count"] = sum(1 for a, b in zip(original, dedup) if a != b)
            info["header_names"] = dedup
        except Exception:
            # Fallback to default names if anything goes wrong
            df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
            info["header_renamed_count"] = 0
            info["header_names"] = df.columns.tolist()
    else:
        # No header line; assign generic names
        df = raw_df.copy()
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        info["header_renamed_count"] = 0
        info["header_names"] = df.columns.tolist()

    info["header_detected"] = bool(header_detected)
    info["n_rows"], info["n_cols"] = int(df.shape[0]), int(df.shape[1])
    # Attach normalization stats if available
    try:
        info["trailing_separator_lines_fixed"] = norm_stats.get("lines_fixed", 0)  # type: ignore[name-defined]
    except Exception:
        info["trailing_separator_lines_fixed"] = 0

    # Detect trailing blank last-column region for user visibility
    try:
        last_col_series = df.iloc[:, -1]
        isna = pd.to_numeric(last_col_series, errors="coerce").isna()
        # Allow blank only at the end; compute consecutive blanks from end
        trailing_blank = 0
        for val in reversed(isna.tolist()):
            if val:
                trailing_blank += 1
            else:
                break
        info["trailing_blank_last_col_rows"] = int(trailing_blank)
    except Exception:
        info["trailing_blank_last_col_rows"] = None

    return df, info


def detect_roles(df: pd.DataFrame):
    ts_candidates = []
    for col in df.columns:
        coerced = pd.to_datetime(df[col], errors="coerce")
        if coerced.notna().mean() >= 0.9:
            ts_candidates.append(col)
    # If multiple date-like columns exist, take the first; if none, return None
    ts_col = ts_candidates[0] if len(ts_candidates) >= 1 else None
    if df.shape[1] == 1:
        target_col = df.columns[-1]
        feature_cols = []
    else:
        target_col = df.columns[-1]
        feature_cols = [c for c in df.columns if c not in [ts_col, target_col]] if ts_col else [c for c in df.columns if c != target_col]
    return ts_col, target_col, feature_cols


def daily_integrity_status(df: pd.DataFrame, ts_col: str):
    if not ts_col:
        return {"status": "no-ts"}
    s = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if s.empty:
        return {"status": "no-ts"}
    diffs = s.diff().dropna()
    ok = (diffs == pd.Timedelta(days=1)).all()
    if ok:
        return {"status": "daily-ts-valid", "missing_days": 0, "duplicate_days": 0}
    full_range = pd.date_range(s.min(), s.max(), freq="D")
    missing = int(len(set(full_range.date) - set(s.dt.date)))
    dupes = int(s.duplicated().sum())
    if missing == 1 and dupes == 0:
        return {"status": "warn-missing-1"}
    if dupes == 1 and missing == 0:
        return {"status": "warn-duplicate-1"}
    return {"status": "warn-demote"}


 


def _is_numeric_series(series: pd.Series) -> Tuple[bool, Optional[pd.Series]]:
    """Return (is_numeric, numeric_series_or_none). Accepts ints or floats, rejects non-numeric."""
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            return True, numeric
        return False, None
    except Exception:
        return False, None


def _infer_target_type(numeric_series: pd.Series) -> str:
    """Return 'integer' if all values are integers within a small tolerance, else 'float'."""
    # If any fractional part exists beyond tolerance, treat as float
    try:
        tol = 1e-9
        fractional = (numeric_series % 1).abs().gt(tol).any()
        return "float" if fractional else "integer"
    except Exception:
        return "float"


def _strip_whitespace_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str).map(lambda x: x.strip())
        return out
    except Exception:
        return df


def _standardize_missing_tokens_df(df: pd.DataFrame) -> pd.DataFrame:
    tokens = {"", "na", "n/a", "nan", "null", "none", "-", "?", "—"}
    out = df.copy()
    try:
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].astype(str)
                out[col] = out[col].map(lambda x: pd.NA if x.strip().lower() in tokens else x)
        return out
    except Exception:
        return df


def _normalize_dates_to_day(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce")
        # Normalize to midnight to ensure day-wise diffs
        return parsed.dt.normalize()
    except Exception:
        return pd.to_datetime([pd.NA] * len(series), errors="coerce")


def build_checklist_grouped(df_any: pd.DataFrame, file_info: dict, series: pd.DataFrame, load_meta: dict, results: List[dict], future_horizon: int):
    """
    Strict validation checklist for CSV daily time series with two auto modes:
      - Mode A (Extend 20%): All target values are known (no trailing missing). Predict additional 20% using internal features only.
      - Mode B (Predict missing): Target is missing only at the end; all non-target features (if present) must be fully provided for those future rows.

    Checklist items cover:
      - CSV readability, column/row counts
      - First column date validity; last column numeric target
      - No missing anywhere in known region (before last observed target)
      - Dates strictly ascending with daily continuity
      - Future rows detection and feature completeness
    """
    groups = {
        "Open & analyze": [],
        "Features & prep": [],
        "Model & predict": [],
    }

    def add(group: str, text: str, status: str):
        groups[group].append((status, text))

    # Overview
    ftype = file_info.get("file_type")
    n, c = file_info.get("n_rows"), file_info.get("n_cols")
    sep = file_info.get("separator")
    header = file_info.get("header_detected")
    if header is None:
        header_note = "headers (unknown)"
    elif bool(header):
        header_note = "headers detected"
    else:
        header_note = "no headers"
    add("Open & analyze", f"File info: type={ftype}; rows={n}, columns={c}; sep={sep}; {header_note}", "ok" if not file_info.get("error") else "error")

    # If file not accepted as CSV, reject early
    if file_info.get("error"):
        add("Open & analyze", f"Readable as CSV: {file_info.get('error')}", "error")
        ordered = [
            ("Open & analyze", groups["Open & analyze"]),
            ("Features & prep", groups["Features & prep"]),
            ("Model & predict", groups["Model & predict"]),
        ]
        return ordered

    # 1) Check column count
    if c is None or c < 2:
        add("Open & analyze", "Column count: must be at least 2 (date + target)", "error")
    else:
        add("Open & analyze", f"Column count: {c} (>=2)", "ok")

    # 2) Check row count (>=10 data rows)
    if n is None or n < 10:
        add("Open & analyze", "Row count: must be at least 10 data rows", "error")
    else:
        add("Open & analyze", f"Row count: {n} (>=10)", "ok")

    # Normalize data for robust checks (trim whitespace, standardize missing tokens)
    df_norm = _strip_whitespace_df(df_any) if not df_any.empty else df_any
    df_norm = _standardize_missing_tokens_df(df_norm) if not df_norm.empty else df_norm

    # Prepare series for checks
    # First column must be parseable as datetime for every row (normalized to day)
    date_ok = False
    parsed_dates: Optional[pd.Series] = None
    if not df_norm.empty and df_norm.shape[1] >= 1:
        try:
            parsed = _normalize_dates_to_day(df_norm.iloc[:, 0])
            if parsed.notna().all():
                date_ok = True
                parsed_dates = parsed
        except Exception:
            date_ok = False

    add("Open & analyze", "First column is valid date", "ok" if date_ok else "error")

    # Last column numeric
    target_ok = False
    target_type = None
    numeric_series = None
    last_observed_idx = None
    if not df_norm.empty and df_norm.shape[1] >= 2:
        try:
            last_col = df_norm.iloc[:, -1]
            nums_full = pd.to_numeric(last_col, errors="coerce")
            numeric_series = nums_full
            non_missing = nums_full.notna()
            if non_missing.any():
                last_observed_idx = int(non_missing[non_missing].index.max())
                historical_values = nums_full.loc[:last_observed_idx]
                if len(historical_values) > 0 and historical_values.notna().all():
                    target_ok = True
                    target_type = _infer_target_type(historical_values)
            else:
                target_ok = False
        except Exception:
            target_ok = False

    add("Open & analyze", f"Last column is numeric" + (f" (target type: {target_type})" if target_type else ""), "ok" if target_ok else "error")

    # Missing values handling: allow target to be missing only at the end
    missing_ok = False
    known_region_ok = False
    trailing_missing_count = 0
    try:
        isna_df = df_norm.isna()
        if last_observed_idx is not None:
            # Known region is up to last_observed_idx, inclusive
            known_region_ok = not isna_df.loc[:last_observed_idx, :].any().any()
            # Allow missing target after last observed target
            if last_observed_idx < len(df_norm) - 1:
                isna_df.loc[last_observed_idx + 1 :, df_norm.columns[-1]] = False
                trailing_missing_count = int(len(df_norm) - (last_observed_idx + 1))
        missing_ok = not isna_df.any().any()
    except Exception:
        missing_ok = False
        known_region_ok = False
    add("Open & analyze", "No missing/blank values in any column (known region)", "ok" if known_region_ok else "error")
    add("Open & analyze", f"Trailing missing targets at end: {trailing_missing_count}", "ok" if target_ok else "warn")

    # Row count consistency: if any row had fewer fields, pandas would have NaN; covered by missing check
    add("Open & analyze", "Row count consistency across columns", "ok" if missing_ok else "error")

    # Date ordering, duplicates, continuity
    if date_ok and parsed_dates is not None:
        is_strictly_ascending = parsed_dates.is_monotonic_increasing and parsed_dates.is_unique
        add("Open & analyze", "Dates are strictly ascending", "ok" if is_strictly_ascending else "error")

        no_duplicates = parsed_dates.is_unique
        add("Open & analyze", "No duplicate dates", "ok" if no_duplicates else "error")

        diffs = parsed_dates.diff().dropna()
        daily_continuous = bool((diffs == pd.Timedelta(days=1)).all())
        add("Open & analyze", "Daily continuity (each day present)", "ok" if daily_continuous else "error")
    else:
        add("Open & analyze", "Dates are strictly ascending", "error")
        add("Open & analyze", "No duplicate dates", "error")
        add("Open & analyze", "Daily continuity (each day present)", "error")

    # Optional guidance checks (non-blocking warnings)
    #  - Header name sanity for the first column
    if file_info.get("header_detected") and isinstance(file_info.get("header_names"), list) and len(file_info.get("header_names")) >= 1:
        first_name = str(file_info["header_names"][0]).lower()
        looks_like_date = any(tok in first_name for tok in ["date", "day", "time", "timestamp"])
        add("Open & analyze", "Header: first column name looks like a date", "ok" if looks_like_date else "warn")
        if int(file_info.get("header_renamed_count", 0)) > 0:
            add("Open & analyze", f"Header: {file_info['header_renamed_count']} duplicate/empty names adjusted", "warn")

    #  - Target constant / near-constant
    if numeric_series is not None:
        nunique = int(pd.Series(numeric_series).nunique(dropna=True))
        add("Open & analyze", f"Target variability: unique values={nunique}", "warn" if nunique <= 1 else "ok")




    # Optional intermediate feature columns and future feature completeness
    feature_cols = []
    if c and c > 2:
        feature_cols = list(df_norm.columns[1:-1])
        add("Features & prep", f"Intermediate feature columns accepted: {len(feature_cols)}", "ok" if missing_ok else "error")
    else:
        add("Features & prep", "No intermediate feature columns", "ok")

    # Detect number of future rows (after last observed target) and validate feature completeness if any
    num_future_rows = 0
    num_future_rows_filled = 0
    future_features_ok = True
    if last_observed_idx is not None and last_observed_idx < len(df_norm) - 1:
        num_future_rows = int(len(df_norm) - (last_observed_idx + 1))
        add("Features & prep", f"Future rows provided: {num_future_rows}", "ok")
        if len(feature_cols) > 0:
            try:
                future_block = df_norm.loc[last_observed_idx + 1 :, feature_cols]
                row_complete = (~future_block.isna()).all(axis=1)
                num_future_rows_filled = int(row_complete.sum())
                future_features_ok = bool(row_complete.all())
            except Exception:
                future_features_ok = False
            add(
                "Features & prep",
                f"Future features filled rows: {num_future_rows_filled}/{num_future_rows}",
                "ok" if future_features_ok else "error",
            )
    else:
        add("Features & prep", "Future rows provided: 0 (required)", "error")

    # Model & predict mode selection summary
    mode = "Predict missing targets (required)"
    planned_steps = num_future_rows
    # If exogenous features exist and are incomplete for future rows, flag here too
    status = "ok"
    if len(feature_cols) > 0 and not future_features_ok:
        status = "error"
    if planned_steps <= 0:
        status = "error"
    add("Model & predict", f"Mode: {mode}; steps={planned_steps}", status)

    # Final summary
    final_error = any(status == "error" for _, items in groups.items() for status, _ in items)
    summary = f"Summary: headers={'yes' if header else 'no'}, rows={n}, columns={c}, target_type={target_type or 'n/a'}"
    add("Model & predict", summary, "error" if final_error else "ok")

    ordered = [
        ("Open & analyze", groups["Open & analyze"]),
        ("Features & prep", groups["Features & prep"]),
        ("Model & predict", groups["Model & predict"]),
    ]
    return ordered


