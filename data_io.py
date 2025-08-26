from typing import List, Tuple, Optional

import math
import os
import csv
from pathlib import Path
import io
from datetime import datetime, date

import pandas as pd
from features import analyze_preprocessing
from data_utils import (
    SUPPORTED_DATE_FORMATS,
    detect_datetime_format,
    _strip_whitespace_df,
    _standardize_missing_tokens_df,
    _normalize_dates_to_day,
)


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

# Removed local duplicate: use utils.data_utils.detect_datetime_format

# Removed local duplicate: use utils.data_utils._strip_whitespace_df

# Removed local duplicate: use utils.data_utils._standardize_missing_tokens_df

# Removed local duplicate: use utils.data_utils._normalize_dates_to_day

def _infer_target_type(numeric_series: pd.Series) -> str:
    """Return 'integer' if all values are integers within a small tolerance, else 'float'."""
    # If any fractional part exists beyond tolerance, treat as float
    try:
        tol = 1e-9
        fractional = (numeric_series % 1).abs().gt(tol).any()
        return "float" if fractional else "integer"
    except Exception:
        return "float"

class DataLoader:
    def __init__(self, file_obj_or_path):
        self.file_obj_or_path = file_obj_or_path
        self.info = {
            "file_type": None, "size_mb": None, "separator": None,
            "header_detected": None, "n_rows": None, "n_cols": None,
            "error": None, "header_renamed_count": 0, "header_names": None,
            "detected_date_format": None,
        }
        self.raw_bytes = None
        self.buf_for_sniff = None

    def execute(self):
        try:
            print("--- Starting Data Loading Process ---")
            self._get_file_metadata()
            self._validate_file()
            self._read_and_normalize_content()
            self._detect_delimiter()
            df = self._read_csv()
            df = self._detect_and_apply_header(df)
            self._finalize_df_and_info(df)
            print("--- Data Loading Process Finished Successfully ---")
            return self.df, self.info
        except ValueError as e:
            self.info["error"] = str(e)
            print(f"❌ ERROR: {e}")
            return pd.DataFrame(), self.info

    def _get_file_metadata(self):
        print("➡️ Reading file metadata...")
        name, size_bytes = None, None
        is_file_like = hasattr(self.file_obj_or_path, "read") or hasattr(self.file_obj_or_path, "getvalue")
        if is_file_like:
            name = getattr(self.file_obj_or_path, "name", None)
            size_bytes = getattr(self.file_obj_or_path, "size", None)
            try:
                if hasattr(self.file_obj_or_path, "getvalue"):
                    self.buf_for_sniff = self.file_obj_or_path.getvalue()[:4096]
                    self.raw_bytes = self.file_obj_or_path.getvalue()
                else:
                    self.buf_for_sniff = self.file_obj_or_path.read(4096)
                    self.file_obj_or_path.seek(0)
                    self.raw_bytes = self.file_obj_or_path.read()
                self.file_obj_or_path.seek(0)
            except Exception:
                self.buf_for_sniff, self.raw_bytes = b"", None
        else:
            path_str = str(self.file_obj_or_path)
            name = path_str
            try:
                size_bytes = os.path.getsize(path_str)
                with open(path_str, "rb") as f:
                    self.buf_for_sniff = f.read(4096)
                with open(path_str, "rb") as f:
                    self.raw_bytes = f.read()
            except Exception:
                size_bytes, self.buf_for_sniff, self.raw_bytes = None, b"", None
        
        self.ext = (Path(name).suffix or "").lower() if name else ""
        self.info["file_type"] = (self.ext.replace(".", "") or "unknown")
        self.info["size_mb"] = (size_bytes / (1024 * 1024)) if size_bytes is not None else None
        print("✅ File metadata read.")

    def _validate_file(self):
        print("➡️ Validating file type and size...")
        if self.info["size_mb"] is not None and self.info["size_mb"] > 10.0:
            raise ValueError("File larger than 10 MB")
        if self.ext not in [".csv", "csv", ""]:
            raise ValueError("Only CSV files are accepted")
        print("✅ File type and size are valid.")

    def _read_and_normalize_content(self):
        print("➡️ Normalizing CSV content...")
        if self.raw_bytes is not None:
            normalized_bytes, sep_suggested, norm_stats = _normalize_trailing_separators(
                self.raw_bytes, self.info["separator"]
            )
            self.read_target = io.BytesIO(normalized_bytes)
            self.info["trailing_separator_lines_fixed"] = norm_stats.get("lines_fixed", 0)
            print(f"✅ Content normalized. Lines fixed: {self.info['trailing_separator_lines_fixed']}")
            # If we don't already have a separator, adopt the one suggested during normalization
            try:
                if not self.info.get("separator") and sep_suggested in {",", ";", "\t", "|"}:
                    self.info["separator"] = sep_suggested
            except Exception:
                pass
        else:
            self.read_target = self.file_obj_or_path
            print("✅ Content will be read directly (no in-memory normalization).")

    def _detect_delimiter(self):
        print("➡️ Detecting CSV delimiter...")
        detected_sep = None
        # Prefer sampling from the normalized buffer if available
        try:
            if hasattr(self, "read_target") and hasattr(self.read_target, "getvalue"):
                sample_bytes = self.read_target.getvalue()[:4096]
            else:
                sample_bytes = (self.buf_for_sniff or b"")
            sample_text = (sample_bytes or b"").decode(errors="ignore")
        except Exception:
            sample_text = (self.buf_for_sniff or b"").decode(errors="ignore")

        try:
            if sample_text:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text, delimiters=[",", ";", "\t", "|"])
                detected_sep = dialect.delimiter
        except Exception:
            detected_sep = None

        # Heuristic override: if header clearly contains a candidate, prefer it
        try:
            first_line = sample_text.splitlines()[0] if sample_text else ""
            if ";" in first_line and detected_sep not in {";"}:
                detected_sep = ";"
            elif "," in first_line and not detected_sep:
                detected_sep = ","
            elif not detected_sep:
                for cand in [",", ";", "\t", "|"]:
                    if cand in first_line:
                        detected_sep = cand
                        break
        except Exception:
            pass

        # If normalization suggested a separator earlier, prefer it when detection failed
        try:
            if not detected_sep and self.info.get("separator") in {",", ";", "\t", "|"}:
                detected_sep = self.info["separator"]
        except Exception:
            pass

        self.info["separator"] = detected_sep
        print(f"✅ Delimiter detected: '{detected_sep}'")

    def _read_csv(self):
        print("➡️ Reading CSV data into DataFrame...")
        try:
            engine = "python"
            raw_df = pd.read_csv(
                self.read_target, 
                sep=self.info["separator"], 
                engine=engine, 
                header=None, 
                dtype=str
            )
            print("✅ CSV data read successfully.")
            return raw_df
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

    def _detect_and_apply_header(self, raw_df):
        print("➡️ Detecting header row (daily sequence check)...")
        header_detected = False
        df = raw_df.copy()

        def _detect_daily_format_on_values(values):
            # Check rows 2..end (index 1..end) for strict daily increments
            tail_vals = [str(v).strip() for v in values[1:]]
            if len(tail_vals) < 1:
                return None
            for fmt in SUPPORTED_DATE_FORMATS:
                try:
                    parsed_tail = pd.to_datetime(tail_vals, format=fmt, errors="coerce").dt.normalize()
                except Exception:
                    continue
                if parsed_tail.isna().any():
                    continue
                diffs = parsed_tail.diff().dropna()
                if not diffs.empty and (diffs == pd.Timedelta(days=1)).all():
                    return fmt
            return None

        try:
            first_col_vals = df.iloc[:, 0].astype(str).tolist()
            fmt = _detect_daily_format_on_values(first_col_vals)
            self.info["detected_date_format"] = fmt

            if fmt is None:
                print("⚠️ Could not confirm daily date format from rows 2..end.")
                header_detected = False
            else:
                # With a confirmed daily format on rows 2..end, decide if row 1 is a header
                if len(first_col_vals) >= 2:
                    second_dt = pd.to_datetime(first_col_vals[1], format=fmt, errors="coerce")
                    first_dt = pd.to_datetime(first_col_vals[0], format=fmt, errors="coerce")
                    if pd.isna(first_dt) or pd.isna(second_dt) or (first_dt.normalize() != (second_dt.normalize() - pd.Timedelta(days=1))):
                        header_detected = True
                    else:
                        header_detected = False
                else:
                    header_detected = False
        except Exception:
            header_detected = False

        self.info["header_detected"] = header_detected
        if header_detected:
            print("✅ Header row detected.")
            header_row = df.iloc[0].tolist()
            df = df.iloc[1:].copy().reset_index(drop=True)
            self._apply_header_names(df, header_row)
        else:
            # No header present: assign semantic names: date, feature_*, target
            print("✅ No header row detected. Assigning default names: date, feature_*, target.")
            ncols = int(df.shape[1])
            names = []
            if ncols >= 1:
                names.append("date")
            if ncols >= 3:
                for i in range(1, ncols - 1):
                    names.append(f"feature_{i}")
            if ncols >= 2:
                names.append("target")
            if len(names) != ncols:
                names = [f"col_{i+1}"] * ncols
            df.columns = names
            self.info["header_names"] = df.columns.tolist()
        return df

    def _apply_header_names(self, df, header_row):
        print("➡️ Applying header names...")
        try:
            dedup, seen = [], set()
            for idx, val in enumerate(header_row):
                name = str(val).strip() if (val is not None and str(val).strip() != "") else f"col_{idx+1}"
                if name in seen:
                    k=2
                    new_name = f"{name}_{k}"
                    while new_name in seen:
                        k+=1
                        new_name = f"{name}_{k}"
                    name = new_name
                seen.add(name)
                dedup.append(name)
            df.columns = dedup
            original = [str(v).strip() if v is not None else "" for v in header_row]
            self.info["header_renamed_count"] = sum(1 for a, b in zip(original, dedup) if a != b)
            self.info["header_names"] = dedup
            print("✅ Header names applied.")
        except Exception:
            df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
            self.info["header_names"] = df.columns.tolist()
            print("⚠️ Could not apply header names, using generic ones.")

    def _finalize_df_and_info(self, df):
        print("➡️ Finalizing DataFrame and metadata...")
        self.df = df
        self.info["n_rows"], self.info["n_cols"] = int(df.shape[0]), int(df.shape[1])
        try:
            last_col_series = df.iloc[:, -1]
            isna = pd.to_numeric(last_col_series, errors="coerce").isna()
            trailing_blank = 0
            for val in reversed(isna.tolist()):
                if val: trailing_blank += 1
                else: break
            self.info["trailing_blank_last_col_rows"] = int(trailing_blank)
        except Exception:
            self.info["trailing_blank_last_col_rows"] = None
        print("✅ DataFrame and metadata finalized.")

def load_data_with_checklist(file_obj_or_path):
    return DataLoader(file_obj_or_path).execute()

class DataValidator:
    def __init__(self, df, file_info):
        self.df_orig = df
        self.file_info = file_info
        self.checklist = []

    def execute(self):
        try:
            print("\n--- Starting Data Validation Process ---")
            if self.file_info.get("error"):
                raise ValueError(f"Cannot validate due to loading error: {self.file_info.get('error')}")

            self._check_dimensions()
            self._normalize_data()
            self._validate_columns()
            self._validate_missing_values()
            self._validate_date_sequence()
            self._perform_guidance_checks()
            self._analyze_features_and_seasonality()
            print("--- Data Validation Process Finished Successfully ---")
            return self.checklist, True
        except ValueError as e:
            print(f"❌ ERROR: {e}")
            self.checklist.append(("error", str(e)))
            return self.checklist, False

    def _add_check(self, text, status="ok"):
        if status == "error":
            raise ValueError(text)
        elif status == "warn":
            print(f"⚠️ {text}")
        else:
            print(f"✅ {text}")
        self.checklist.append((status, text))

    def _check_dimensions(self):
        print("\n➡️ Checking data dimensions...")
        n, c = self.file_info.get("n_rows"), self.file_info.get("n_cols")
        if c is None or c < 2:
            self._add_check("Column count: must be at least 2 (date + target)", "error")
        self._add_check(f"Column count: {c} (>=2)")
        if n is None or n < 10:
            self._add_check("Row count: must be at least 10 data rows", "error")
        self._add_check(f"Row count: {n} (>=10)")

    def _normalize_data(self):
        print("\n➡️ Normalizing data for validation...")
        self.df_norm = _strip_whitespace_df(self.df_orig)
        self.df_norm = _standardize_missing_tokens_df(self.df_norm)
        print("✅ Data normalized (whitespace, missing tokens).")

    def _validate_columns(self):
        print("\n➡️ Validating date and target columns...")
        # Date column
        self.parsed_dates = None
        if not self.df_norm.empty:
            try:
                col_vals_all = self.df_norm.iloc[:, 0].astype(str).tolist()
                detected_fmt = self.file_info.get("detected_date_format")
                # Fallback: if no format saved yet, detect on entire column now
                if not detected_fmt:
                    detected_now = detect_datetime_format(col_vals_all)
                    if detected_now:
                        self.file_info["detected_date_format"] = detected_now
                        detected_fmt = detected_now
                parsed = _normalize_dates_to_day(self.df_norm.iloc[:, 0], date_format=detected_fmt)
                ok_ratio = parsed.notna().mean()
                if ok_ratio >= 0.95:
                    self.parsed_dates = parsed
                else:
                    # Try mixed parsing without a fixed format
                    parsed_mixed = _normalize_dates_to_day(self.df_norm.iloc[:, 0], date_format=None)
                    if parsed_mixed.notna().mean() >= 0.95:
                        self.parsed_dates = parsed_mixed
                    else:
                        # Try re-detecting on non-empty unique values only
                        non_empty_vals = [v for v in col_vals_all if str(v).strip()]
                        detected_retry = detect_datetime_format(non_empty_vals)
                        if detected_retry:
                            self.file_info["detected_date_format"] = detected_retry
                            parsed_retry = _normalize_dates_to_day(self.df_norm.iloc[:, 0], date_format=detected_retry)
                            if parsed_retry.notna().mean() >= 0.95:
                                self.parsed_dates = parsed_retry
            except Exception: pass
        if self.parsed_dates is None:
            self._add_check("First column must be valid dates", "error")
        self._add_check("First column is valid date")

        # Target column (robust numeric check)
        self.numeric_series, self.last_observed_idx, target_type = None, None, None
        if not self.df_norm.empty and self.df_norm.shape[1] >= 2:
            try:
                last_col = self.df_norm.iloc[:, -1]
                nums_full = pd.to_numeric(last_col, errors="coerce")
                self.numeric_series = nums_full
                non_missing_mask = nums_full.notna()
                valid_ratio_overall = float(non_missing_mask.mean()) if len(nums_full) > 0 else 0.0
                if non_missing_mask.any():
                    self.last_observed_idx = int(non_missing_mask[non_missing_mask].index.max())
                    historical_values = nums_full.loc[:self.last_observed_idx]
                    if len(historical_values) > 0:
                        hist_valid_ratio = float(historical_values.notna().mean())
                        # Accept if the known region is mostly numeric
                        if hist_valid_ratio >= 0.95:
                            target_type = _infer_target_type(historical_values.dropna())
                # As a fallback, accept if overall numeric ratio is high enough
                if target_type is None and valid_ratio_overall >= 0.95:
                    target_type = _infer_target_type(nums_full.dropna())
            except Exception:
                target_type = None

        if target_type is None:
            self._add_check("Last column must be numeric", "error")
        self._add_check(f"Last column is numeric (target type: {target_type})")

    def _validate_missing_values(self):
        print("\n➡️ Checking for missing values...")
        try:
            isna_df = self.df_norm.isna()
            known_region_ok = False
            if self.last_observed_idx is not None:
                known_region_ok = not isna_df.loc[:self.last_observed_idx, :].any().any()
            if not known_region_ok:
                self._add_check("No missing/blank values in any column (known region)", "error")
            self._add_check("No missing/blank values in known region.")
        except Exception:
            self._add_check("Missing value check failed", "error")

    def _validate_date_sequence(self):
        print("\n➡️ Validating date sequence...")
        if self.parsed_dates is not None:
            # Use only valid dates for ordering/continuity checks
            valid_dates = self.parsed_dates.dropna()

            # Check for duplicates (consider only valid dates)
            if not valid_dates.is_unique:
                duplicates = valid_dates[valid_dates.duplicated()].dt.strftime('%Y-%m-%d').tolist()
                self._add_check(f"Dates must be unique. Found duplicates for: {', '.join(duplicates)}", "error")
            self._add_check("Dates are unique.")

            # Check for ascending order
            if not valid_dates.is_monotonic_increasing:
                # Find the first negative diff among valid dates, if any
                diffs = valid_dates.diff()
                neg = diffs[diffs < pd.Timedelta(days=0)]
                if not neg.empty:
                    violation_idx = neg.index[0]
                    prev_idx = valid_dates.index.get_loc(violation_idx) - 1
                    if prev_idx >= 0:
                        # Map to actual dates for message
                        prev_date = valid_dates.iloc[prev_idx].strftime('%Y-%m-%d')
                        curr_date = valid_dates.loc[violation_idx].strftime('%Y-%m-%d')
                        self._add_check(f"Dates must be in ascending order. Violation at index {violation_idx}: '{curr_date}' follows '{prev_date}'", "error")
                    else:
                        self._add_check("Dates must be in ascending order.", "error")
                else:
                    # Non-monotonic but no negative diffs likely due to missing/NaT; flag ordering error generically
                    self._add_check("Dates must be in ascending order (ignoring blanks)", "error")
            self._add_check("Dates are in ascending order.")
            
            diffs = valid_dates.diff().dropna()
            if not diffs.empty and not (diffs == pd.Timedelta(days=1)).all():
                self._add_check("Daily continuity (each day present)", "error")
            self._add_check("Daily continuity is valid.")
        else:
            self._add_check("Date sequence validation skipped (invalid date column)", "error")

    def _perform_guidance_checks(self):
        print("\n➡️ Performing optional guidance checks...")
        if self.file_info.get("header_detected") and isinstance(self.file_info.get("header_names"), list):
            first_name = str(self.file_info["header_names"][0]).lower()
            looks_like_date = any(tok in first_name for tok in ["date", "day", "time"])
            self._add_check("Header: first column name looks like a date", "ok" if looks_like_date else "warn")
        if self.numeric_series is not None:
            nunique = int(pd.Series(self.numeric_series).nunique(dropna=True))
            self._add_check(f"Target variability: unique values={nunique}", "ok" if nunique > 1 else "warn")

    def _analyze_features_and_seasonality(self):
        print("\n➡️ Analyzing features and seasonality...")
        c = self.file_info.get("n_cols")
        feature_cols = []
        if c and c > 2:
            feature_cols = list(self.df_norm.columns[1:-1])
            self._add_check(f"Intermediate feature columns accepted: {len(feature_cols)}")
        else:
            self._add_check("No intermediate feature columns.")

        num_future_rows = 0
        if self.last_observed_idx is not None and self.last_observed_idx < len(self.df_norm) - 1:
            num_future_rows = int(len(self.df_norm) - (self.last_observed_idx + 1))
        
        if num_future_rows == 0:
            # Do not hard-stop here; the main app will show a targeted message.
            self._add_check("Future rows provided: 0 (add future dates with blank target to predict)", "warn")
        self._add_check(f"Future rows provided: {num_future_rows}")

        try:
            series_for_analysis = self.df_orig.rename(columns={self.df_orig.columns[0]: 'ds', self.df_orig.columns[-1]: 'y'})
            detected_fmt = self.file_info.get("detected_date_format")
            series_for_analysis['ds'] = _normalize_dates_to_day(series_for_analysis['ds'], date_format=detected_fmt)
            series_for_analysis = series_for_analysis.dropna(subset=['ds'])
            # Ensure numeric target and restrict to known values for analysis
            series_for_analysis['y'] = pd.to_numeric(series_for_analysis['y'], errors='coerce')
            series_for_analysis = series_for_analysis.dropna(subset=['y'])
            if series_for_analysis.empty:
                self._add_check("No strong seasonal periods detected", "warn")
                return
            info = analyze_preprocessing(series_for_analysis)
            detected_periods = [int(p) for p in info.get("fourier_periods_detected", []) if int(p) > 1]
            if detected_periods:
                self._add_check(f"Detected seasonal periods: {detected_periods}")
            else:
                self._add_check("No strong seasonal periods detected", "warn")
        except Exception as e:
            self._add_check(f"Seasonality analysis failed: {e}", "warn")

def validate_data_with_checklist(df, file_info):
    return DataValidator(df, file_info).execute()


def detect_roles(df: pd.DataFrame):
    ts_candidates = []
    for col in df.columns:
        try:
            coerced = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except Exception:
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
    try:
        s = pd.to_datetime(df[ts_col], errors="coerce", format="mixed").dropna().sort_values().reset_index(drop=True)
    except Exception:
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


 


"""
Removed duplicate helper functions that shadowed imports from utils.data_utils
(_is_numeric_series, _infer_target_type, _strip_whitespace_df, _standardize_missing_tokens_df,
_normalize_dates_to_day).
"""


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
        "Seasonality": [],
        # Keep for backward-compatibility with callers that expect this key
        "Model & predict": [],
    }

    def add(group: str, text: str, status: str):
        # Tolerate unknown groups by creating them on the fly
        if group not in groups:
            groups[group] = []
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
            ("Open & analyze", groups.get("Open & analyze", [])),
            ("Features & prep", groups.get("Features & prep", [])),
            ("Seasonality", groups.get("Seasonality", [])),
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
            detected_fmt = file_info.get("detected_date_format")
            parsed = _normalize_dates_to_day(df_norm.iloc[:, 0], date_format=detected_fmt)
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

    else:
        add("Features & prep", "Future rows provided: 0 (required)", "error")

    # Seasonality insight (non-blocking) delegated to features.analyze_preprocessing
    fixed_periods: List[int] = []
    detected_periods: List[int] = []
    fourier_harmonics = None
    try:
        if isinstance(series, pd.DataFrame) and not series.empty:
            info = analyze_preprocessing(series)
            fixed_periods = [int(p) for p in info.get("fourier_periods_fixed", []) if int(p) > 1]
            detected_periods = [int(p) for p in info.get("fourier_periods_detected", []) if int(p) > 1]
            fourier_harmonics = info.get("fourier_harmonics", None)
    except Exception:
        fixed_periods = []
        detected_periods = []
        fourier_harmonics = None

    if fixed_periods:
        if fourier_harmonics is not None:
            groups["Seasonality"].append(("ok", f"Fixed Fourier periods configured: {fixed_periods}; harmonics={fourier_harmonics}"))
        else:
            groups["Seasonality"].append(("ok", f"Fixed Fourier periods configured: {fixed_periods}"))
    if detected_periods:
        groups["Seasonality"].append(("ok", f"Detected periods: {detected_periods}"))
    else:
        groups["Seasonality"].append(("warn", "No strong seasonal periods detected"))

    # Final summary
    final_error = any(status == "error" for _, items in groups.items() for status, _ in items)
    summary = f"Summary: headers={'yes' if header else 'no'}, rows={n}, columns={c}, target_type={target_type or 'n/a'}"
    add("Seasonality", summary, "error" if final_error else "ok")

    ordered = [
        ("Open & analyze", groups.get("Open & analyze", [])),
        ("Features & prep", groups.get("Features & prep", [])),
        ("Seasonality", groups.get("Seasonality", [])),
    ]
    return ordered


