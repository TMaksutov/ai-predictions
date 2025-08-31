# === Data loading with dynamic validation checklist (Kaggle-style) ===

import os
import pandas as pd
from pathlib import Path
import csv

def load_data_with_checklist(file_obj_or_path, progress_callback=None):
    checklist = []
    info = {"error": None}
    
    def add_check(status, message):
        """Helper to add check result and trigger progress callback"""
        checklist.append((status, message))
        if progress_callback:
            try:
                progress_callback(checklist.copy())
            except Exception:
                pass
    
    def format_file_size(size_bytes):
        """Format file size in appropriate units"""
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_bytes / 1024:.1f} KB" if size_mb < 1 else f"{size_mb:.1f} MB"
    
    def _safe_seek_start(file_like):
        """Attempt to rewind a file-like object to the start."""
        try:
            if hasattr(file_like, 'seek'):
                file_like.seek(0)
        except Exception:
            pass

    def _detect_encoding_and_sample(file_obj_or_path, sample_bytes: int = 4096):
        """Detect a reasonable text encoding and return (encoding, sample_text).
        Tries utf-8-sig, utf-8, then latin-1.
        """
        encodings_to_try = ["utf-8-sig", "utf-8", "latin-1"]
        sample_text = ""
        chosen_encoding = None

        try:
            if hasattr(file_obj_or_path, 'read'):
                # Read bytes from current position, then restore
                current_pos = file_obj_or_path.tell()
                try:
                    sample_bytes_data = file_obj_or_path.read(sample_bytes)
                finally:
                    try:
                        file_obj_or_path.seek(current_pos)
                    except Exception:
                        pass

                if isinstance(sample_bytes_data, str):
                    # Already text; assume utf-8 as encoding marker
                    sample_text = sample_bytes_data
                    chosen_encoding = "utf-8"
                else:
                    for enc in encodings_to_try:
                        try:
                            sample_text = (sample_bytes_data or b"").decode(enc)
                            chosen_encoding = enc
                            break
                        except Exception:
                            continue
            else:
                # Path-like: read bytes
                with open(file_obj_or_path, 'rb') as fb:
                    sample_bytes_data = fb.read(sample_bytes)
                for enc in encodings_to_try:
                    try:
                        sample_text = (sample_bytes_data or b"").decode(enc)
                        chosen_encoding = enc
                        break
                    except Exception:
                        continue
        except Exception:
            # Fallbacks
            chosen_encoding = "utf-8"
            sample_text = ""

        if not chosen_encoding:
            chosen_encoding = "utf-8"
        return chosen_encoding, sample_text

    def _infer_delimiter(sample_text: str) -> str:
        """Infer delimiter using csv.Sniffer with fallback to frequency counts."""
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=[',', ';', '\t', '|'])
            return dialect.delimiter
        except Exception:
            # Fallback to frequency counts
            delim_counts = {
                ',': sample_text.count(','),
                ';': sample_text.count(';'),
                '\t': sample_text.count('\t'),
                '|': sample_text.count('|')
            }
            max_count = max(delim_counts.values()) if sample_text else 0
            return max(delim_counts.items(), key=lambda x: x[1])[0] if max_count > 0 else ','

    def check_file_size(size_bytes):
        """Check file size and return error if too large"""
        size_mb = size_bytes / (1024 * 1024)
        if size_mb >= 1:
            error_msg = f"File too large: {size_mb:.1f} MB (max 1MB)"
            add_check("error", f"File too large: {format_file_size(size_bytes)} (max 1MB)")
            return error_msg
        add_check("ok", f"File size: {format_file_size(size_bytes)}")
        return None
    
    def early_return_error(error_msg):
        """Helper for early error returns"""
        info["error"] = error_msg
        info["checklist"] = checklist
        return pd.DataFrame(), info
    
    try:
        # Extract file info
        is_uploaded_file = hasattr(file_obj_or_path, "name") and hasattr(file_obj_or_path, "read")
        file_path = getattr(file_obj_or_path, "name", None) or (str(file_obj_or_path) if isinstance(file_obj_or_path, (str, Path)) else None)
        
        # Pre-loading validation
        if file_path:
            if is_uploaded_file:
                add_check("ok", f"File uploaded: {Path(file_path).name}")
                try:
                    size_bytes = getattr(file_obj_or_path, "size", None)
                    if size_bytes is None:
                        current_pos = file_obj_or_path.tell()
                        file_obj_or_path.seek(0, 2)
                        size_bytes = file_obj_or_path.tell()
                        file_obj_or_path.seek(current_pos)
                    
                    size_error = check_file_size(size_bytes)
                    if size_error:
                        return early_return_error(size_error)
                except Exception:
                    add_check("warning", "Cannot determine file size")
            else:
                if not os.path.exists(file_path):
                    add_check("error", f"File not found: {file_path}")
                    return early_return_error(f"File not found: {file_path}")
                
                add_check("ok", f"File found: {Path(file_path).name}")
                try:
                    size_error = check_file_size(os.path.getsize(file_path))
                    if size_error:
                        return early_return_error(size_error)
                except Exception:
                    add_check("error", "Cannot read file size")
                    return early_return_error("Cannot access file")
            
            # Check file format
            ext = Path(file_path).suffix.lower()
            if ext == ".csv":
                add_check("ok", "CSV format detected")
            elif ext in [".xlsx", ".xls"]:
                add_check("ok", f"Excel format detected ({ext})")
            else:
                add_check("warning", f"Unexpected format: {ext or 'no extension'}")
        
        # Load data
        try:
            if file_path and Path(file_path).suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_obj_or_path)
            else:
                # Robust encoding + delimiter detection using a small sample
                detected_encoding, sample_text = _detect_encoding_and_sample(file_obj_or_path)
                detected_delimiter = _infer_delimiter(sample_text or "")

                # Ensure file-like pointer is at start before full read
                if hasattr(file_obj_or_path, 'read'):
                    _safe_seek_start(file_obj_or_path)

                # Load CSV with detected delimiter and encoding
                df = pd.read_csv(file_obj_or_path, sep=detected_delimiter, encoding=detected_encoding)
            add_check("ok", "Data loaded successfully")
        except Exception as load_error:
            add_check("error", f"Failed to load file: {str(load_error)}")
            return early_return_error(f"Failed to load file: {str(load_error)}")
        
        # Post-loading validation
        if len(df) == 0:
            add_check("error", "No data rows found")
            return early_return_error("Empty dataset")
        elif len(df) > 10000:
            add_check("error", f"Too many rows: {len(df)} (max 10000)")
            return early_return_error("Dataset exceeds 10,000 rows limit")
        add_check("ok", f"{len(df)} rows loaded")
        
        if len(df.columns) == 0:
            add_check("error", "No columns found")
            return early_return_error("No columns found")
        elif len(df.columns) == 1:
            add_check("error", f"Only 1 column detected. Need at least 2 columns: date (first) and target (last). Check CSV format - ensure a consistent delimiter and proper quoting is used.")
            return early_return_error("Dataset must have at least 2 columns: date column (first) and target column (last)")
        add_check("ok", f"{len(df.columns)} columns detected")
        
        # Check missing values - critical for prediction logic
        total_missing = df.isnull().sum().sum()
        
        if total_missing == 0:
            add_check("ok", "No missing values found - showing model comparison table")
            # Continue - this might be training data only
        else:
            # Check if missing values are only in the last column (target)
            other_cols_missing = df.iloc[:, :-1].isnull().any().any()
            if other_cols_missing:
                add_check("error", "Missing values found in feature columns (only target column may have missing values)")
                return early_return_error("Missing values are only allowed in the target column (last column)")
            
            # Check missing values in target column (last column)
            last_col_missing = df[df.columns[-1]].isnull()
            n_missing_target = int(last_col_missing.sum())
            if not last_col_missing.any():
                add_check("ok", "No missing values in target column - showing model comparison table")
                # Continue - this might be training data only
            else:
                # Validate that missing values are consecutive and at the end
                missing_indices = df[df[df.columns[-1]].isnull()].index.tolist()
                expected_indices = list(range(len(df) - len(missing_indices), len(df)))
                
                if missing_indices != expected_indices:
                    add_check("error", "Missing values in target column must be consecutive and at the end")
                    return early_return_error("Missing values in the target column must be consecutive and at the end")
                
                add_check("ok", f"{len(missing_indices)} values in target column for prediction")
        
        # Add encoding/delimiter info if available
        try:
            info_encoding = locals().get('detected_encoding') if 'detected_encoding' in locals() else None
            info_delimiter = locals().get('detected_delimiter') if 'detected_delimiter' in locals() else None
            if info_encoding:
                add_check("ok", f"Encoding detected: {info_encoding}")
            if info_delimiter:
                shown = {'\t': 'TAB'}.get(info_delimiter, info_delimiter)
                add_check("ok", f"Delimiter detected: {shown}")
        except Exception:
            pass
        
        # Duplicate checking removed per user request
        
        # Check date format - reject month-first numeric formats (MM/DD/YY, MM-DD-YY, MM.DD.YY), only allow day-first formats
        first_col = df.columns[0]
        date_sample = df[first_col].dropna().head(10).astype(str).tolist()
        
        if date_sample:
            # Check for MM/DD/YY patterns that we want to reject
            import re
            rejected_patterns = [
                r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # MM/DD/YY or MM/DD/YYYY
                r'^\d{1,2}-\d{1,2}-\d{2,4}$',  # MM-DD-YY or MM-DD-YYYY
                r'^\d{1,2}\.\d{1,2}\.\d{2,4}$',  # MM.DD.YY or MM.DD.YYYY
            ]
            
            has_rejected_format = False
            for sample_date in date_sample[:5]:  # Check first 5 samples
                for pattern in rejected_patterns:
                    if re.match(pattern, sample_date.strip()):
                        # Additional check: if first number > 12, it's likely day-first (allowed)
                        parts = re.split(r'[./-]', sample_date.strip())
                        if len(parts) >= 2:
                            try:
                                first_num = int(parts[0])
                                if first_num <= 12:  # Could be month-first format (rejected)
                                    has_rejected_format = True
                                    break
                            except ValueError:
                                pass
                if has_rejected_format:
                    break
            
            if has_rejected_format:
                add_check("error", "Date format appears to be month-first (e.g., MM/DD/YYYY, MM-DD-YYYY, or MM.DD.YYYY). Please use day-first formats like DD/MM/YYYY or YYYY-MM-DD")
                return early_return_error("Date format must be day-first (DD/MM/YYYY) or ISO format (YYYY-MM-DD), not month-first numeric formats")
            else:
                add_check("ok", "Date format validation passed")

        # Check consecutive daily dates and detect date format
        detected_date_format = None
        if len(df) > 1:
            try:
                date_col = df.columns[0]
                sample_date = str(df[date_col].iloc[0]).strip()
                if '-' in sample_date and len(sample_date.split('-')[0]) == 4:
                    dates = pd.to_datetime(df[date_col], errors='coerce', format='%Y-%m-%d')
                    detected_date_format = '%Y-%m-%d'
                else:
                    dates = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
                    # Try to determine specific format for day-first dates
                    if '.' in sample_date:
                        detected_date_format = '%d.%m.%Y'
                    elif '/' in sample_date:
                        detected_date_format = '%d/%m/%Y'
                    elif '-' in sample_date:
                        detected_date_format = '%d-%m-%Y'
                
                if dates.isna().any():
                    num_unparsed = int(dates.isna().sum())
                    add_check("warning", f"Some dates could not be parsed for consecutive check (unparsed: {num_unparsed})")
                else:
                    # Check differences between consecutive rows (assume data is already in order)
                    date_diffs = dates.diff()[1:]  # Skip first NaT
                    one_day = pd.Timedelta(days=1)
                    
                    if not (date_diffs == one_day).all():
                        add_check("error", "Dates are not consecutive daily")
                        return early_return_error("Dataset must have consecutive daily dates with no missing days")
                    else:
                        add_check("ok", "All dates are consecutive (1 day apart)")
                        
            except Exception as e:
                add_check("warning", f"Could not validate consecutive dates: {str(e)}")

        
        info["checklist"] = checklist
        info.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "header_names": df.columns.tolist(),
            "detected_date_format": detected_date_format,
            "delimiter_used": locals().get('detected_delimiter') if 'detected_delimiter' in locals() else None,
            "encoding_used": locals().get('detected_encoding') if 'detected_encoding' in locals() else None,
            "num_unparsed_dates": int(dates.isna().sum()) if 'dates' in locals() else None,
            "n_missing_target": int(df[df.columns[-1]].isnull().sum()) if df.shape[1] > 0 else None
        })
        
        return df, info
        
    except Exception as e:
        add_check("error", f"Unexpected error: {str(e)}")
        info["error"] = str(e)
        info["checklist"] = checklist
        return pd.DataFrame(), info




