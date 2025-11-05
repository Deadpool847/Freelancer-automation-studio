import polars as pl
import duckdb
from typing import Dict, Tuple, Optional
from pathlib import Path
import hashlib
from loguru import logger
import dateparser
from rapidfuzz import fuzz
import re

class DataCleaner:
    """ETL pipeline with Polars and DuckDB"""
    
    def __init__(self, remove_duplicates: bool = True, handle_missing: str = "drop", standardize_dates: bool = True):
        self.remove_duplicates = remove_duplicates
        self.handle_missing = handle_missing
        self.standardize_dates = standardize_dates
        
    def clean_and_profile(self, file_path: str) -> Tuple[pl.DataFrame, Dict, float]:
        """Clean data and generate profile"""
        logger.info(f"Starting ETL on {file_path}")
        
        # Load data with Polars
        df = self._load_data(file_path)
        original_rows = len(df)
        original_cols = len(df.columns)
        
        logger.info(f"Loaded {original_rows} rows, {original_cols} columns")
        
        # Cleaning steps
        df = self._remove_duplicates(df) if self.remove_duplicates else df
        df = self._handle_missing_values(df)
        
        # Check if DataFrame is empty after cleaning
        if len(df) == 0:
            logger.error("All rows were dropped during cleaning. Cannot proceed.")
            raise ValueError(
                "ETL Error: All rows were removed during cleaning. "
                "Reasons: (1) All rows had missing values with handle_missing='drop', "
                "or (2) All rows were duplicates. "
                "Solution: Change handle_missing strategy to 'fill_mean', 'fill_median', or 'fill_mode'."
            )
        
        df = self._standardize_columns(df)
        df = self._detect_and_parse_dates(df) if self.standardize_dates else df
        df = self._clean_text_columns(df)
        
        # Profiling
        profile = self._generate_profile(df)
        quality_score = self._calculate_quality_score(df, original_rows, original_cols)
        
        logger.info(f"ETL complete. Quality score: {quality_score:.2%}")
        
        return df, profile, quality_score
    
    def _load_data(self, file_path: str) -> pl.DataFrame:
        """Load data from various formats"""
        path = Path(file_path)
        
        if path.suffix == '.parquet':
            return pl.read_parquet(file_path)
        elif path.suffix == '.csv':
            return pl.read_csv(file_path, infer_schema_length=10000)
        elif path.suffix in ['.xlsx', '.xls']:
            import pandas as pd
            return pl.from_pandas(pd.read_excel(file_path))
        elif path.suffix == '.json':
            return pl.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _remove_duplicates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove duplicate rows"""
        before = len(df)
        df = df.unique()
        after = len(df)
        
        if before > after:
            logger.info(f"Removed {before - after} duplicate rows")
        
        return df
    
    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values based on strategy"""
        if self.handle_missing == "drop":
            before = len(df)
            df = df.drop_nulls()
            logger.info(f"Dropped {before - len(df)} rows with null values")
        
        elif self.handle_missing == "fill_mean":
            numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            for col in numeric_cols:
                mean_val = df[col].mean()
                if mean_val is not None and not pl.col(col).is_null().all():
                    df = df.with_columns(pl.col(col).fill_null(mean_val))
                else:
                    # If all values are null, fill with 0
                    df = df.with_columns(pl.col(col).fill_null(0))
        
        elif self.handle_missing == "fill_median":
            numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            for col in numeric_cols:
                median_val = df[col].median()
                if median_val is not None and not pl.col(col).is_null().all():
                    df = df.with_columns(pl.col(col).fill_null(median_val))
                else:
                    # If all values are null, fill with 0
                    df = df.with_columns(pl.col(col).fill_null(0))
        
        elif self.handle_missing == "fill_mode":
            for col in df.columns:
                mode_result = df[col].mode()
                # Check if mode exists and is not empty
                if len(mode_result) > 0:
                    mode_val = mode_result.first()
                    if mode_val is not None:
                        df = df.with_columns(pl.col(col).fill_null(mode_val))
                    else:
                        # If mode is None, fill with appropriate default
                        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                            df = df.with_columns(pl.col(col).fill_null(0))
                        elif df[col].dtype == pl.Utf8:
                            df = df.with_columns(pl.col(col).fill_null(""))
                else:
                    # Column has all nulls, fill with default
                    if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                        df = df.with_columns(pl.col(col).fill_null(0))
                    elif df[col].dtype == pl.Utf8:
                        df = df.with_columns(pl.col(col).fill_null(""))
        
        return df
    
    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize column names"""
        # Convert to snake_case
        new_names = {}
        for col in df.columns:
            new_name = re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())
            new_name = re.sub(r'_+', '_', new_name).strip('_')
            new_names[col] = new_name
        
        return df.rename(new_names)
    
    def _detect_and_parse_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Detect and parse date columns"""
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                # Sample first non-null value
                sample = df[col].drop_nulls().head(1)
                if len(sample) > 0:
                    sample_val = sample[0]
                    try:
                        parsed = dateparser.parse(str(sample_val))
                        if parsed:
                            # Attempt to parse entire column
                            logger.info(f"Detected date column: {col}")
                            # Use Polars native date parsing
                            try:
                                df = df.with_columns(
                                    pl.col(col).str.to_datetime().alias(f"{col}_parsed")
                                )
                            except:
                                logger.warning(f"Could not parse all dates in {col}")
                    except:
                        pass
        
        return df
    
    def _clean_text_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Clean text columns"""
        for col in df.columns:
            if df[col].dtype == pl.Utf8:
                df = df.with_columns(
                    pl.col(col).str.strip_chars().alias(col)
                )
        
        return df
    
    def _generate_profile(self, df: pl.DataFrame) -> Dict:
        """Generate data profile using DuckDB"""
        con = duckdb.connect(':memory:')
        con.register('df_view', df.to_arrow())
        
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_mb': df.estimated_size('mb'),
            'columns': {}
        }
        
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            null_count = df[col].null_count()
            
            col_profile = {
                'dtype': col_dtype,
                'null_count': null_count,
                'null_percentage': (null_count / len(df)) * 100 if len(df) > 0 else 0
            }
            
            # Numeric stats
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                col_profile.update({
                    'mean': float(df[col].mean()) if df[col].mean() else None,
                    'std': float(df[col].std()) if df[col].std() else None,
                    'min': float(df[col].min()) if df[col].min() else None,
                    'max': float(df[col].max()) if df[col].max() else None,
                    'median': float(df[col].median()) if df[col].median() else None
                })
            
            # Categorical stats
            elif df[col].dtype == pl.Utf8:
                col_profile['unique_count'] = df[col].n_unique()
                col_profile['top_values'] = df[col].value_counts().head(5).to_dicts()
            
            profile['columns'][col] = col_profile
        
        con.close()
        return profile
    
    def _calculate_quality_score(self, df: pl.DataFrame, original_rows: int, original_cols: int) -> float:
        """Calculate data quality score"""
        # Factors: completeness, consistency, validity
        
        # Completeness (how much data retained)
        completeness = len(df) / original_rows if original_rows > 0 else 0
        
        # Missing values ratio
        total_cells = len(df) * len(df.columns)
        null_cells = sum(df[col].null_count() for col in df.columns)
        missing_ratio = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        
        # Combine scores
        quality_score = (completeness * 0.4) + (missing_ratio * 0.6)
        
        return quality_score