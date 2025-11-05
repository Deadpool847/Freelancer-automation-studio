import polars as pl
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict
import json
from loguru import logger
from datetime import datetime

class IOHelper:
    """I/O utilities for data handling"""
    
    def __init__(self, base_path: Path = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "data"
        
        self.base_path = Path(base_path)
        
        # Create directory structure
        for subdir in ['bronze', 'silver', 'gold', 'feature_store', 'models', 'reports']:
            (self.base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def save_upload(self, uploaded_file, layer: str = "bronze") -> Path:
        """Save uploaded file to specified layer"""
        file_ext = Path(uploaded_file.name).suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}{file_ext}"
        
        file_path = self.base_path / layer / filename
        
        # Read and save as parquet for consistency
        if file_ext == '.csv':
            df = pl.read_csv(uploaded_file)
        elif file_ext in ['.xlsx', '.xls']:
            df = pl.from_pandas(pd.read_excel(uploaded_file))
        elif file_ext == '.json':
            df = pl.read_json(uploaded_file)
        elif file_ext == '.parquet':
            df = pl.read_parquet(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Save as parquet
        parquet_path = file_path.with_suffix('.parquet')
        df.write_parquet(parquet_path)
        
        logger.info(f"Saved upload to {parquet_path}")
        return parquet_path
    
    def save_scraped_data(self, data: List[Dict], layer: str = "bronze") -> Path:
        """Save scraped data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraped_{timestamp}.parquet"
        file_path = self.base_path / layer / filename
        
        # Convert to DataFrame
        df = pl.DataFrame(data)
        df.write_parquet(file_path)
        
        logger.info(f"Saved scraped data to {file_path}")
        return file_path
    
    def save_to_silver(self, df: pl.DataFrame, run_id: str) -> Path:
        """Save cleaned data to silver layer"""
        filename = f"cleaned_{run_id}.parquet"
        file_path = self.base_path / "silver" / filename
        
        df.write_parquet(file_path)
        
        logger.info(f"Saved to silver: {file_path}")
        return file_path
    
    def save_to_gold(self, df: pl.DataFrame, run_id: str, name: str = "final") -> Path:
        """Save final processed data to gold layer"""
        filename = f"{name}_{run_id}.parquet"
        file_path = self.base_path / "gold" / filename
        
        df.write_parquet(file_path)
        
        logger.info(f"Saved to gold: {file_path}")
        return file_path
    
    def save_report(self, report: Dict, run_id: str) -> Path:
        """Save JSON report"""
        filename = f"report_{run_id}.json"
        file_path = self.base_path / "reports" / filename
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved report: {file_path}")
        return file_path
    
    def get_artifact_path(self, run_id: str, artifact_type: str) -> Path:
        """Get path to artifact"""
        artifact_map = {
            'bronze': self.base_path / "bronze",
            'silver': self.base_path / "silver" / f"cleaned_{run_id}.parquet",
            'gold': self.base_path / "gold",
            'model': self.base_path / "models" / f"{run_id}_*.joblib",
            'report': self.base_path / "reports" / f"report_{run_id}.json"
        }
        
        return artifact_map.get(artifact_type, self.base_path)
    
    def export_to_format(self, df: pl.DataFrame, output_path: Path, format: str = "parquet"):
        """Export DataFrame to specified format"""
        if format.lower() == "parquet":
            df.write_parquet(output_path)
        elif format.lower() == "csv":
            df.write_csv(output_path)
        elif format.lower() == "json":
            df.write_json(output_path)
        elif format.lower() == "excel":
            df.to_pandas().to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported to {output_path}")

class IOHelper:
    """I/O utilities for data handling"""
    
    def __init__(self, base_path: Path = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "data"
        
        self.base_path = Path(base_path)
        
        # Create directory structure
        for subdir in ['bronze', 'silver', 'gold', 'feature_store', 'models', 'reports']:
            (self.base_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def save_upload(self, uploaded_file, layer: str = "bronze") -> Path:
        """Save uploaded file to specified layer"""
        file_ext = Path(uploaded_file.name).suffix
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}{file_ext}"
        
        file_path = self.base_path / layer / filename
        
        # Read and save as parquet for consistency
        if file_ext == '.csv':
            df = pl.read_csv(uploaded_file)
        elif file_ext in ['.xlsx', '.xls']:
            df = pl.from_pandas(pd.read_excel(uploaded_file))
        elif file_ext == '.json':
            df = pl.read_json(uploaded_file)
        elif file_ext == '.parquet':
            df = pl.read_parquet(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Save as parquet
        parquet_path = file_path.with_suffix('.parquet')
        df.write_parquet(parquet_path)
        
        logger.info(f"Saved upload to {parquet_path}")
        return parquet_path
    
    def save_scraped_data(self, data: List[Dict], layer: str = "bronze") -> Path:
        """Save scraped data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scraped_{timestamp}.parquet"
        file_path = self.base_path / layer / filename
        
        # Convert to DataFrame
        df = pl.DataFrame(data)
        df.write_parquet(file_path)
        
        logger.info(f"Saved scraped data to {file_path}")
        return file_path
    
    def save_to_silver(self, df: pl.DataFrame, run_id: str) -> Path:
        """Save cleaned data to silver layer"""
        filename = f"cleaned_{run_id}.parquet"
        file_path = self.base_path / "silver" / filename
        
        df.write_parquet(file_path)
        
        logger.info(f"Saved to silver: {file_path}")
        return file_path
    
    def save_to_gold(self, df: pl.DataFrame, run_id: str, name: str = "final") -> Path:
        """Save final processed data to gold layer"""
        filename = f"{name}_{run_id}.parquet"
        file_path = self.base_path / "gold" / filename
        
        df.write_parquet(file_path)
        
        logger.info(f"Saved to gold: {file_path}")
        return file_path
    
    def save_report(self, report: Dict, run_id: str) -> Path:
        """Save JSON report"""
        filename = f"report_{run_id}.json"
        file_path = self.base_path / "reports" / filename
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved report: {file_path}")
        return file_path
    
    def get_artifact_path(self, run_id: str, artifact_type: str) -> Path:
        """Get path to artifact"""
        artifact_map = {
            'bronze': self.base_path / "bronze",
            'silver': self.base_path / "silver" / f"cleaned_{run_id}.parquet",
            'gold': self.base_path / "gold",
            'model': self.base_path / "models" / f"{run_id}_*.joblib",
            'report': self.base_path / "reports" / f"report_{run_id}.json"
        }
        
        return artifact_map.get(artifact_type, self.base_path)
    
    def export_to_format(self, df: pl.DataFrame, output_path: Path, format: str = "parquet"):
        """Export DataFrame to specified format"""
        if format.lower() == "parquet":
            df.write_parquet(output_path)
        elif format.lower() == "csv":
            df.write_csv(output_path)
        elif format.lower() == "json":
            df.write_json(output_path)
        elif format.lower() == "excel":
            df.to_pandas().to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported to {output_path}")