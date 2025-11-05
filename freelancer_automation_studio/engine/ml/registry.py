import joblib
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json
from loguru import logger

class ModelRegistry:
    """Model registry with SQLite backend"""
    
    def __init__(self, base_path: Path = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent / "data" / "models"
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.db_path = Path(__file__).parent.parent.parent / "metadata" / "runs.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                model_path TEXT NOT NULL,
                metrics TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_model(self, model, run_id: str, task_type: str, metrics: Dict) -> Path:
        """Save model to registry"""
        model_id = f"{run_id}_{task_type}"
        model_path = self.base_path / f"{model_id}.joblib"
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")
        
        # Register in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO models (model_id, run_id, task_type, model_path, metrics, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            run_id,
            task_type,
            str(model_path),
            json.dumps(metrics),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        return model_path
    
    def load_model(self, run_id: str, task_type: str):
        """Load model from registry"""
        model_id = f"{run_id}_{task_type}"
        model_path = self.base_path / f"{model_id}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return joblib.load(model_path)
    
    def get_model_info(self, run_id: str, task_type: str) -> Optional[Dict]:
        """Get model information"""
        model_id = f"{run_id}_{task_type}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model_id, run_id, task_type, model_path, metrics, created_at
            FROM models
            WHERE model_id = ?
        """, (model_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'model_id': row[0],
                'run_id': row[1],
                'task_type': row[2],
                'model_path': row[3],
                'metrics': json.loads(row[4]),
                'created_at': row[5]
            }
        
        return None
    
    def list_models(self, limit: int = 50) -> list:
        """List all registered models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model_id, run_id, task_type, metrics, created_at
            FROM models
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'model_id': row[0],
                'run_id': row[1],
                'task_type': row[2],
                'metrics': json.loads(row[3]),
                'created_at': row[4]
            }
            for row in rows
        ]