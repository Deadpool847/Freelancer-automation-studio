import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import uuid
import hashlib
from loguru import logger

class ManifestManager:
    """Manage run manifests and metadata"""
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "metadata" / "runs.sqlite"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                source TEXT NOT NULL,
                config TEXT,
                artifacts TEXT,
                timings TEXT,
                status TEXT DEFAULT 'created',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def create_run(self, metadata: Dict) -> str:
        """Create a new run"""
        run_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO runs (run_id, mode, source, config, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'created', ?, ?)
        """, (
            run_id,
            metadata.get('mode', 'unknown'),
            metadata.get('source', 'unknown'),
            json.dumps(metadata.get('config', {})),
            now,
            now
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created run: {run_id}")
        return run_id
    
    def update_run(self, run_id: str, updates: Dict):
        """Update run metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        set_clauses = []
        values = []
        
        for key, value in updates.items():
            if key in ['artifacts', 'timings', 'config']:
                set_clauses.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key in ['status', 'mode', 'source']:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        
        values.append(run_id)
        
        query = f"UPDATE runs SET {', '.join(set_clauses)} WHERE run_id = ?"
        cursor.execute(query, values)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated run: {run_id}")
    
    def get_manifest(self, run_id: str) -> Optional[Dict]:
        """Get run manifest"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT run_id, mode, source, config, artifacts, timings, status, created_at, updated_at
            FROM runs
            WHERE run_id = ?
        """, (run_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'run_id': row[0],
                'mode': row[1],
                'source': row[2],
                'config': json.loads(row[3]) if row[3] else {},
                'artifacts': json.loads(row[4]) if row[4] else {},
                'timings': json.loads(row[5]) if row[5] else {},
                'status': row[6],
                'created_at': row[7],
                'updated_at': row[8]
            }
        
        return None
    
    def list_runs(self, limit: int = 50) -> List[Dict]:
        """List all runs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT run_id, mode, source, status, created_at
            FROM runs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'run_id': row[0],
                'mode': row[1],
                'source': row[2],
                'status': row[3],
                'created_at': row[4]
            }
            for row in rows
        ]
    
    def compute_hash(self, data: str) -> str:
        """Compute SHA-256 hash of data"""
        return hashlib.sha256(data.encode()).hexdigest()