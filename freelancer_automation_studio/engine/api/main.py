from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import sys
from pathlib import Path
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.utils.manifest import ManifestManager
from engine.utils.io_helpers import IOHelper

app = FastAPI(title="Freelancer Automation Studio API", version="1.0.0")

# Models
class RunCreate(BaseModel):
    mode: str
    source: str
    config: Optional[Dict] = {}

class RunStatus(BaseModel):
    run_id: str
    status: str
    progress: float
    current_stage: str

class ArtifactRequest(BaseModel):
    run_id: str
    artifact_type: str

# In-memory run tracking
active_runs: Dict[str, Dict] = {}

@app.get("/")
async def root():
    return {
        "service": "Freelancer Automation Studio API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.post("/api/runs/create")
async def create_run(run_data: RunCreate):
    """Create a new ML pipeline run"""
    manifest_mgr = ManifestManager()
    run_id = manifest_mgr.create_run({
        "mode": run_data.mode,
        "source": run_data.source,
        "config": run_data.config
    })
    
    active_runs[run_id] = {
        "status": "created",
        "progress": 0.0,
        "stage": "initialized"
    }
    
    return {"run_id": run_id, "status": "created"}

@app.get("/api/runs/{run_id}/status")
async def get_run_status(run_id: str):
    """Get status of a run"""
    if run_id not in active_runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return active_runs[run_id]

@app.get("/api/runs/{run_id}/manifest")
async def get_run_manifest(run_id: str):
    """Get manifest for a run"""
    manifest_mgr = ManifestManager()
    manifest = manifest_mgr.get_manifest(run_id)
    
    if not manifest:
        raise HTTPException(status_code=404, detail="Manifest not found")
    
    return manifest

@app.get("/api/runs/list")
async def list_runs(limit: int = 50):
    """List all runs"""
    manifest_mgr = ManifestManager()
    runs = manifest_mgr.list_runs(limit=limit)
    return {"runs": runs}

@app.post("/api/artifacts/download")
async def download_artifact(artifact_req: ArtifactRequest):
    """Download artifact from a run"""
    io_helper = IOHelper()
    artifact_path = io_helper.get_artifact_path(artifact_req.run_id, artifact_req.artifact_type)
    
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return FileResponse(
        path=artifact_path,
        filename=artifact_path.name,
        media_type='application/octet-stream'
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_runs": len(active_runs)
    }