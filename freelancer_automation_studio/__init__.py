"""Freelancer Automation Studio - Full-Stack ML Pipeline"""

__version__ = "1.0.0"
__author__ = "Freelancer Automation Studio Team"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_ROOT = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_ROOT / "bronze"
SILVER_DIR = DATA_ROOT / "silver"
GOLD_DIR = DATA_ROOT / "gold"
FEATURE_STORE_DIR = DATA_ROOT / "feature_store"
MODELS_DIR = DATA_ROOT / "models"
REPORTS_DIR = DATA_ROOT / "reports"

# Metadata
METADATA_DIR = PROJECT_ROOT / "metadata"

# Configs
CONFIG_DIR = PROJECT_ROOT / "configs"

# Create directories
for directory in [
    BRONZE_DIR, SILVER_DIR, GOLD_DIR, FEATURE_STORE_DIR,
    MODELS_DIR, REPORTS_DIR, METADATA_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)