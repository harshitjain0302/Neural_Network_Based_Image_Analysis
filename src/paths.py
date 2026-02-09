from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
ROCKS_360_DIR = DATA_DIR / "360_rocks"
ROCKS_120_DIR = DATA_DIR / "120_rocks"

HUMAN_DIR = DATA_DIR / "human_ratings"
MDS_360_PATH = HUMAN_DIR / "mds_360.txt"
MDS_120_PATH = HUMAN_DIR / "mds_120.txt"
