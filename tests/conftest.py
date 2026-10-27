import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "backend", ROOT / "ml"):
    sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")
os.environ.setdefault("LIVE_DB_PATH", ":memory:")
os.environ.setdefault("LIVE_AUTOSTART", "false")  # tests drive ticks by hand
