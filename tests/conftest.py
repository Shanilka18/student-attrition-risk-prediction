"""Root pytest configuration and environment preparation."""

import os
import tempfile
from pathlib import Path

# Fix Matplotlib sandbox cache directory permissions
mpl_dir = Path(__file__).resolve().parent.parent / ".matplotlib_cache"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir)
