"""Project root used to find configs/, artifacts/, data/, reports/ and dashboard/.

In a source checkout this is the repository root. When the package is installed
into site-packages (the Docker image), set PROMPTSHIELD_HOME to the directory
that holds those folders.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("PROMPTSHIELD_HOME") or Path(__file__).resolve().parents[2])
