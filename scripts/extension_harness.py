"""Serve the API plus the extension test harness on one origin.

    python scripts/extension_harness.py        # then open http://localhost:8001/harness

The harness page loads the real extension/content.js and content.css with only
chrome.runtime stubbed, so the paste-interception flow can be tested in any browser
without installing the extension.
"""

from pathlib import Path

import uvicorn
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount, Route

from promptshield.serving.app import app

EXT = Path(__file__).resolve().parents[1] / "extension"

# insert ahead of the dashboard's catch-all "/" mount
app.router.routes.insert(0, Mount("/ext", app=StaticFiles(directory=EXT), name="ext"))
app.router.routes.insert(0, Route("/harness", lambda _request: FileResponse(EXT / "harness.html")))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
