"""Capture README screenshots of the dashboard with headless Edge/Chrome (DevTools protocol).

    python scripts/capture_screenshots.py --url http://localhost:8000 --browser path/to/msedge-or-chrome
"""

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import websockets

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"


class Page:
    def __init__(self, ws):
        self.ws, self.n = ws, 0

    async def send(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def shot(self, name):
        data = (await self.send("Page.captureScreenshot", format="png"))["data"]
        (OUT / name).write_bytes(base64.b64decode(data))
        print("saved", OUT / name)


async def main(url: str, browser: str, port: int = 9333) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([browser, "--headless=new", f"--remote-debugging-port={port}",
                             f"--user-data-dir={profile}", "--window-size=1280,860", "about:blank"])
    try:
        for _ in range(50):
            try:
                targets = httpx.get(f"http://127.0.0.1:{port}/json").json()
                break
            except httpx.HTTPError:
                time.sleep(0.2)
        page_ws = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
        async with websockets.connect(page_ws, max_size=50_000_000) as ws:
            p = Page(ws)
            await p.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
            await p.send("Emulation.setDeviceMetricsOverride",
                         width=1280, height=860, deviceScaleFactor=1.5, mobile=False)
            await p.send("Page.enable")
            await p.send("Page.navigate", url=url)
            await asyncio.sleep(2.5)

            await p.js("document.getElementById('btnSend').click()")
            await asyncio.sleep(2.5)
            await p.shot("playground_redact.png")

            await p.js("document.querySelector('[data-example=\"secret\"]').click();"
                       "document.getElementById('btnSend').click()")
            await asyncio.sleep(2.5)
            await p.shot("playground_block.png")

            await p.js("document.querySelector('[data-tab=\"monitoring\"]').click()")
            await asyncio.sleep(3.5)
            await p.shot("monitoring.png")
    finally:
        proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--browser", default=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.browser))
