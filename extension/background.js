// Calls the PromptShield API on behalf of content scripts (avoids page CSP / CORS issues).
const DEFAULTS = { apiUrl: "http://localhost:8000", apiKey: "dev-key", team: "" };

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.sync.get(Object.keys(DEFAULTS))) };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "promptshield:redact") return false;
  (async () => {
    try {
      const s = await settings();
      const res = await fetch(`${s.apiUrl}/v1/redact`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": s.apiKey },
        body: JSON.stringify({ text: msg.text, team: s.team || null }),
      });
      const body = await res.json();
      sendResponse(res.ok ? { ok: true, result: body } : { ok: false, error: body.detail || res.status });
    } catch (err) {
      sendResponse({ ok: false, error: String(err) });
    }
  })();
  return true; // async response
});
