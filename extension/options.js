const FIELDS = ["apiUrl", "apiKey", "team"];
const DEFAULTS = { apiUrl: "http://localhost:8000", apiKey: "dev-key", team: "" };

chrome.storage.sync.get(FIELDS, (stored) => {
  for (const f of FIELDS) document.getElementById(f).value = stored[f] ?? DEFAULTS[f];
});

document.getElementById("save").addEventListener("click", async () => {
  const values = Object.fromEntries(FIELDS.map((f) => [f, document.getElementById(f).value.trim()]));
  // a non-localhost gateway needs an explicit host permission grant
  if (!values.apiUrl.startsWith("http://localhost")) {
    const origin = new URL(values.apiUrl).origin + "/*";
    const granted = await chrome.permissions.request({ origins: [origin] });
    if (!granted) {
      document.getElementById("saved").textContent = "permission denied";
      return;
    }
  }
  await chrome.storage.sync.set(values);
  document.getElementById("saved").textContent = "saved";
});
