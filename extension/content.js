// Intercepts paste events in AI chat inputs, asks PromptShield,
// and shows a warning panel with "Redact & insert" / "Insert anyway" / "Cancel".
(() => {
  "use strict";

  const MIN_CHARS = 8;

  function isChatInput(el) {
    return el && (el.isContentEditable || el.tagName === "TEXTAREA");
  }

  function scan(text) {
    return new Promise((resolve) =>
      chrome.runtime.sendMessage({ type: "promptshield:redact", text }, (r) => resolve(r || { ok: false })));
  }

  function insertText(target, text) {
    target.focus();
    // execCommand keeps rich editors (ProseMirror etc.) in sync with their own state
    if (!document.execCommand("insertText", false, text)) {
      const start = target.selectionStart ?? target.value.length;
      target.setRangeText(text, start, target.selectionEnd ?? start, "end");
      target.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function panel(result, original, onChoice) {
    document.getElementById("promptshield-panel")?.remove();
    const counts = {};
    for (const e of result.entities) counts[e.label] = (counts[e.label] || 0) + 1;
    const summary = Object.entries(counts).map(([k, v]) => `${v} × ${k}`).join(", ");
    const blocked = result.action === "block";
    const el = document.createElement("div");
    el.id = "promptshield-panel";
    el.setAttribute("role", "alertdialog");
    el.innerHTML = `
      <div class="ps-title">${blocked ? "Secret detected" : "Sensitive data detected"}</div>
      <div class="ps-summary">${escapeHtml(summary)}</div>
      ${blocked
        ? `<div class="ps-note">Your company policy blocks sending credentials or keys to AI tools.</div>`
        : `<div class="ps-preview">${escapeHtml(result.redacted_text.slice(0, 400))}</div>`}
      <div class="ps-actions">
        ${blocked ? "" : '<button data-c="redact" class="ps-primary">Redact &amp; insert</button>'}
        <button data-c="raw">Insert anyway</button>
        <button data-c="cancel">Cancel</button>
      </div>`;
    el.addEventListener("click", (ev) => {
      const c = ev.target?.dataset?.c;
      if (!c) return;
      el.remove();
      onChoice(c);
    });
    document.body.appendChild(el);
    el.querySelector("button")?.focus();
  }

  document.addEventListener("paste", async (ev) => {
    const target = ev.target.closest?.("[contenteditable='true'], textarea");
    if (!isChatInput(target)) return;
    const text = ev.clipboardData?.getData("text/plain") || "";
    if (text.length < MIN_CHARS) return;
    ev.preventDefault();
    ev.stopImmediatePropagation();

    const r = await scan(text);
    if (!r.ok) {
      console.warn("PromptShield unavailable, pasting unchanged:", r.error);
      insertText(target, text);
      return;
    }
    if (r.result.action === "allow" || r.result.entities.length === 0) {
      insertText(target, text);
      return;
    }
    panel(r.result, text, (choice) => {
      if (choice === "redact") insertText(target, r.result.redacted_text);
      else if (choice === "raw") insertText(target, text);
    });
  }, true);
})();
