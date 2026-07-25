// Relay: content script -> local Python bridge. Content scripts can't make
// cross-origin requests in MV3, so the fetch lives here (host_permissions).
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "advise") return;
  fetch("http://127.0.0.1:8765/advise", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log: msg.log, request: msg.request, mode: msg.mode }),
  })
    .then((r) => r.json())
    .then(sendResponse)
    .catch((err) => sendResponse({ ok: false, error: String(err) }));
  return true; // async sendResponse
});
