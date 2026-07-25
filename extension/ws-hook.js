// Runs in the page's MAIN world. Wraps WebSocket so the content script can
// (a) see every message the client receives and (b) send /choose commands
// through the live, logged-in socket — which is what makes private battles
// work: we're reading the user's own stream, not spectating from outside.
(() => {
  let sock = null;
  const Orig = window.WebSocket;

  function Hooked(url, protocols) {
    const ws = protocols === undefined ? new Orig(url) : new Orig(url, protocols);
    sock = ws;
    ws.addEventListener("message", (e) => {
      document.dispatchEvent(new CustomEvent("psa-recv", { detail: String(e.data) }));
    });
    return ws;
  }
  Hooked.prototype = Orig.prototype;
  ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach((k, i) => { Hooked[k] = i; });
  window.WebSocket = Hooked;

  document.addEventListener("psa-send", (e) => {
    if (sock && sock.readyState === 1) sock.send(String(e.detail));
  });
})();
