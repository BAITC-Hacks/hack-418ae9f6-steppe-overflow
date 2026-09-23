// Service worker Steppe Wind: делает сайт устанавливаемым (PWA) и показывает заставку без сети.
// Данные и прогнозы не кэшируются — Streamlit работает только онлайн через живое соединение.
const CACHE = "steppewind-v1";
const OFFLINE = "/offline.html";
const ASSETS = [OFFLINE, "/pwa/icon-192.png", "/pwa/favicon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  if (event.request.mode === "navigate") {
    event.respondWith(fetch(event.request).catch(() => caches.match(OFFLINE)));
  }
});
