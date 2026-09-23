"""PWA: манифест, иконки и правила Caddy для установки сайта как приложения."""

import json
import struct

from windagent.config import resolve

PWA = resolve("deploy/pwa")


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", head[16:24])


def test_manifest_is_installable():
    m = json.loads((PWA / "manifest.webmanifest").read_text(encoding="utf-8"))
    for key in ("name", "short_name", "start_url", "display", "icons", "theme_color"):
        assert key in m
    assert m["display"] == "standalone"
    sizes = {i["sizes"] for i in m["icons"] if i["type"] == "image/png"}
    assert {"192x192", "512x512"} <= sizes
    assert any(i.get("purpose") == "maskable" for i in m["icons"])
    for icon in m["icons"]:
        assert (PWA / icon["src"].removeprefix("/pwa/")).exists(), icon["src"]


def test_icon_sizes():
    assert _png_size(PWA / "icon-192.png") == (192, 192)
    assert _png_size(PWA / "icon-512.png") == (512, 512)
    assert _png_size(PWA / "maskable-512.png") == (512, 512)
    assert _png_size(PWA / "apple-touch-icon.png") == (180, 180)


def test_caddy_serves_pwa_files():
    caddy = resolve("deploy/Caddyfile").read_text(encoding="utf-8")
    assert "/manifest.webmanifest" in caddy and "/sw.js" in caddy and "handle_path /pwa/*" in caddy
    compose = resolve("docker-compose.yml").read_text(encoding="utf-8")
    assert "./deploy/pwa:/srv/pwa" in compose
    sw = (PWA / "sw.js").read_text(encoding="utf-8")
    assert "offline.html" in sw and "fetch" in sw
