"""Запись демо-видео Steppe Wind: сценарий проходит боевой сайт, субтитры рисуются прямо на странице.

Нужны Playwright (pip install playwright) и Google Chrome; ffmpeg — для mp4/gif:
  python scripts/record_demo.py video
  ffmpeg -ss <INTRO_AT> -i video/*.webm -c:v libx264 -crf 23 -pix_fmt yuv420p -movflags +faststart -an docs/media/demo.mp4
"""

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://steppewind.energy"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "video")
W, H = 1600, 900

CAPTION_JS = """
(text) => {
  let el = document.getElementById('sw-cap');
  if (!el) {
    el = document.createElement('div'); el.id = 'sw-cap';
    Object.assign(el.style, {position:'fixed', left:'50%', bottom:'34px', transform:'translateX(-50%)',
      zIndex: 99999, background:'rgba(12,71,65,0.94)', color:'#fff', padding:'14px 26px', borderRadius:'14px',
      font:'600 21px Manrope, Roboto, sans-serif', maxWidth:'1180px', textAlign:'center', lineHeight:'1.4',
      boxShadow:'0 10px 30px rgba(12,71,65,.35)', transition:'opacity .35s'});
    document.body.appendChild(el);
  }
  el.style.opacity = text ? '1' : '0';
  if (text) el.innerHTML = text;
}
"""

CARD_JS = """
([title, sub]) => {
  let el = document.getElementById('sw-card');
  if (!title) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement('div'); el.id = 'sw-card';
    Object.assign(el.style, {position:'fixed', inset:'0', zIndex: 100000, display:'flex', flexDirection:'column',
      alignItems:'center', justifyContent:'center', gap:'18px', background:'linear-gradient(135deg,#0c4741 0%,#0f6e36 100%)',
      color:'#fff', font:'800 54px Manrope, Roboto, sans-serif', textAlign:'center'});
    document.body.appendChild(el);
  }
  el.innerHTML = `<img src="/pwa/favicon.svg" style="width:110px;height:110px;border-radius:24px">` +
    `<div>${title}</div><div style="font:500 26px Manrope, Roboto, sans-serif;opacity:.9;max-width:1100px">${sub}</div>`;
}
"""


def caption(page, text, hold_ms=0):
    page.evaluate(CAPTION_JS, text)
    if hold_ms:
        page.wait_for_timeout(hold_ms)


def card(page, title, sub, hold_ms):
    page.evaluate(CARD_JS, [title, sub])
    page.wait_for_timeout(hold_ms)
    page.evaluate(CARD_JS, [None, None])


def ready(page, text, timeout=30000):
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout)
    page.wait_for_timeout(1200)


def sweep_chart(page, y, x0=520, x1=1480, steps=24, pause=110):
    for i in range(steps + 1):
        page.mouse.move(x0 + (x1 - x0) * i / steps, y)
        page.wait_for_timeout(pause)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT),
                                  record_video_size={"width": W, "height": H}, locale="ru-RU")
        page = ctx.new_page()
        t_start = time.time()

        # 0. Заставка
        page.goto(URL)
        ready(page, "AI-прогноз", 60000)
        page.locator(".js-plotly-plot").first.wait_for(timeout=30000)
        page.wait_for_timeout(800)
        intro_at = time.time() - t_start
        card(page, "Steppe Wind", "Agentic AI-прогноз почасовой выработки ветроэлектростанции на 24–48 часов", 2800)

        # 1. Главная: прогноз, карточки, объяснение агента, график
        caption(page, "Прогноз на 48 часов по прогнозам погоды, опубликованным до момента выпуска (14:00)", 3000)
        caption(page, "AI-агент объясняет прогноз и почему уверенность такая: модели ветра расходятся на 3 м/с", 3000)
        page.mouse.wheel(0, 520)
        page.wait_for_timeout(900)
        caption(page, "Текущий прогноз, предыдущий выпуск и интервал P10–P90; в подсказке — ветер ECMWF")
        sweep_chart(page, 520, steps=20, pause=100)
        page.wait_for_timeout(600)

        # 2. Следующие выпуски
        page.mouse.wheel(0, -900)
        page.wait_for_timeout(600)
        caption(page, "28 ежедневных выпусков февраля 2026 — листаем, как диспетчер")
        nxt = page.locator(".st-key-issue_forecast_next button").first
        for _ in range(2):
            nxt.click()
            page.wait_for_timeout(1500)

        # 3. Живой режим
        page.locator(".st-key-fc_mode").get_by_text("Сейчас", exact=True).click()
        ready(page, "Живой прогноз")
        caption(page, "Живой режим: прогноз на настоящее «завтра». Планировщик сам пересчитывает при выходе нового прогона ECMWF", 4000)

        # 4. AI-агент: демо сбоя
        page.goto(URL + "/agent")
        ready(page, "Выпуск · данные на")
        caption(page, "AI-агент: погода → проверка → модель → анализ → публикация → пересчёт при новых данных", 3500)
        page.get_by_text("Правила", exact=True).first.click()
        page.wait_for_timeout(500)
        page.locator('[data-testid="stSelectbox"]').nth(1).click()
        page.wait_for_timeout(500)
        page.get_by_text("Испорчен прогноз ICON", exact=False).first.click()
        page.wait_for_timeout(500)
        caption(page, "Демо сбоя: портим прогноз ICON — агент должен заметить сам")
        page.get_by_role("button", name="Запустить агента").click()
        ready(page, "Решения агента", 60000)
        page.mouse.wheel(0, 180)
        caption(page, "Агент нашёл нефизичные значения ICON, исключил источник и объяснил решение", 5000)

        # 5. Погода
        page.goto(URL + "/weather")
        ready(page, "Погода на момент выпуска")
        page.locator(".js-plotly-plot").first.wait_for(timeout=30000)
        page.wait_for_timeout(1000)
        caption(page, "Прогнозы ECMWF, GFS, ICON на момент выпуска — и проверка, что все опубликованы до него", 3800)

        # 6. Качество
        page.goto(URL + "/quality")
        ready(page, "Средняя ошибка")
        page.locator(".js-plotly-plot").first.wait_for(timeout=30000)
        caption(page, "Бэктест по протоколу теста: ошибка 17 % номинала — на 49 % лучше климатологии", 3200)
        page.mouse.wheel(0, 700)
        caption(page, "Каждая точка — час прогноза против факта; модель обучена только на прошлом", 3500)

        # 7. Свои данные
        page.goto(URL + "/upload")
        ready(page, "Попробовать на примере")
        caption(page, "Загрузите свой CSV (например, факт за февраль) — точность посчитается сразу")
        page.get_by_role("button", name="Попробовать на примере").click()
        ready(page, "Сопоставлено", 60000)
        page.mouse.wheel(0, 300)
        page.wait_for_timeout(2800)

        # 8. Чат
        page.goto(URL + "/")
        ready(page, "AI-прогноз", 60000)
        caption(page, "AI-помощник отвечает на вопросы о прогнозе — только по фактам выпуска")
        page.locator(".st-key-chat_fab button").first.click()
        page.wait_for_timeout(1200)
        page.get_by_role("button", name="Когда завтра пик выработки?").click()
        page.locator('[data-testid="stChatMessage"]').nth(1).wait_for(timeout=60000)
        page.wait_for_timeout(3800)

        # 9. Финал
        page.keyboard.press("Escape")
        page.wait_for_timeout(400)
        caption(page, "")
        card(page, "steppewind.energy", "Код, данные и инструкции — в README репозитория", 3500)

        video = page.video.path()
        ctx.close()
        browser.close()
        print(video)
        print(f"INTRO_AT={intro_at:.2f}")


if __name__ == "__main__":
    main()
