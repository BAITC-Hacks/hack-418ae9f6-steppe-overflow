"""Проверка основного пользовательского сценария на сайте (по умолчанию — https://steppewind.energy).

Проходит все страницы как пользователь и печатает PASS/FAIL по каждому шагу.
Нужны Playwright (pip install playwright) и Google Chrome:
  python scripts/check_site.py [URL] [--no-llm]
--no-llm — не задавать вопрос в чат (не тратить запрос к OpenAI).
"""

import sys

from playwright.sync_api import sync_playwright

URL = next((a for a in sys.argv[1:] if a.startswith("http")), "https://steppewind.energy").rstrip("/")
USE_LLM = "--no-llm" not in sys.argv
results = []


def step(name):
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((True, name, detail))
        except Exception as e:  # noqa: BLE001 — нужен отчёт по всем шагам
            results.append((False, name, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"))
        return fn
    return wrap


def no_errors(page):
    n = page.locator('[data-testid="stException"]').count()
    assert n == 0, f"на странице {n} ошибок Streamlit"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 900}, accept_downloads=True)
        wait = page.wait_for_timeout

        @step("Главная: карточки, AI-прогноз, график")
        def _():
            page.goto(URL)
            page.get_by_text("AI-прогноз").first.wait_for(timeout=60000)
            page.locator(".js-plotly-plot").first.wait_for(timeout=30000)
            assert page.locator(".sw-kpi").count() == 4
            no_errors(page)
            return page.locator(".sw-issue").inner_text().replace("\n", " · ")

        @step("Листание выпусков кнопкой ›")
        def _():
            issue = page.locator(".sw-issue")
            before = issue.inner_text().splitlines()[0]
            page.locator(".st-key-issue_forecast_next button").first.click()
            wait(2500)
            after = issue.inner_text().splitlines()[0]
            assert before != after, "дата не сменилась"
            no_errors(page)
            return f"{before} → {after.split(': ')[-1]}"

        @step("Пересчёт выпуска совпадает с сохранённым")
        def _():
            page.get_by_role("button", name="Пересчитать").click()
            page.get_by_text("Пересчитано за").wait_for(timeout=30000)
            text = page.get_by_text("Пересчитано за").inner_text()
            assert "совпадает" in text, text
            return text

        @step("Скачивание CSV прогноза")
        def _():
            with page.expect_download() as d:
                page.get_by_role("button", name="Скачать данные").click()
            path = d.value.path()
            lines = open(path, encoding="utf-8").read().splitlines()
            assert len(lines) == 49, f"строк {len(lines)}"
            return d.value.suggested_filename

        @step("Режим «Сейчас» — живой прогноз")
        def _():
            page.locator(".st-key-fc_mode").get_by_text("Сейчас", exact=True).click()
            page.get_by_text("Живой прогноз").first.wait_for(timeout=30000)
            wait(1500)
            no_errors(page)
            return page.locator(".sw-issue").inner_text().replace("\n", " · ")

        @step("Режим «История» — прогноз против факта")
        def _():
            page.locator(".st-key-fc_mode").get_by_text("История", exact=True).click()
            page.get_by_text("Прогноз против факта").first.wait_for(timeout=30000)
            wait(1500)
            no_errors(page)

        @step("Погода: 4 модели и проверка публикации")
        def _():
            page.goto(URL + "/weather")
            page.locator(".js-plotly-plot").first.wait_for(timeout=60000)
            wait(1500)
            assert page.get_by_text("ICON (DWD)").count() > 0
            no_errors(page)

        @step("Качество прогноза: метрики бэктеста")
        def _():
            page.goto(URL + "/quality")
            page.get_by_text("Средняя ошибка").wait_for(timeout=60000)
            wait(1500)
            no_errors(page)
            return page.locator(".sw-kpi, [data-testid='stMetric']").first.inner_text().replace("\n", " ")

        @step("Данные: пример январь 2026 сопоставлен с прогнозом")
        def _():
            page.goto(URL + "/upload")
            page.get_by_role("button", name="Попробовать на примере").click()
            page.get_by_text("Сопоставлено").wait_for(timeout=60000)
            no_errors(page)
            return page.get_by_text("Сопоставлено").inner_text()[:80]

        @step("AI-агент: запуск (правила) и демо сбоя ICON")
        def _():
            page.goto(URL + "/agent")
            page.get_by_text("Выпуск · данные на").first.wait_for(timeout=60000)
            page.get_by_text("Правила", exact=True).first.click()
            page.locator('[data-testid="stSelectbox"]').nth(1).click()
            page.get_by_text("Испорчен прогноз ICON", exact=False).first.click()
            page.get_by_role("button", name="Запустить агента").click()
            page.get_by_text("Решения агента").wait_for(timeout=60000)
            assert page.get_by_text("исключаю подозрительные источники").count() > 0
            no_errors(page)

        if USE_LLM:
            @step("Чат: ответ AI-помощника")
            def _():
                page.goto(URL + "/")
                page.get_by_text("AI-прогноз").first.wait_for(timeout=60000)
                page.locator(".st-key-chat_fab button").first.click()
                page.get_by_role("button", name="Когда завтра пик выработки?").click()
                msg = page.locator('[data-testid="stChatMessage"]').nth(1)
                msg.wait_for(timeout=60000)
                return msg.inner_text()[:100]

        @step("О системе")
        def _():
            page.goto(URL + "/about")
            page.get_by_text("Честность ретроспективы").wait_for(timeout=60000)
            no_errors(page)

        @step("PWA: манифест и service worker")
        def _():
            for path in ("/manifest.webmanifest", "/sw.js", "/pwa/icon-512.png"):
                r = page.request.get(URL + path)
                assert r.status == 200, f"{path}: {r.status}"
            return "3 файла доступны"

        browser.close()

    ok = sum(r[0] for r in results)
    for passed, name, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    print(f"\nИтого: {ok}/{len(results)} шагов пройдено")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
