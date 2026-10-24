"""Exercise the real HTTP preview and HUD routes for every bundled game.

Run inside the runtime pod with Playwright Chromium and games npm dependencies.
"""
import asyncio
import json
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer
from playwright.async_api import async_playwright

from eden.apps.game2v.panel import handle_game2v
from flashdreams.runtime_v2.serving.webrtc_server import WebRTCServer


async def main():
    # These handlers need only the checkout resources, not a running model.
    server = WebRTCServer.__new__(WebRTCServer)
    app = web.Application()
    app.router.add_get("/vendor/{filename}", server._serve_game_vendor)
    app.router.add_get("/host/{filename}", server._serve_game_host)
    app.router.add_get("/game/{slug}/{tail:.*}", server._serve_registered_game)
    app.router.add_get("/games/{slug}/hud/{filename}", server._serve_game_overlay)

    async def extension(request):
        return await handle_game2v(request, request.match_info["tail"])

    app.router.add_get("/ext/game2v/{tail:.*}", extension)
    games = Path(__file__).resolve().parents[3] / "games"
    registry = json.loads((games / "registry.json").read_text())["games"]
    async with TestServer(app) as http, async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        try:
            for entry in registry:
                slug = entry["slug"]
                page = await browser.new_page(viewport={"width": 1024, "height": 650})
                errors, missing = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("response", lambda response: missing.append(response.url) if response.status >= 400 else None)
                await page.goto(str(http.make_url(f"/ext/game2v/play/{slug}")))
                await page.wait_for_selector(".demo-hud .stats")
                await page.wait_for_function("document.querySelector('.stats').textContent.length > 0")
                await page.keyboard.press("p")
                await page.wait_for_selector(".result:not([hidden])")
                assert await page.locator(".result h2").text_content() == "Paused"
                await page.keyboard.press("p")
                await page.wait_for_selector(".result", state="hidden")
                await page.keyboard.down("f")
                await page.wait_for_timeout(300)
                await page.keyboard.up("f")
                # The generated-video HUD uses a separate module route and stylesheet.
                result = await page.evaluate("""async slug => {
                    const module = await import(`/games/${slug}/hud/overlay.js`);
                    const root = document.createElement('div');
                    document.body.append(root);
                    const view = module.mount(root);
                    view.update({slug, objective:'TEST', instruction:'Test', stats:[], crosshair:true,
                        outcome:['WON','You won','Play again'], paused:false});
                    let requested;
                    const original = window.fetch;
                    window.fetch = async (url, options) => {
                        requested = {url, body:JSON.parse(options.body)};
                        return {ok:true};
                    };
                    try { await root.querySelector('button').onclick(); }
                    finally { window.fetch = original; root.remove(); }
                    return requested;
                }""", slug)
                assert result == {"url": "/ext/game2v/switch", "body": {"slug": slug, "restart": True}}
                assert not errors, (slug, errors)
                assert not missing, (slug, missing)
                print(slug, "PASS: preview, shared modules, Three.js, HUD, pause and rollout restart", flush=True)
                if slug == "freeze":
                    output = Path(".cache/game-migration/preview.png")
                    await page.screenshot(path=str(output))
                await page.close()
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
