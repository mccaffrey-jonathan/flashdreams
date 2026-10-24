# SPDX-License-Identifier: Apache-2.0
"""Browser integration checks: python .../check_game_backend.py --render.

Run inside the runtime pod with Playwright Chromium and games npm dependencies.
No model weights are needed. Exercises the real Python-to-Chromium capture path.
"""
import json
import sys
import types
from pathlib import Path

package = types.ModuleType('game_backend_check')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'apps/game2v')]
sys.modules[package.__name__] = package
from game_backend_check.game_timeline import GameTimeline

timeline = GameTimeline('freeze')
assert timeline.begin_chunk(24, 24).simulation_tick_end == 60
assert timeline.begin_chunk(16, 16).simulation_tick_end == 120
print('Timeline checks passed', flush=True)

if '--render' in sys.argv:
    from game_backend_check.game_renderer import GameRenderer, registered_game_slugs

    games = Path(__file__).resolve().parents[3] / 'games'
    for slug in registered_game_slugs():
        renderer = GameRenderer(slug, width=832, height=480)
        try:
            errors = []
            renderer.page.on('pageerror', lambda error: errors.append(str(error)))
            initial = json.loads((games / slug / 'initial-state.json').read_text())
            state = renderer.probe()['state']
            assert state['frame'] == initial['frame'], (slug, state, initial['frame'])
            assert state['state'] == 'RUNNING'
            for fps in (24, 16):
                renderer.probe(drain_events=lambda: [{'action': 'restart', 'phase': 'down'}])
                # Backend target ticks stay monotonic across a game reset.
                start = renderer.page.evaluate('window.gameBackend.probe([]).tick')
                states = []
                for index in range(1, fps + 1):
                    result = renderer.page.evaluate('args => window.gameBackend.advanceToFrame(args)', {
                        'events': [], 'targetTick': start + round(index * 60 / fps),
                    })
                    states.append(result['state'])
                    assert (result['width'], result['height']) == (832, 480)
                    assert result['ui'].startswith('data:image/jpeg;base64,')
                assert states[-1]['frame'] - initial['frame'] == 24, (slug, fps, states[-1])
                assert any(s['events'] for s in states)
            # P pauses without advancing simulation; another P resumes while held.
            renderer.probe(drain_events=lambda: [{'action': 'p', 'phase': 'down'}])
            before = renderer.probe()['state']['frame']
            tick = renderer.page.evaluate('window.gameBackend.probe([]).tick')
            renderer.page.evaluate('tick => window.gameBackend.advanceToFrame({targetTick:tick,events:[]})', tick + 60)
            assert renderer.probe()['state']['frame'] == before
            renderer.probe(drain_events=lambda: [{'action': 'p', 'phase': 'down'}])
            renderer.page.evaluate('tick => window.gameBackend.advanceToFrame({targetTick:tick,events:[]})', tick + 120)
            assert renderer.probe()['state']['frame'] == before + 24
            # Validate native controls, victory and snapshot restore through the imported game.
            result = renderer.page.evaluate('''async slug => {
              const {Game, config, dynamicPrompt} = await import(`/${slug}/adapter.js`);
              const initial = await (await fetch(`/${slug}/initial-state.json`)).json();
              const game = new Game(initial);
              if (!config.validSnapshot(game.snapshot())) throw Error('Invalid bundled initial state');
              for (let i = 0; i < 2400 && !game.s.won && !game.s.dead; i++) game.step(game.demoInput());
              return {won:game.s.won, dead:game.s.dead, frame:game.s.frame, prompt:dynamicPrompt([game.snapshot()])};
            }''', slug)
            assert result['won'], (slug, result)
            assert not errors, errors
            print(slug, 'PASS: cover state, 24/16 FPS timing, capture, prompt, pause/resume, demo victory', flush=True)
        finally:
            renderer.close()
