# SPDX-License-Identifier: Apache-2.0
"""Import the skill's MENU_GAME fixture and drive it through GameCaptureWorker.

Run with timeout 90s python integrations_v2/eden/tests/check_standalone_game.py. Needs
Playwright with Chromium, and Three.js 0.161.0 in games/node_modules or at
EDEN_THREE_MODULE.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SKILL = REPO / 'skills/edengolem'
sys.path.insert(0, str(SKILL / 'forge/tests'))
from gate_fixtures import MENU_GAME, MENU_STRATEGY  # noqa: E402

from eden.apps.game2v.game_renderer import GameCaptureWorker, GameRenderer  # noqa: E402
from eden.apps.game2v.game_timeline import GameTimeline  # noqa: E402
from eden.apps.game2v.import_game import import_game  # noqa: E402

work = Path(tempfile.mkdtemp(prefix='standalone-check-'))
try:
    project = work / 'project'
    run = project / 'runs/step3/final'
    run.mkdir(parents=True)
    for name, content in MENU_GAME.items():
        (run / name).write_text(content)
    (run / 'strategy.json').write_text(json.dumps(MENU_STRATEGY))
    shutil.move(run / 'brief.json', project / 'brief.json')
    (project / 'package.json').write_text(json.dumps({'private': True, 'type': 'module', 'dependencies': {'three': '0.161.0'}}))
    stage = {'status': 'passed', 'evidence': [], 'loop_count': 0}
    (project / '.state.json').write_text(json.dumps({'current_stage': 'done', 'stages': {
        'intake': {**stage, 'artifact': 'brief.json'}, 'scene': {**stage, 'artifact': 'runs/step1/scene'},
        'behavior': {**stage, 'artifact': 'runs/step2/behavior'}, 'acceptance': {**stage, 'artifact': 'runs/step3/final'}}}))
    for rel in ('assets/interaction/MenuSelectionController.js', 'assets/math/ScalarUtils.js'):
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SKILL / rel, project / rel)

    games = work / 'games'
    games.mkdir()
    shutil.copytree(REPO / 'games/host', games / 'host')
    entry = import_game(project, 'menu-route', root=games,
                        event_prompts={'menu.committed': 'The traveller commits to a route.'})
    print('imported', entry['slug'], '| restart keys', entry['restart_keys'])
    cover = games / 'menu-route/cover.png'
    from PIL import Image
    print('captured cover', Image.open(cover).size)

    renderer = GameRenderer('menu-route', games_root=games)
    try:
        # Screenshot decoding needs native RAF, but must not advance the game.
        result = renderer.page.evaluate('''async () => {
            const clock = () => ({now: performance.now(), ...window.__gameTest.read().clock});
            const before = clock();
            const shot = await Promise.race([
                window.gameBackend.advanceToFrame({events: [], targetTick: 0}),
                new Promise(resolve => setTimeout(() => resolve(null), 5000)),
            ]);
            await new Promise(resolve => setTimeout(resolve, 200));
            return {captured: Boolean(shot?.ui), before, after: clock()};
        }''')
        assert result['captured'], 'Screenshot stalled waiting for a virtual animation frame'
        assert result['before'] == result['after'], result
        print('screenshot completed with the simulation clock frozen')
    finally:
        renderer.close()

    worker = GameCaptureWorker('menu-route', games_root=games)
    try:
        timeline = GameTimeline('menu-route')
        first = worker.capture(timeline.begin_chunk(12, 16))
        assert len(first.frames) == 12
        assert first.frames[-1].simulation_tick == 45, first.frames[-1].simulation_tick
        assert first.frames[0].ui_frame.shape == (480, 832, 3), first.frames[0].ui_frame.shape
        assert first.frames[0].ui_frame.std() > 1
        state = first.frames[-1].ui_state
        assert state['state'] == 'RUNNING' and state['selected'] == 0, state
        assert 'frame' not in state and 'clock' not in state
        print('chunk 1: 12 frames, tick', first.frames[-1].simulation_tick, '| state', state['state'])

        # Keys arrive the way the WebRTC client sends them: KeyboardEvent.key lowercased.
        sent = {'n': 0}
        def win():
            sent['n'] += 1
            return {1: [{'action': 'arrowdown', 'phase': 'down'}, {'action': 'arrowdown', 'phase': 'up'}],
                    3: [{'action': 'enter', 'phase': 'down'}, {'action': 'enter', 'phase': 'up'}]}.get(sent['n'], [])
        won = worker.capture(timeline.begin_chunk(12, 16), drain_events=win)
        final = won.frames[-1].ui_state
        assert final['state'] == 'WON', final
        prompts = [e['prompt'] for e in final['events'] if e['type'] == 'menu.committed']
        assert prompts == ['The traveller commits to a route.'], final['events']
        moved = [e['prompt'] for e in final['events'] if e['type'] == 'menu.moved']
        assert moved == [None], moved
        print('chunk 2: arrowdown+enter -> WON | event prompt', prompts[0])

        # The overlay's restart button sends client_action "restart"; the host
        # presses the game's own restart key. probe() must not advance the tick.
        frozen = won.simulation_tick_end
        reset = worker.probe(drain_events=lambda: [{'action': 'restart', 'phase': 'down'},
                                                   {'action': 'restart', 'phase': 'up'}])
        assert reset['state']['state'] == 'RUNNING', reset
        assert reset['state']['roundId'] == final['roundId'] + 1, (reset, final['roundId'])
        assert reset['tick'] == frozen, (reset['tick'], frozen)
        resumed = worker.capture(timeline.begin_chunk(4, 16))
        assert resumed.simulation_tick_start == frozen
        print('probe: restart -> RUNNING, roundId', final['roundId'], '->', reset['state']['roundId'],
              '| tick held at', frozen)
    finally:
        worker.close()
    print('Standalone host checks passed')
finally:
    shutil.rmtree(work, ignore_errors=True)
