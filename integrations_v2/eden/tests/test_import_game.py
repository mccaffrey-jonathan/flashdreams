# SPDX-License-Identifier: Apache-2.0
"""Tests for the GamePrototypeTurbo → Eden import path.

Validates import_game and standalone.js without a browser or GPU by
building a minimal skill project fixture and checking the resulting
registry, file layout, and overlay generation.
"""

import json
import os
import shutil
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

# ---------------------------------------------------------------------------
# Fixture: a minimal accepted skill project
# ---------------------------------------------------------------------------

_BRIEF = {
    "title": "Star Course",
    "core_verb": "Run and jump to collect stars.",
    "fantasy": "A mascot platformer on floating platforms.",
    "control_scheme": "third_person_chase",
    "player_modality": "avatar",
    "fail_state": "You fall.",
    "win_state": "Collect all stars.",
    "interaction_tags": [],
    "time_budget_seconds": 60,
    "context": {},
    "scene": {
        "description": "Bright floating platforms above a void.",
        "theme_elements": [{"id": "player", "label": "mascot", "role": "actor"}],
    },
    "ui": {"mode": "polish"},
}

_META = {
    "control_scheme": "third_person_chase",
    "player_modality": "avatar",
    "core_verb": "Run and jump",
    "modules": ["/assets/math/TimeUtils.js"],
    "controls": ["WASD: move", "Space: jump", "R: restart"],
    "gate": "acceptance",
    "capabilities": [],
    "scene_manifest": "scene-manifest.json",
    "brief": "../../../brief.json",
    "signature_actions": [],
}

_STRATEGY = {
    "restart_actions": [{"type": "tap", "key": "r"}],
    "interaction_events": ["player.jumped"],
    "scenarios": [
        {"expect": "WON", "rules": []},
        {"expect": "LOST", "rules": []},
        {"expect": "WON", "rules": []},
    ],
}

# A minimal index.html that the standalone host recognises.
_INDEX_HTML = textwrap.dedent("""\
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Star Course</title>
      <script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script>
    </head>
    <body>
      <div id="app"></div>
      <div id="hud"></div>
      <div id="fatal" style="display:none"></div>
      <script type="module">
        import('./main.js').then(({start}) => start()).catch(e => {
          document.getElementById('fatal').style.display = 'block';
          document.getElementById('fatal').textContent = e?.stack ?? String(e);
        });
      </script>
    </body>
    </html>
""")

_MAIN_JS = textwrap.dedent("""\
    export async function start() {
      const c = document.createElement('canvas');
      c.width = 320; c.height = 240;
      document.querySelector('#app').appendChild(c);
      const ctx = c.getContext('2d');
      ctx.fillStyle = '#444'; ctx.fillRect(0, 0, 320, 240);
      let roundId = 1, state = 'RUNNING';
      window.__scene = { scene: {}, camera: {} };
      window.__gameTest = Object.freeze({
        read: () => ({ state, roundId, elapsed: 0, terminalCount: 0, events: [] }),
      });
    }
""")

_PACKAGE_JSON = {
    "private": True,
    "type": "module",
    "dependencies": {"three": "0.161.0"},
}


def _write_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    run = project / "runs" / "step3" / "final"
    run.mkdir(parents=True)
    (project / "brief.json").write_text(json.dumps(_BRIEF))
    (project / "package.json").write_text(json.dumps(_PACKAGE_JSON))
    (project / ".state.json").write_text(json.dumps({
        "current_stage": "done",
        "stages": {
            "intake": {"status": "passed", "evidence": [], "loop_count": 0, "artifact": "brief.json"},
            "scene": {"status": "passed", "evidence": [], "loop_count": 0, "artifact": "runs/step1/scene"},
            "behavior": {"status": "passed", "evidence": [], "loop_count": 0, "artifact": "runs/step2/behavior"},
            "acceptance": {"status": "passed", "evidence": [], "loop_count": 0, "artifact": "runs/step3/final"},
        },
    }))
    (run / "index.html").write_text(_INDEX_HTML)
    (run / "main.js").write_text(_MAIN_JS)
    (run / "step3.json").write_text(json.dumps(_META))
    (run / "strategy.json").write_text(json.dumps(_STRATEGY))
    (run / "scene-manifest.json").write_text('{"elements":[]}')
    # Evidence directory should be excluded from the copy.
    evidence = run / "evidence"
    evidence.mkdir()
    (evidence / "report.json").write_text('{"ok": true}')
    # Assets closure.
    asset_dir = project / "assets" / "math"
    asset_dir.mkdir(parents=True)
    (asset_dir / "TimeUtils.js").write_text("export class FrameStepper {}")
    return project


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_import_game_creates_registry_entry(tmp_path):
    """A full import produces a game dir, overlay, and a registry entry."""
    from eden.apps.game2v.import_game import import_game

    project = _write_project(tmp_path)
    root = tmp_path / "games"
    root.mkdir()
    # Provide a dummy cover so we don't need Playwright.
    cover = tmp_path / "cover.png"
    from PIL import Image
    Image.new("RGB", (1024, 576), (80, 120, 160)).save(cover)

    entry = import_game(
        project, "star-course", root=root, cover=cover,
        capture=lambda *a, **kw: None,  # skip browser capture
    )
    assert entry["slug"] == "star-course"
    assert entry["host"] == "standalone"
    assert entry["vendor"] == {"three": "0.161.0"}
    assert entry["restart_keys"] == ["r"]
    assert entry["controls"][0] == "WASD: move"

    # Files on disk.
    game_dir = root / "star-course"
    assert (game_dir / "index.html").is_file()
    assert (game_dir / "main.js").is_file()
    assert (game_dir / "assets" / "math" / "TimeUtils.js").is_file()
    assert (game_dir / "hud" / "overlay.js").is_file()
    assert (game_dir / "hud" / "overlay.css").is_file()
    assert (game_dir / "cover.png").is_file()
    # Evidence was not copied.
    assert not (game_dir / "evidence").exists()
    # Run metadata was not copied.
    assert not (game_dir / "step3.json").exists()

    # Registry.
    registry = json.loads((root / "registry.json").read_text())
    slugs = [game["slug"] for game in registry["games"]]
    assert "star-course" in slugs


def test_import_game_replace(tmp_path):
    """--replace overwrites an earlier import."""
    from eden.apps.game2v.import_game import import_game

    project = _write_project(tmp_path)
    root = tmp_path / "games"
    root.mkdir()
    cover = tmp_path / "cover.png"
    from PIL import Image
    Image.new("RGB", (1024, 576)).save(cover)

    noop = lambda *a, **kw: None
    import_game(project, "star-course", root=root, cover=cover, capture=noop)
    # A second import without --replace fails.
    from eden.apps.game2v.import_game import GameImportError
    with pytest.raises(GameImportError, match="already exists"):
        import_game(project, "star-course", root=root, cover=cover, capture=noop)
    # With --replace it succeeds.
    import_game(project, "star-course", root=root, cover=cover, replace=True, capture=noop)
    registry = json.loads((root / "registry.json").read_text())
    assert sum(1 for g in registry["games"] if g["slug"] == "star-course") == 1


def test_import_game_rollback_on_failure(tmp_path):
    """If the import fails mid-way, the registry is restored."""
    from eden.apps.game2v.import_game import import_game

    project = _write_project(tmp_path)
    root = tmp_path / "games"
    root.mkdir()
    original = {"games": [{"slug": "existing", "host": "standalone"}]}
    (root / "registry.json").write_text(json.dumps(original))

    def exploding_capture(*args, **kwargs):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        import_game(project, "star-course", root=root, capture=exploding_capture)
    # Registry should be restored.
    registry = json.loads((root / "registry.json").read_text())
    assert registry == original
    # Game directory should be cleaned up.
    assert not (root / "star-course").exists()


def test_import_game_refuses_handported(tmp_path):
    """A game without host: standalone cannot be replaced."""
    from eden.apps.game2v.import_game import GameImportError, import_game

    project = _write_project(tmp_path)
    root = tmp_path / "games"
    root.mkdir()
    (root / "registry.json").write_text(json.dumps({"games": [{"slug": "star-course"}]}))
    (root / "star-course").mkdir()

    with pytest.raises(GameImportError, match="hand-ported"):
        import_game(project, "star-course", root=root, replace=True,
                     capture=lambda *a, **kw: None)


def test_import_game_allow_incomplete(tmp_path):
    """--allow-incomplete imports a run that hasn't passed acceptance."""
    from eden.apps.game2v.import_game import import_game

    project = _write_project(tmp_path)
    # Rewrite state so acceptance is not passed.
    state = json.loads((project / ".state.json").read_text())
    state["current_stage"] = "behavior"
    state["stages"]["acceptance"]["status"] = "not_started"
    state["stages"]["acceptance"].pop("artifact", None)
    state["stages"]["behavior"]["status"] = "passed"
    state["stages"]["behavior"]["artifact"] = "runs/step2/behavior"
    # Create a step2 run.
    run2 = project / "runs" / "step2" / "behavior"
    run2.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project / "runs" / "step3" / "final", run2, dirs_exist_ok=True)
    (project / ".state.json").write_text(json.dumps(state))

    root = tmp_path / "games"
    root.mkdir()
    cover = tmp_path / "cover.png"
    from PIL import Image
    Image.new("RGB", (1024, 576)).save(cover)

    entry = import_game(
        project, "star-course", root=root, cover=cover,
        allow_incomplete=True, capture=lambda *a, **kw: None,
    )
    assert entry["slug"] == "star-course"
    assert (root / "star-course" / "index.html").is_file()


def test_standalone_page_injection():
    """standalone_page injects the config and script tag after <head>."""
    from eden.apps.game2v.game_renderer import standalone_page

    entry = {"restart_keys": ["r", "escape"], "event_prompts": {"round.won": "Victory!"}}
    result = standalone_page(_INDEX_HTML, entry)
    assert "window.gameStandalone=" in result
    assert "/host/standalone.js" in result
    assert '"restartKeys": ["r", "escape"]' in result
    assert "Victory!" in result
    # The injection appears after <head> and before the first element.
    head_end = result.index("</head>")
    script_pos = result.index("window.gameStandalone=")
    assert script_pos < head_end


def test_overlay_has_restart_button(tmp_path):
    """The generated overlay has a data-game-action=restart button."""
    from eden.apps.game2v.import_game import write_overlay

    write_overlay(tmp_path, "Test Game")
    js = (tmp_path / "hud" / "overlay.js").read_text()
    assert 'data-game-action="restart"' in js
    assert "Test Game" in js
    css = (tmp_path / "hud" / "overlay.css").read_text()
    assert ".result" in css


def test_event_prompts_come_from_the_game_source(tmp_path):
    """A game that emits typed events gets a sentence per type without being told.

    The chunk-event panel shows `event.prompt`, and a skill game's snapshot
    carries only the type, so the sentence has to be derived at import. A table
    passed in explicitly still wins, and a type built dynamically is skipped.
    """
    from eden.apps.game2v.import_game import event_prompts_from_source, registry_entry

    project = _write_project(tmp_path)
    run = project / "runs" / "step3" / "final"
    (run / "game.mjs").write_text(
        "emit('race.checkpoint', {n});\n"
        "emit(state === 'WON' ? 'round.won' : 'round.lost', {reason});\n"
        "emit(kind, {});\n",
        encoding="utf-8",
    )
    derived = event_prompts_from_source(run)
    assert derived == {
        "race.checkpoint": "race checkpoint",
        "round.lost": "round lost",
        "round.won": "round won",
    }
    entry = registry_entry(project, run, "test-game")
    assert entry["event_prompts"] == derived
    explicit = registry_entry(project, run, "test-game", event_prompts={"race.checkpoint": "A checkpoint."})
    assert explicit["event_prompts"] == {"race.checkpoint": "A checkpoint."}
