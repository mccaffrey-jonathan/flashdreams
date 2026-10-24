# SPDX-License-Identifier: Apache-2.0
"""Register a GamePrototypeTurbo project as an Eden game.

The skill (``skills/edengolem``) produces a standalone project: a
brief, a state file, ``runs/step{1,2,3}/<name>/`` and the ``assets/`` closure
its runs import. This copies the accepted run and that closure into
``games/<slug>/``, writes a minimal HUD overlay, and appends a registry entry
with ``host: standalone`` so :class:`GameRenderer` drives the page through
``host/standalone.js`` instead of the ``createRuntime()`` contract.

Generating the project is the skill's job and needs an agent; this only
imports the result::

    python -m eden.apps.game2v.import_game /path/to/project --slug star-course

The first frame of an Eden rollout is the game's ``cover``. Pass ``--cover``
with a photoreal image of the opening view when one exists; without it the
importer captures the game's own first frame, which conditions the model on a
greybox image.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .game_renderer import STANDALONE_HOST

THREE_VERSION = "0.161.0"
"""The one Three.js build every Eden game shares (``games/node_modules``)."""

_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
# Top-level names the renderer and the browser server already route.
_RESERVED_SLUGS = frozenset({"host", "assets", "vendor", "bench", "shared", "node_modules"})
# Per-run files that describe the skill's gates, not the game.
_RUN_METADATA = frozenset({
    "step1.json", "step2.json", "step3.json", "strategy.json",
    "source-artifact.json", "scene-manifest.json",
})


class GameImportError(ValueError):
    """The project cannot be imported as given."""


def games_root() -> Path:
    return Path(
        os.environ.get("EDEN_GAMES_ROOT", Path(__file__).resolve().parents[4] / "games")
    ).resolve()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GameImportError(f"Missing {path}") from None
    except json.JSONDecodeError as error:
        raise GameImportError(f"{path} is not valid JSON: {error}") from None


def accepted_run(project: Path, *, allow_incomplete: bool = False) -> Path:
    """Return the run directory to import.

    A finished project names its accepted run as the ``acceptance`` artifact.
    ``allow_incomplete`` falls back to the latest populated run, for trying a
    game that has not passed acceptance yet.
    """
    state = _read_json(project / ".state.json")
    acceptance = state.get("stages", {}).get("acceptance", {})
    if acceptance.get("status") == "passed" and acceptance.get("artifact"):
        run = (project / acceptance["artifact"]).resolve()
    elif allow_incomplete:
        candidates = [
            entry.parent
            for step in ("step3", "step2", "step1")
            for entry in sorted((project / "runs" / step).glob("*/index.html"))
        ]
        if not candidates:
            raise GameImportError(f"{project} has no runnable run under runs/")
        run = candidates[0].resolve()
    else:
        raise GameImportError(
            f"{project} has not passed acceptance (current stage "
            f"{state.get('current_stage')!r}). Finish the skill's acceptance stage, "
            "or pass --allow-incomplete to import the latest run anyway."
        )
    if not run.is_relative_to(project) or not (run / "index.html").is_file():
        raise GameImportError(f"Accepted run {run} is not a runnable directory inside {project}")
    return run


def _run_meta(run: Path) -> Mapping[str, Any]:
    for name in ("step3.json", "step2.json", "step1.json"):
        if (run / name).is_file():
            meta = _read_json(run / name)
            return meta if isinstance(meta, dict) else {}
    return {}


def _vendor(project: Path) -> dict[str, str]:
    manifest = _read_json(project / "package.json")
    dependencies = dict(manifest.get("dependencies") or {})
    three = dependencies.get("three")
    if three != THREE_VERSION:
        raise GameImportError(
            f"{project}/package.json pins three {three!r}; Eden games share {THREE_VERSION}."
        )
    unknown = sorted(set(dependencies) - {"three", "@dimforge/rapier3d-compat"})
    if unknown:
        raise GameImportError(f"Eden serves only three and rapier; the project also needs {unknown}")
    return dependencies


# An event line the runtime shows names its type in words: `race.checkpoint`
# becomes "race checkpoint". The game's own snapshot carries no sentence, so
# the sentence has to come from the registry's event_prompts table.
_EVENT_CALL = re.compile(
    r"""(?:\bemit|\bpushEvent|\b_emit)\(\s*([^,\n]{1,240}?)\s*[,)]"""
)
_EVENT_TYPE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)+$")
_EVENT_LITERAL = re.compile(r"""^['"`]([^'"`]+)['"`]$""")
_EVENT_TERNARY = re.compile(
    r"""^[^?:]*\?\s*['"`]([^'"`]+)['"`]\s*:\s*['"`]([^'"`]+)['"`]$"""
)


def _event_type_text(event_type: str) -> str:
    """One readable line for an event type, e.g. `round.won` -> "round won"."""
    return event_type.replace(".", " ").replace("_", " ")


def event_prompts_from_source(run: Path) -> dict[str, str]:
    """Sentences for the event types a game's scripts emit.

    Covers the two shapes the skill's games actually emit with: a string
    literal, and a ternary choosing between two. A type built any other way is
    left out rather than guessed, and the caller's own table wins over these.
    """
    found: list[str] = []
    for path in sorted(run.rglob("*")):
        if "evidence" in path.relative_to(run).parts or path.suffix not in {".js", ".mjs"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for call in _EVENT_CALL.finditer(text):
            argument = " ".join(call.group(1).split())
            literal = _EVENT_LITERAL.match(argument)
            candidates = [literal.group(1)] if literal else []
            ternary = _EVENT_TERNARY.match(argument)
            if ternary:
                candidates = [ternary.group(1), ternary.group(2)]
            found.extend(candidate for candidate in candidates if _EVENT_TYPE.match(candidate))
    return {event_type: _event_type_text(event_type) for event_type in sorted(set(found))}


def _restart_keys(run: Path) -> list[str]:

    """Keys the game's own strategy uses to restart, so overlay clicks can press them."""
    path = run / "strategy.json"
    if not path.is_file():
        return ["r"]
    keys = [
        str(action["key"]).lower()
        for action in _read_json(path).get("restart_actions") or ()
        if isinstance(action, dict) and action.get("type") == "tap" and action.get("key")
    ]
    return keys or ["r"]


def _copy_game(project: Path, run: Path, destination: Path) -> None:
    for path in sorted(run.rglob("*")):
        relative = path.relative_to(run)
        if path.is_symlink():
            raise GameImportError(f"Run contains a symlink, which is not copied: {path}")
        if not path.is_file() or "evidence" in relative.parts or path.name in _RUN_METADATA:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    assets = project / "assets"
    if assets.is_dir():
        shutil.copytree(
            assets,
            destination / "assets",
            ignore=shutil.ignore_patterns("receipts"),
            symlinks=False,
            dirs_exist_ok=True,
        )


def write_overlay(destination: Path, title: str) -> None:
    """Write the client HUD: a title chip and a result card with a restart button.

    The streamed frames are the bare canvas, so whatever HUD the game draws in
    its own DOM never reaches the client. This card is what lets a player
    dismiss an outcome, which is also what releases a held rollout.
    """
    hud = destination / "hud"
    hud.mkdir(parents=True, exist_ok=True)
    (hud / "overlay.js").write_text(
        # REUSE-IgnoreStart
        "// SPDX-License-Identifier: Apache-2.0\n"
        # REUSE-IgnoreEnd
        "// Generated by eden.apps.game2v.import_game for a standalone game.\n"
        f"const TITLE = {json.dumps(title, ensure_ascii=False)};\n"
        + _OVERLAY_JS,
        encoding="utf-8",
    )
    (hud / "overlay.css").write_text(_OVERLAY_CSS, encoding="utf-8")


_OVERLAY_JS = """const setText = (node, text) => { if (node && node.textContent !== text) node.textContent = text; };

export function mount(root) {
  root.innerHTML = `
    <div class="chip"><span class="title"></span><span class="state"></span></div>
    <div class="result" role="dialog" aria-modal="true" aria-labelledby="result-title" hidden>
      <div class="card">
        <h1 id="result-title"></h1>
        <p class="reason"></p>
        <button type="button" class="restart" data-game-action="restart">Play again</button>
        <div class="hint">or press R</div>
      </div>
    </div>`;
  const [chip, result] = root.children;
  setText(chip.querySelector('.title'), TITLE);
  return {
    update(state) {
      if (!state) return;
      const play = typeof state.state === 'string' ? state.state : '';
      setText(chip.querySelector('.state'), play);
      const terminal = play !== '' && play !== 'RUNNING' && play !== 'READY';
      if (result.hidden === terminal) result.hidden = !terminal;
      if (!terminal) return;
      setText(result.querySelector('h1'), play === 'WON' ? 'You won' : play === 'LOST' ? 'You lost' : play);
      setText(result.querySelector('.reason'), typeof state.outcomeReason === 'string' ? state.outcomeReason : '');
    },
  };
}
"""

_OVERLAY_CSS = """:host,.chip,.result{font-family:system-ui,sans-serif;color:#eef3f8}
.chip{position:absolute;left:16px;top:14px;display:flex;gap:10px;align-items:center;padding:6px 12px;border-radius:8px;background:#0d1520cc;font-size:12px;pointer-events:none}
.title{font-weight:700}
.state{color:#9fb4c8;letter-spacing:.08em;font-variant-numeric:tabular-nums}
.result{position:absolute;inset:0;display:grid;place-items:center;background:#05090f88;pointer-events:auto}
.result[hidden]{display:none}
.card{min-width:min(360px,86%);padding:28px 32px;border-radius:14px;background:#111b28f2;text-align:center}
.card h1{margin:0 0 10px;font-size:28px}
.reason{margin:0 0 20px;color:#b9c8d6;font-size:14px;line-height:1.6}
.restart{border:0;border-radius:8px;padding:10px 22px;font:600 14px system-ui,sans-serif;background:#7cc4ff;color:#0b1a29;cursor:pointer}
.restart:focus-visible{outline:2px solid #eef3f8;outline-offset:3px}
.hint{margin-top:12px;color:#7d90a3;font-size:11px}
"""


def registry_entry(
    project: Path,
    run: Path,
    slug: str,
    *,
    prompt: str | None = None,
    cover_prompt: str | None = None,
    event_prompts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the registry entry from the project's brief and accepted run card."""
    brief = _read_json(project / "brief.json")
    meta = _run_meta(run)
    scene = str(brief.get("scene", {}).get("description", "")).strip()
    static_prompt = (prompt or scene).strip()
    if not static_prompt:
        raise GameImportError("The brief has no scene.description; pass --prompt.")
    return {
        "slug": slug,
        "title": str(brief.get("title") or slug),
        "description": str(brief.get("core_verb") or brief.get("fantasy") or ""),
        "prompt": static_prompt,
        "cover_prompt": (cover_prompt or static_prompt).strip(),
        "cover": "cover.png",
        "entry": "index.html",
        "host": STANDALONE_HOST,
        "vendor": _vendor(project),
        "controls": [str(line) for line in meta.get("controls") or ()],
        "modules": [str(module) for module in meta.get("modules") or ()],
        "restart_keys": _restart_keys(run),
        "event_prompts": dict(event_prompts if event_prompts is not None else event_prompts_from_source(run)),
        "source": {"skill": "edengolem", "run": run.relative_to(project).as_posix()},
    }


def _write_registry(root: Path, registry: Mapping[str, Any]) -> None:
    text = json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False, suffix=".json") as handle:
        handle.write(text)
    os.replace(handle.name, root / "registry.json")


def capture_cover(slug: str, path: Path, root: Path) -> None:
    """Save the game's first rendered frame. Needs Playwright and Chromium."""
    from PIL import Image

    from .game_renderer import GameCaptureWorker
    from .game_timeline import GameTimeline

    worker = GameCaptureWorker(slug, games_root=root)
    try:
        chunk = worker.capture(GameTimeline(slug).begin_chunk(1, 16))
    finally:
        worker.close()
    Image.fromarray(chunk.frames[0].ui_frame).save(path)


def import_game(
    project: Path,
    slug: str,
    *,
    root: Path | None = None,
    cover: Path | None = None,
    prompt: str | None = None,
    cover_prompt: str | None = None,
    event_prompts: Mapping[str, str] | None = None,
    allow_incomplete: bool = False,
    replace: bool = False,
    capture=capture_cover,
    run: Path | None = None,
) -> dict[str, Any]:
    """Copy the project into ``<root>/<slug>/`` and register it. Returns the entry.

    ``run`` names the run directory to import instead of the accepted one. A
    revision of a passed game lives in a run of its own, which the acceptance
    record does not point at.
    """
    project = project.resolve()
    root = (root or games_root()).resolve()
    if not _SLUG.fullmatch(slug) or slug in _RESERVED_SLUGS:
        raise GameImportError(f"Slug must be lowercase words joined by '-', not a reserved name: {slug!r}")
    if cover is not None and not cover.is_file():
        raise GameImportError(f"Cover image {cover} does not exist")
    registry_path = root / "registry.json"
    original_registry = registry_path.read_text(encoding="utf-8") if registry_path.is_file() else None
    registry = json.loads(original_registry) if original_registry else {"games": []}
    existing = [game for game in registry["games"] if game.get("slug") == slug]
    destination = root / slug
    if (existing or destination.exists()) and not replace:
        raise GameImportError(f"Game {slug!r} already exists in {root}; pass --replace to overwrite it")
    if existing and existing[0].get("host") != STANDALONE_HOST:
        raise GameImportError(f"{slug!r} is a hand-ported game, not an import; refusing to replace it")

    if run is None:
        run = accepted_run(project, allow_incomplete=allow_incomplete)
    else:
        run = run.resolve()
        if not run.is_relative_to(project) or not (run / "index.html").is_file():
            raise GameImportError(f"Run {run} is not a runnable directory inside {project}")
    entry = registry_entry(
        project, run, slug,
        prompt=prompt, cover_prompt=cover_prompt, event_prompts=event_prompts,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{slug}-", dir=root))
    previous = root / f".{slug}.previous"
    moved_aside = installed = False
    try:
        _copy_game(project, run, staging)
        write_overlay(staging, entry["title"])
        if cover is not None:
            shutil.copy2(cover, staging / "cover.png")
        if destination.exists():
            shutil.rmtree(previous, ignore_errors=True)
            destination.rename(previous)
            moved_aside = True
        staging.rename(destination)
        installed = True
        registry["games"] = [game for game in registry["games"] if game.get("slug") != slug] + [entry]
        _write_registry(root, registry)
        if cover is None:
            capture(slug, destination / "cover.png", root)
    except BaseException:
        # Put back exactly what was there: the registry text and any replaced directory.
        shutil.rmtree(staging, ignore_errors=True)
        if installed:
            shutil.rmtree(destination, ignore_errors=True)
        if moved_aside:
            previous.rename(destination)
        if original_registry is None:
            registry_path.unlink(missing_ok=True)
        else:
            registry_path.write_text(original_registry, encoding="utf-8")
        raise
    shutil.rmtree(previous, ignore_errors=True)
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eden.apps.game2v.import_game",
        description="Register a GamePrototypeTurbo project as an Eden game.",
    )
    parser.add_argument("project", type=Path, help="Skill output directory (holds .state.json).")
    parser.add_argument("--slug", required=True, help="Registry slug, e.g. star-course.")
    parser.add_argument("--games-root", type=Path, default=None, help="Defaults to EDEN_GAMES_ROOT or games/.")
    parser.add_argument("--cover", type=Path, default=None,
                        help="First-frame image. Without it the game's own first frame is captured.")
    parser.add_argument("--prompt", default=None, help="Static scene prompt. Defaults to the brief's scene.description.")
    parser.add_argument("--cover-prompt", default=None, help="Description of the cover. Defaults to the prompt.")
    parser.add_argument("--event-prompts", type=Path, default=None,
                        help="JSON object mapping game event types to prompt sentences.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Import the latest run even if acceptance has not passed.")
    parser.add_argument("--replace", action="store_true", help="Overwrite an earlier import of this slug.")
    args = parser.parse_args(argv)
    event_prompts = None
    if args.event_prompts is not None:
        event_prompts = _read_json(args.event_prompts)
        if not isinstance(event_prompts, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in event_prompts.items()
        ):
            parser.error("--event-prompts must be a JSON object of strings")
    try:
        entry = import_game(
            args.project, args.slug,
            root=args.games_root, cover=args.cover, prompt=args.prompt,
            cover_prompt=args.cover_prompt, event_prompts=event_prompts,
            allow_incomplete=args.allow_incomplete, replace=args.replace,
        )
    except GameImportError as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Registered {entry['slug']!r} from {entry['source']['run']}. "
          f"Run it with EDEN_GAME={entry['slug']} bash start.sh <model>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
