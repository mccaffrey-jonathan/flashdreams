# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk prompt composition for Game2V.

The static scene description comes from the game registry. Event lines come
from the chunk simulated while the previous one was generated, last for a
configured number of chunks, and the text sent to the client is that same
string.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from eden.apps.game2v.game_renderer import game_static_prompt
from eden.apps.game2v.game_timeline import GameFramePacket, GameTimeline
from eden.apps.game2v.session import (
    Game2VModelState,
    Game2VSessionConfig,
    _chunk_event_lines,
    _compose_chunk_prompt,
)

pytestmark = pytest.mark.ci_cpu


def _chunk(lines_by_frame):
    """One chunk whose frames publish the given ``(id, type, text)`` events."""
    timeline = GameTimeline('freeze')
    packet = timeline.begin_chunk(len(lines_by_frame), 16)
    frames = []
    for frame, lines in zip(packet.frames, lines_by_frame):
        frames.append(replace(frame, ui_state={
            'events': [
                {'id': event_id, 'type': event_type, 'prompt': text}
                for event_id, event_type, text in lines
            ],
        }))
    return replace(packet, frames=tuple(frames))


def _state(*, rank='0', lifetime=1, last_chunk=None):
    from flashdreams.runtime_v2.session_desc import SessionDesc

    config = SimpleNamespace(
        game_slug='freeze',
        prompt_event_chunks=lifetime,
    )
    state = Game2VModelState(
        pipeline=SimpleNamespace(get_num_output_frames=lambda step_index: 9),
        session_desc=SessionDesc(frames_per_second_for_step=16),
        config=config,
        keyboard_resampler=SimpleNamespace(
            input_timeline=SimpleNamespace(),
            keyboard_track=SimpleNamespace(),
        ),
    )
    # Rank zero is the only rank that renders, so it is the only one composing.
    state.game_capture = object() if rank == '0' else None
    state.last_game_chunk = last_chunk
    return state


def test_registry_static_prompt(tmp_path):
    registry = {
        'games': [
            {'slug': 'freeze', 'prompt': '  A quiet courtyard.  '},
            {'slug': 'tracer'},
        ],
    }
    (tmp_path / 'registry.json').write_text(json.dumps(registry))

    assert game_static_prompt('freeze', games_root=tmp_path) == 'A quiet courtyard.'
    assert game_static_prompt('tracer', games_root=tmp_path) == ''
    assert game_static_prompt('missing', games_root=tmp_path) == ''
    assert game_static_prompt('freeze', games_root=tmp_path / 'absent') == ''


def test_shipped_registries_declare_a_static_prompt():
    """Both vendored games describe their scene, which is the prompt base."""
    for slug in ('ember-den', 'jade-duel'):
        assert game_static_prompt(slug)


def test_event_lines_union_every_frame_and_keep_the_earliest():
    """A line that scrolled out of the last frame's log still counts."""
    chunk = _chunk([
        [(1, 'keeper.moved', 'The keeper steps left.'), (2, 'crate.pushed', 'A crate slides.')],
        # The game log is capped, so the first line is already gone here, and
        # the same move said differently is a second line rather than a repeat.
        [(2, 'crate.pushed', 'A crate slides.'), (3, 'keeper.moved', 'The keeper steps right.')],
    ])

    assert _chunk_event_lines(chunk) == [
        ('keeper.moved', 'The keeper steps left.'),
        ('crate.pushed', 'A crate slides.'),
        ('keeper.moved', 'The keeper steps right.'),
    ]


def test_event_lines_skip_entries_without_a_prompt():
    chunk = _chunk([
        [(1, 'keeper.moved', 'The keeper steps left.'), (2, 'ignored', None), (3, '', 'no type')],
        [(4, 'crate.pushed', '   ')],
    ])

    assert _chunk_event_lines(chunk) == [('keeper.moved', 'The keeper steps left.')]
    assert _chunk_event_lines(None) == []
    bare = GameFramePacket(
        game_frame_id=1, logical_time_s=0.0, simulation_tick=1,
        input_sequence=0, game_slug='freeze', ui_state={'moves': 1},
    )
    assert _chunk_event_lines(replace(_chunk([[]]), frames=(bare,))) == []


def test_prompt_is_static_until_a_chunk_has_been_simulated(monkeypatch, tmp_path):
    monkeypatch.setenv('EDEN_GAMES_ROOT', str(tmp_path))
    (tmp_path / 'registry.json').write_text(json.dumps(
        {'games': [{'slug': 'freeze', 'prompt': 'A quiet courtyard.'}]}
    ))
    state = _state()

    assert _compose_chunk_prompt(state) == ('A quiet courtyard.', ())
    # Resolved once; a registry change mid-rollout does not move it.
    (tmp_path / 'registry.json').write_text(json.dumps(
        {'games': [{'slug': 'freeze', 'prompt': 'Changed.'}]}
    ))
    assert _compose_chunk_prompt(state) == ('A quiet courtyard.', ())


def test_event_lines_last_the_configured_number_of_chunks(monkeypatch, tmp_path):
    monkeypatch.setenv('EDEN_GAMES_ROOT', str(tmp_path))
    (tmp_path / 'registry.json').write_text(json.dumps(
        {'games': [{'slug': 'freeze', 'prompt': 'A quiet courtyard.'}]}
    ))
    state = _state(lifetime=2, last_chunk=_chunk([
        [(1, 'keeper.moved', 'The keeper steps left.')],
    ]))

    # The chunk just collected conditions the one being generated. The event
    # lines come back on their own so the client can drop them per chunk.
    first = _compose_chunk_prompt(state)
    assert first == ('A quiet courtyard.', ('The keeper steps left.',))
    assert state.active_prompt == 'A quiet courtyard. The keeper steps left.'

    # Still inside its two-chunk lifetime, ahead of a newer line.
    state.last_game_chunk = _chunk([[(2, 'crate.pushed', 'A crate slides.')]])
    assert _compose_chunk_prompt(state) == (
        'A quiet courtyard.', ('The keeper steps left.', 'A crate slides.')
    )

    # The first line has now covered its two chunks and drops out.
    state.last_game_chunk = _chunk([[]])
    assert _compose_chunk_prompt(state) == ('A quiet courtyard.', ('A crate slides.',))
    assert _compose_chunk_prompt(state) == ('A quiet courtyard.', ())


def test_recurring_sentence_is_restated_by_its_newest_chunk(monkeypatch, tmp_path):
    """The same sentence refreshes its lifetime; a different one is added."""
    monkeypatch.setenv('EDEN_GAMES_ROOT', str(tmp_path))
    (tmp_path / 'registry.json').write_text(json.dumps({'games': [{'slug': 'freeze', 'prompt': ''}]}))
    state = _state(lifetime=3, last_chunk=_chunk([
        [(1, 'lane.changed', 'The runner shifts into the left lane.')],
    ]))
    _compose_chunk_prompt(state)

    state.last_game_chunk = _chunk([[
        (2, 'lane.changed', 'The runner shifts into the left lane.'),
        (3, 'lane.changed', 'The runner shifts into the right lane.'),
    ]])
    assert _compose_chunk_prompt(state) == (
        '',
        (
            'The runner shifts into the left lane.',
            'The runner shifts into the right lane.',
        ),
    )
    # One entry per sentence, so the repeat refreshed the first rather than
    # stacking a second copy of it.
    assert [event_type for _, event_type, _ in state.prompt_events] == [
        'lane.changed', 'lane.changed',
    ]


def test_other_ranks_receive_the_composed_prompt(monkeypatch):
    """Every rank must condition on the same text, and only rank zero has frames."""
    state = _state(rank='1')
    broadcast = SimpleNamespace(payload=None)

    def broadcast_object_list(payload, src):
        assert src == 0
        # What rank zero composed arrives in the same list.
        payload[0] = ('A quiet courtyard.', ('The keeper steps left.',))

    monkeypatch.setattr(
        'eden.apps.game2v.session.is_distributed_initialized', lambda: True
    )
    monkeypatch.setattr(
        'flashdreams.runtime_v2.distributed_sync.dist.broadcast_object_list',
        broadcast_object_list,
    )

    assert _compose_chunk_prompt(state) == (
        'A quiet courtyard.', ('The keeper steps left.',)
    )
    assert state.active_prompt == 'A quiet courtyard. The keeper steps left.'
    assert state.prompt_events == []
    assert broadcast.payload is None


def test_other_ranks_skip_the_opening_encode_with_rank_zero(monkeypatch):
    """A peer must make rank zero's skip decision, or the collectives diverge.

    Only rank zero resolves the registry text. A peer that compared the
    broadcast opening prompt against its own empty copy would join an
    embeddings broadcast rank zero never makes, and the next collective hangs.
    """
    monkeypatch.setenv('RANK', '1')
    state = _state(rank='1', last_chunk=_chunk([[]]))
    state.config.dynamic_prompt = True
    state.pipeline = SimpleNamespace(
        encode_text=lambda text: pytest.fail('only rank zero encodes'),
        replace_text_embeddings=lambda cache, embeddings: None,
    )
    state.cache = object()
    broadcasts: list[int] = []

    def broadcast_object_list(payload, src):
        payload[0] = ('A quiet courtyard.', ())

    monkeypatch.setattr(
        'eden.apps.game2v.session.is_distributed_initialized', lambda: True
    )
    monkeypatch.setattr(
        'flashdreams.runtime_v2.distributed_sync.dist.broadcast_object_list',
        broadcast_object_list,
    )
    monkeypatch.setattr(
        'torch.distributed.broadcast', lambda tensor, src: broadcasts.append(src)
    )

    _compose_chunk_prompt(state)
    assert broadcasts == []
    assert state.encoded_prompt == 'A quiet courtyard.'


def test_dynamic_prompt_encodes_only_when_the_text_changes(monkeypatch, tmp_path):
    """The text encoder runs on a changed prompt and is skipped on a repeat.

    Rank zero encodes; the embeddings are broadcast and every rank swaps them
    into its own cache. A quiet stretch of the game composes the same text
    twice, and the second one must not encode again.
    """
    monkeypatch.setenv('EDEN_GAMES_ROOT', str(tmp_path))
    (tmp_path / 'registry.json').write_text(json.dumps(
        {'games': [{'slug': 'freeze', 'prompt': 'A quiet courtyard.'}]}
    ))
    encoded: list[str] = []
    replaced: list[str] = []

    class Pipeline:
        def encode_text(self, text):
            encoded.append(text[0])
            return text[0]

        def replace_text_embeddings(self, cache, embeddings):
            replaced.append(embeddings)

    state = _state(last_chunk=_chunk([[]]))
    state.config.dynamic_prompt = True
    state.pipeline = Pipeline()
    state.cache = object()

    _compose_chunk_prompt(state)
    # The first chunk carries the static prompt the cache was built from, so
    # nothing is encoded until the text actually changes.
    assert encoded == []

    state.last_game_chunk = _chunk([[(1, 'keeper.moved', 'The keeper steps left.')]])
    _compose_chunk_prompt(state)
    assert encoded == ['A quiet courtyard. The keeper steps left.']

    # The same sentence again only refreshes its lifetime; the text is unchanged.
    state.last_game_chunk = _chunk([[(2, 'keeper.moved', 'The keeper steps left.')]])
    _compose_chunk_prompt(state)
    assert encoded == ['A quiet courtyard. The keeper steps left.']
    assert replaced == encoded


def test_dynamic_prompt_is_rejected_without_a_text_encoder(monkeypatch, tmp_path):
    monkeypatch.setenv('EDEN_GAMES_ROOT', str(tmp_path))
    (tmp_path / 'registry.json').write_text(json.dumps(
        {'games': [{'slug': 'freeze', 'prompt': 'A quiet courtyard.'}]}
    ))
    state = _state(last_chunk=_chunk([[]]))
    state.config.dynamic_prompt = True

    _compose_chunk_prompt(state)
    state.last_game_chunk = _chunk([[(1, 'x', 'Something happens.')]])
    with pytest.raises(RuntimeError, match='encode_text'):
        _compose_chunk_prompt(state)


def test_prompt_event_chunks_must_be_a_positive_integer():
    config = dict(
        conditioning=SimpleNamespace(),
        total_blocks=1,
        device='cpu',
        first_frame_dtype=None,
        first_frame_interpolation='cubic',
        warmup_blocks=0,
    )
    with pytest.raises(ValueError, match='prompt_event_chunks'):
        Game2VSessionConfig(**config, prompt_event_chunks=0)
    with pytest.raises(ValueError, match='prompt_event_chunks'):
        Game2VSessionConfig(**config, prompt_event_chunks=True)


def test_dynamic_prompt_keeps_the_text_encoder_on_rank_zero(monkeypatch):
    """The first per-chunk prompt must not reload UMT5 mid-rollout.

    ``initialize_cache`` frees the text encoder by default. Rank zero, the only
    rank that re-encodes, asks it to stay loaded; the other ranks free theirs,
    and a static-prompt run keeps the default.
    """
    from eden.apps.game2v.session import _text_encoder_kwargs

    class Pipeline:
        def initialize_cache(self, text, image=None, *, release_oneshot_encoders=True):
            del text, image, release_oneshot_encoders

    state = _state()
    state.pipeline = Pipeline()

    state.config.dynamic_prompt = False
    assert _text_encoder_kwargs(state) == {}

    state.config.dynamic_prompt = True
    monkeypatch.setenv('RANK', '0')
    assert _text_encoder_kwargs(state) == {'release_oneshot_encoders': False}
    monkeypatch.setenv('RANK', '1')
    assert _text_encoder_kwargs(state) == {}

    # A pipeline that cannot be asked keeps its own behavior.
    monkeypatch.setenv('RANK', '0')
    state.pipeline = SimpleNamespace(initialize_cache=lambda text, image=None: None)
    assert _text_encoder_kwargs(state) == {}
