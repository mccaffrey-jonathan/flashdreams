# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout pause while a blocking client card is on screen."""

import threading
from types import SimpleNamespace

import pytest

from eden.apps.game2v.game_hud import is_dialog_open
from eden.apps.game2v.game_timeline import GameChunkPacket, GameFramePacket
from eden.apps.game2v.session import (
    _await_dialog_dismissed,
    _await_start,
    _hold_for_dialog,
    request_game_switch,
)
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc

pytestmark = pytest.mark.ci_cpu


def _frame(ui_state):
    return GameFramePacket(
        game_frame_id=1,
        logical_time_s=0.0,
        simulation_tick=4,
        input_sequence=0,
        game_slug='freeze',
        ui_state=ui_state,
    )


def _chunk(*ui_states):
    return GameChunkPacket(
        chunk_id=1,
        start_time_s=0.0,
        end_time_s=0.25,
        input_sequence_start=0,
        input_sequence_end=0,
        simulation_tick_start=0,
        simulation_tick_end=4,
        frames=tuple(_frame(state) for state in ui_states),
    )


def _state(chunk, *, mode=PresentationMode.CONTINUOUS, game_capture=None):
    return SimpleNamespace(
        last_game_chunk=chunk,
        session_desc=SessionDesc(presentation_mode=mode),
        dialog_held=False,
        game_capture=game_capture,
        event_buffer=None,
    )


@pytest.mark.parametrize(
    'state, expected',
    [
        ({'state': 'RUNNING', 'helpOpen': False}, False),
        ({'state': 'WON'}, True),
        ({'state': 'LOST'}, True),
        ({'state': 'RUNNING', 'helpOpen': True}, True),
        # READY is a game waiting on its starting line. Generation must keep
        # running there -- letting the frame rate settle before play starts is
        # the entire point of the state.
        ({'state': 'READY'}, False),
        # A card raised while still on the starting line does pause.
        ({'state': 'READY', 'helpOpen': True}, True),
        # The ported games' pause card holds generation like a help card.
        ({'state': 'RUNNING', 'paused': True}, True),
        ({'state': 'RUNNING', 'paused': False}, False),
        # A game that publishes neither convention never pauses the rollout.
        ({'score': 3}, False),
        ({'state': ''}, False),
        (None, False),
    ],
)
def test_is_dialog_open_reads_both_card_conventions(state, expected):
    assert is_dialog_open(state) is expected


def test_hold_for_dialog_inspects_the_last_frame_only():
    # The card has to be up when the chunk ends; a card opened and dismissed
    # mid-chunk leaves nothing for the client to close.
    held = _chunk({'state': 'RUNNING'}, {'state': 'WON'})
    passed = _chunk({'state': 'WON'}, {'state': 'RUNNING'})
    assert _hold_for_dialog(_state(held)) is True
    assert _hold_for_dialog(_state(passed)) is False


def test_hold_for_dialog_sets_the_state_flag():
    state = _state(_chunk({'state': 'WON'}))
    assert _hold_for_dialog(state) is True
    assert state.dialog_held is True


def test_hold_for_dialog_skips_file_renders():
    # An mp4 render has nobody to dismiss a card, so holding would never resume.
    state = _state(_chunk({'state': 'WON'}), mode=PresentationMode.ON_DEMAND)
    assert _hold_for_dialog(state) is False
    assert state.dialog_held is False


def test_hold_for_dialog_tolerates_a_missing_chunk():
    assert _hold_for_dialog(_state(None)) is False
    assert _hold_for_dialog(_state(_chunk())) is False


class _FakeCapture:
    """Capture worker that reports a card for the first ``open_polls`` probes."""

    def __init__(self, open_polls):
        self.open_polls = open_polls
        self.probes = 0
        self.drained = 0

    def probe(self, *, drain_events=None):
        self.probes += 1
        if drain_events is not None:
            drain_events()
            self.drained += 1
        card_open = self.probes <= self.open_polls
        return {'tick': 4, 'state': {'state': 'WON' if card_open else 'RUNNING'}}


def test_await_returns_immediately_when_nothing_is_held():
    capture = _FakeCapture(open_polls=5)
    state = _state(_chunk({'state': 'RUNNING'}), game_capture=capture)
    assert _await_dialog_dismissed(state, threading.Event()) == 0.0
    assert capture.probes == 0


def test_await_reports_the_time_it_held_the_rollout(monkeypatch):
    # The caller subtracts this from the step time the presentation cadence is
    # derived from. Reported as zero, the chunk after a pause would look like it
    # took seconds to generate and playback would crawl.
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.0)
    # One reading when the wait starts, one when it ends.
    clock = iter([100.0, 104.5])
    monkeypatch.setattr(
        'eden.apps.game2v.session.time.perf_counter', lambda: next(clock)
    )
    capture = _FakeCapture(open_polls=2)
    state = _state(_chunk({'state': 'WON'}), game_capture=capture)
    assert _hold_for_dialog(state) is True
    assert _await_dialog_dismissed(state, threading.Event()) == pytest.approx(4.5)


def test_await_polls_until_the_card_is_dismissed(monkeypatch):
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.0)
    capture = _FakeCapture(open_polls=3)
    state = _state(_chunk({'state': 'WON'}), game_capture=capture)
    assert _hold_for_dialog(state) is True
    _await_dialog_dismissed(state, threading.Event())
    # Three probes saw the card, the fourth saw it gone.
    assert capture.probes == 4
    assert capture.drained == 4
    assert state.dialog_held is False


def test_await_gives_up_when_the_run_is_shutting_down(monkeypatch):
    # A closed tab must not strand the model thread behind a card that nobody
    # is left to dismiss.
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.0)
    capture = _FakeCapture(open_polls=10_000)
    state = _state(_chunk({'state': 'WON'}), game_capture=capture)
    assert _hold_for_dialog(state) is True
    shutdown = threading.Event()
    shutdown.set()
    _await_dialog_dismissed(state, shutdown)
    # One probe still runs, so the decision is taken on the same code path as
    # a live poll; the loop then leaves rather than waiting again.
    assert capture.probes == 1
    assert state.dialog_held is False


def test_await_gives_up_when_a_game_switch_is_requested(monkeypatch):
    # The workbench switches games while a finished round's card is still up and
    # its client has left; the hold must yield so the switch can be applied.
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.0)
    capture = _FakeCapture(open_polls=10_000)
    state = _state(_chunk({'state': 'WON'}), game_capture=capture)
    assert _hold_for_dialog(state) is True
    state.pending_game_slug = 'desert-crossfire'
    _await_dialog_dismissed(state, threading.Event())
    assert capture.probes == 1
    assert state.dialog_held is False


def test_start_gate_waits_for_the_first_switch(monkeypatch):
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.01)
    monkeypatch.setattr('eden.apps.game2v.session.is_distributed_initialized', lambda: False)
    state = SimpleNamespace(
        start_requested=False, pending_game_slug=None,
        config=SimpleNamespace(game_slug='freeze'),
    )
    done = threading.Event()
    thread = threading.Thread(target=lambda: (_await_start(state, threading.Event()), done.set()))
    thread.start()
    assert not done.wait(0.1)
    request_game_switch(state, 'desert-crossfire')
    assert done.wait(2.0)
    thread.join()
    assert state.pending_game_slug == 'desert-crossfire'


def test_start_gate_releases_on_switch_to_the_current_game(monkeypatch):
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.0)
    monkeypatch.setattr('eden.apps.game2v.session.is_distributed_initialized', lambda: False)
    state = SimpleNamespace(
        start_requested=False, pending_game_slug=None, pending_home=False,
        pending_game_restart=False,
        config=SimpleNamespace(game_slug='freeze'),
    )
    request_game_switch(state, 'freeze')
    assert state.pending_game_slug is None
    _await_start(state, threading.Event())


def test_await_gives_up_when_home_is_requested(monkeypatch):
    # Returning home has to reach the gate, which a held step would never get
    # to: the card that paused it belongs to a client that may already be gone.
    monkeypatch.setattr('eden.apps.game2v.session._DIALOG_POLL_SECONDS', 0.0)
    capture = _FakeCapture(open_polls=10_000)
    state = _state(_chunk({'state': 'WON'}), game_capture=capture)
    assert _hold_for_dialog(state) is True
    state.pending_home = True
    _await_dialog_dismissed(state, threading.Event())
    assert capture.probes == 1
    assert state.dialog_held is False


def _round_chunk(*round_ids):
    return _chunk(*({'state': 'RUNNING', 'roundId': round_id} for round_id in round_ids))


def _rollout_state(chunk, *, round_id=None, game_capture=object()):
    from eden.apps.game2v.controls import CameraPoseIntegrator
    from flashdreams.runtime_v2.input_timeline import RealtimeInputTimeline
    from flashdreams.runtime_v2.keyboard_input import KeyboardStateTrack

    state = _state(chunk, game_capture=game_capture)
    state.game_round_id = round_id
    state.cache = object()
    state.ar_index = 4
    state.blocks_generated = 4
    state.frames_generated = 48
    state.dialog_held = True
    state.pending_game_chunk = None
    state.pending_game_frames = __import__('collections').deque([object(), object()])
    state.input_timeline = RealtimeInputTimeline(samples_per_second=16)
    state.input_timeline.next_window(8)
    state.keyboard_track = KeyboardStateTrack()
    state.pose_integrator = CameraPoseIntegrator()
    state.steady_started_at = 10.0
    state.steady_frames_generated = 48
    state.steady_held_s = 1.5
    state.postprocess_stream = None
    state.comparison_pending_generated_frames = None
    state._recent_model_frame_rate_tracker = SimpleNamespace(reset=lambda: None)
    # A restart opens the new rollout on the static prompt, so the lines the
    # previous round contributed are dropped with the cache.
    state.prompt_events = [(2, 'keeper.moved', 'The keeper steps left.')]
    state.active_prompt = 'The keeper steps left.'
    return state


def test_restart_is_silent_until_a_round_has_been_seen():
    from eden.apps.game2v.session import _restart_rollout

    # The first rendered chunk establishes the round; there is no rollout to rewind.
    state = _rollout_state(_round_chunk(1), round_id=None)
    assert _restart_rollout(state) is False
    assert state.game_round_id == 1
    assert state.cache is not None
    assert state.ar_index == 4


def test_restart_rewinds_the_rollout_when_the_round_changes():
    from eden.apps.game2v.session import _restart_rollout

    # The chunk straddles the reset: it opens in round 1 and ends in round 2.
    state = _rollout_state(_round_chunk(1, 2), round_id=1)
    pending = list(state.pending_game_frames)
    assert _restart_rollout(state) is True
    assert state.game_round_id == 2
    # The cache is dropped so the caller rebuilds it from the anchor frame, and
    # generation resumes at autoregressive index zero.
    assert state.cache is None
    assert state.ar_index == 0
    assert state.blocks_generated == 0
    assert state.frames_generated == 0
    # Frames rendered for the old round must not be paired with the new output.
    assert list(state.pending_game_frames) == []
    assert pending
    # A card that the restart closed is not still a pause.
    assert state.dialog_held is False
    # The camera clock restarts with the rollout rather than continuing.
    assert state.input_timeline.next_window_start_s == 0.0
    assert state.steady_started_at is None
    assert state.steady_held_s == 0.0
    # The chunk that straddles the reset belongs to neither round, so its
    # event lines do not open the new one.
    assert state.prompt_events == []
    assert state.active_prompt == ''


def test_restart_ignores_a_chunk_that_stays_in_its_round():
    from eden.apps.game2v.session import _restart_rollout

    state = _rollout_state(_round_chunk(3, 3), round_id=3)
    assert _restart_rollout(state) is False
    assert state.cache is not None
    assert state.ar_index == 4
    assert state.blocks_generated == 4


def test_restart_is_invisible_to_a_rank_that_does_not_render():
    from eden.apps.game2v.session import _restart_rollout

    # Only rank zero renders, so only it can see a round change. A peer with no
    # capture worker must not rewind on its own; the broadcast carries the answer.
    state = _rollout_state(_round_chunk(1, 2), round_id=1, game_capture=None)
    assert _restart_rollout(state) is False
    assert state.cache is not None
    assert state.game_round_id == 1


def test_restart_tolerates_a_game_that_publishes_no_round():
    from eden.apps.game2v.session import _restart_rollout

    state = _rollout_state(_chunk({'state': 'RUNNING'}), round_id=1)
    assert _restart_rollout(state) is False
    # Nothing was published, so the remembered round is left untouched.
    assert state.game_round_id == 1
    assert state.cache is not None
