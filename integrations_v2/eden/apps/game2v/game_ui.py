# SPDX-License-Identifier: Apache-2.0
from flashdreams.api_v2.loop import IUILoop, ModelInferenceState
from flashdreams.runtime_v2.blit_model_output_to_screen_loop import _frame_to_layout
from flashdreams.runtime_v2.step_result import StepResult


class Game2VUILoop(IUILoop[None]):
    def _initialize_loop_state(self):
        self._last_count = 0
        self._prompt_chunk = None

    def step(self, step_index, events):
        frames = self.presented_model_frames()
        count = self._presentation_manager.presented_frame_count
        if not frames or count == self._last_count:
            return None
        self._last_count = count
        meta = self.presented_model_metadata(1) if len(frames) > 1 else None
        messages = self._prompt_messages()
        if meta is not None:
            messages.append(meta)
        return StepResult(
            step_index=step_index,
            output=_frame_to_layout(frames[0], self.output_layout),
            frame_count=1,
            output_layout=self.output_layout,
            auxiliary_frames={'game': frames[1]} if len(frames) > 1 else {},
            client_messages=tuple(messages),
        )

    def _prompt_messages(self):
        """Send a chunk's event lines once, on its first presented frame.

        The panel shows exactly one chunk's set at a time: the next chunk's
        message replaces it wholesale, so nothing accumulates and the box is
        never blanked in between. The lines therefore stay up for the whole
        chunk they belong to instead of clearing on its last frame.
        """
        prompt = next(
            (
                message
                for message in self._presentation_manager.presented_client_messages()
                if message.get("type") == "prompt"
            ),
            None,
        )
        if prompt is None:
            return []
        chunk = self._presentation_manager.presented_chunk_count
        if chunk == self._prompt_chunk:
            return []
        self._prompt_chunk = chunk
        return [prompt]

    def is_finished(self):
        return (self.model_inference_state is ModelInferenceState.FINISHED
                and not self._presentation_manager.has_pending_frames()
                and self._last_count == self._presentation_manager.presented_frame_count)

    def reset(self):
        self._last_count = 0
        self._prompt_chunk = None
