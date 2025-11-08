from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image

from lluma_agent.cinematics import (
    CinematicDetector,
    CinematicDetectionResult,
    CinematicKind,
    CinematicObservation,
    PlaybackState,
)
from lluma_agent.coordinator import CinematicControlState, GameLoopCoordinator
from lluma_os.config import load_configs


CAPTURE_DIR = Path(__file__).resolve().parent / "data" / "cinematics"


@pytest.fixture(scope="module")
def capture_config():
    _, capture_cfg, _ = load_configs(Path("config.yaml"))
    return capture_cfg


def _load_rgb_image(name: str) -> Image.Image:
    return Image.open(CAPTURE_DIR / name).convert("RGB")


def test_pin_present_breaks_out_of_fullscreen(capture_config):
    detector = CinematicDetector(capture_config)
    result = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_non_cutscene_1.png"),
            menu_is_usable=True,
        )
    )
    assert result.kind is CinematicKind.NONE
    assert result.pin_present is True
    assert result.skip_hint_primary is False
    assert result.skip_hint_tabs is False
    assert result.playback is PlaybackState.UNKNOWN


def test_fullscreen_cutscene_requires_menu_streak(capture_config):
    detector = CinematicDetector(capture_config)
    first = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_1.png"),
            menu_is_usable=False,
        )
    )
    assert first.kind is CinematicKind.NONE
    second = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_2.png"),
            menu_is_usable=False,
        )
    )
    assert second.kind is CinematicKind.FULLSCREEN
    assert second.playback is PlaybackState.PLAYING
    third = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_3.png"),
            menu_is_usable=False,
        )
    )
    assert third.kind is CinematicKind.FULLSCREEN
    assert third.playback is PlaybackState.PLAYING


def test_fullscreen_story_dialogue_detects_pause(capture_config):
    detector = CinematicDetector(capture_config)
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_story_dialogue_1.png"),
            menu_is_usable=False,
        )
    )
    paused = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_story_dialogue_2.png"),
            menu_is_usable=False,
        )
    )
    assert paused.kind is CinematicKind.FULLSCREEN
    assert paused.playback is PlaybackState.PAUSED
    assert paused.skip_hint_tabs is True


def test_fullscreen_skip_overlay_flagged(capture_config):
    detector = CinematicDetector(capture_config)
    # Prime the detector so the menu streak is satisfied.
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_1.png"),
            menu_is_usable=False,
        )
    )
    skip_frame = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_paused.png"),
            menu_is_usable=False,
        )
    )
    assert skip_frame.kind is CinematicKind.FULLSCREEN
    assert skip_frame.skip_hint_tabs is True


def test_dimmed_pin_prevents_false_fullscreen_detection(capture_config):
    detector = CinematicDetector(capture_config)
    frame = _load_rgb_image("fullscreen_pin_dimmed.png")
    for _ in range(3):
        result = detector.observe(
            CinematicObservation(
                image=frame,
                menu_is_usable=False,
            )
        )
        assert result.kind is CinematicKind.NONE
        assert result.pin_present is True


@pytest.mark.parametrize(
    "filename",
    [
        "primary_cutscene_skip.png",
        "primary_cutscene_skip_white.png",
    ],
)
def test_primary_cutscene_skip_chip_detection(capture_config, filename):
    detector = CinematicDetector(capture_config)
    result = detector.observe(
        CinematicObservation(
            image=_load_rgb_image(filename),
            menu_is_usable=False,
        )
    )
    assert result.kind is CinematicKind.PRIMARY
    assert result.skip_hint_primary is True
    assert result.playback in {PlaybackState.PAUSED, PlaybackState.PLAYING}


def test_primary_modal_highlight_does_not_trigger_cutscene(capture_config):
    detector = CinematicDetector(capture_config)
    frame = _load_rgb_image("primary_story_modal_no_cinematic.png")
    result = detector.observe(
        CinematicObservation(
            image=frame,
            menu_is_usable=False,
        )
    )
    assert result.kind is CinematicKind.NONE
    assert result.skip_hint_primary is False


def test_coordinator_holds_for_post_cinematic_buffer() -> None:
    coordinator = GameLoopCoordinator.__new__(GameLoopCoordinator)
    coordinator._logger = logging.getLogger("test_coordinator_buffer")
    coordinator._cinematic_state = CinematicControlState()
    coordinator._cinematic_min_low_frames = 2
    coordinator._cinematic_max_hold_turns = 12
    coordinator._cinematic_release_buffer_turns = 2
    coordinator._cinematic_release_cooldown = 0
    coordinator._turn_counter = 25

    active_detection = CinematicDetectionResult(
        kind=CinematicKind.PRIMARY,
        playback=PlaybackState.PLAYING,
        diff_score=0.2,
        skip_hint_primary=True,
        skip_hint_tabs=False,
        skip_label_hint=False,
        menu_unusable_streak=1,
        pin_present=False,
        pin_bright_ratio=0.0,
        primary_diff_score=0.2,
    )
    blocked, info = coordinator._update_cinematic_state(active_detection)
    assert blocked is True
    assert info["kind"] == "primary"

    coordinator._turn_counter += 1
    release_detection = CinematicDetectionResult(
        kind=CinematicKind.PRIMARY,
        playback=PlaybackState.PAUSED,
        diff_score=0.0,
        skip_hint_primary=False,
        skip_hint_tabs=False,
        skip_label_hint=False,
        menu_unusable_streak=2,
        pin_present=False,
        pin_bright_ratio=0.0,
        primary_diff_score=0.0,
    )
    blocked, info = coordinator._update_cinematic_state(release_detection)
    assert blocked is True
    assert coordinator._cinematic_release_cooldown == 1
    assert info["buffer_turns_remaining"] == 1

    idle_detection = CinematicDetectionResult(
        kind=CinematicKind.NONE,
        playback=PlaybackState.UNKNOWN,
        diff_score=0.0,
        skip_hint_primary=False,
        skip_hint_tabs=False,
        skip_label_hint=False,
        menu_unusable_streak=0,
        pin_present=True,
        pin_bright_ratio=1.0,
        primary_diff_score=0.0,
    )
    coordinator._turn_counter += 1
    blocked, info = coordinator._update_cinematic_state(idle_detection)
    assert blocked is True
    assert coordinator._cinematic_release_cooldown == 0

    coordinator._turn_counter += 1
    blocked, info = coordinator._update_cinematic_state(idle_detection)
    assert blocked is False
    assert info["blocked"] is False
