import logging
from pathlib import Path

import pytest
from PIL import Image

from lluma_agent.cinematics import (
    CinematicDetectionResult,
    CinematicDetector,
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


def test_diff_detects_fullscreen_motion(capture_config):
    detector = CinematicDetector(capture_config)
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_1.png"),
            menu_is_usable=False,
        )
    )
    active = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_2.png"),
            menu_is_usable=False,
        )
    )
    assert active.kind is CinematicKind.FULLSCREEN
    assert active.playback is PlaybackState.PLAYING
    assert active.diff_score > 0.0
    assert not active.loading_screen


def test_loading_screen_flagged_even_when_static(capture_config):
    detector = CinematicDetector(capture_config)
    frame = _load_rgb_image("loading_fullscreen.png")
    first = detector.observe(
        CinematicObservation(
            image=frame,
            menu_is_usable=False,
        )
    )
    repeat = detector.observe(
        CinematicObservation(
            image=frame,
            menu_is_usable=False,
        )
    )
    for result in (first, repeat):
        assert result.kind is CinematicKind.FULLSCREEN
        assert result.loading_screen is True
        assert result.playback is PlaybackState.PLAYING
        assert result.diff_score == pytest.approx(0.0, abs=1e-6)


def test_aggressive_static_menu_remains_interactive(capture_config):
    detector = CinematicDetector(capture_config)
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image("aggressive_static_1.png"),
            menu_is_usable=True,
        )
    )
    stable = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("aggressive_static_2.png"),
            menu_is_usable=True,
        )
    )
    assert stable.kind is CinematicKind.NONE
    assert stable.anchor_stable_regions >= 1
    assert stable.pin_present is True


def test_paused_frame_retains_active_kind(capture_config):
    detector = CinematicDetector(capture_config)
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_1.png"),
            menu_is_usable=False,
        )
    )
    playing = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_2.png"),
            menu_is_usable=False,
        )
    )
    paused = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("fullscreen_cutscene_2.png"),
            menu_is_usable=False,
        )
    )
    assert playing.kind is CinematicKind.FULLSCREEN
    assert playing.playback is PlaybackState.PLAYING
    assert paused.kind is playing.kind
    assert paused.playback is PlaybackState.PAUSED


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
        skip_hint_primary=False,
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
    assert coordinator._cinematic_release_cooldown == 0
    assert info["buffer_turns_remaining"] == 0

    coordinator._turn_counter += 1
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
