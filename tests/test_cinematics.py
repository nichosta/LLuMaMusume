from __future__ import annotations

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
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def capture_config():
    _, capture_cfg, _ = load_configs(Path("config.yaml"))
    return capture_cfg


def _load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def test_loading_screen_detection_white(capture_config):
    detector = CinematicDetector(capture_config)
    image = _load_rgb_image(REPO_ROOT / "fullscreen loading screen.png")

    result = detector.observe(CinematicObservation(image=image))

    assert result.kind is CinematicKind.LOADING
    assert result.is_loading_screen is True
    assert result.playback is PlaybackState.PAUSED


def test_loading_screen_detection_black(capture_config):
    detector = CinematicDetector(capture_config)
    image = Image.new("RGB", (1920, 1080), (0, 0, 0))

    result = detector.observe(CinematicObservation(image=image))

    assert result.kind is CinematicKind.LOADING
    assert result.is_loading_screen is True
    assert result.playback is PlaybackState.PAUSED


def test_cutscene_frames_trigger_cutscene_kind(capture_config):
    detector = CinematicDetector(capture_config)
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image(CAPTURE_DIR / "fullscreen_cutscene_1.png"),
            menu_is_usable=False,
        )
    )

    active = detector.observe(
        CinematicObservation(
            image=_load_rgb_image(CAPTURE_DIR / "fullscreen_cutscene_2.png"),
            menu_is_usable=False,
        )
    )

    assert active.kind is CinematicKind.CUTSCENE
    assert active.playback is PlaybackState.PLAYING
    assert active.aggressive_static is False


def test_aggressive_static_backgrounds_do_not_gate(capture_config):
    detector = CinematicDetector(capture_config)
    detector.observe(
        CinematicObservation(
            image=_load_rgb_image(REPO_ROOT / "most aggressive non cutscene 1.png"),
            menu_is_usable=True,
        )
    )

    animated = detector.observe(
        CinematicObservation(
            image=_load_rgb_image(REPO_ROOT / "most aggressive non cutscene 2.png"),
            menu_is_usable=True,
        )
    )

    assert animated.kind is CinematicKind.NONE
    assert animated.playback is PlaybackState.PLAYING
    assert animated.aggressive_static is True


def test_coordinator_requires_consecutive_low_motion_frames() -> None:
    coordinator = GameLoopCoordinator.__new__(GameLoopCoordinator)
    coordinator._logger = logging.getLogger("test_coordinator")
    coordinator._cinematic_state = CinematicControlState()
    coordinator._cinematic_min_low_frames = 2
    coordinator._cinematic_max_hold_turns = 12
    coordinator._cinematic_release_buffer_turns = 0
    coordinator._cinematic_release_cooldown = 0
    coordinator._turn_counter = 10

    active_detection = CinematicDetectionResult(
        kind=CinematicKind.CUTSCENE,
        playback=PlaybackState.PLAYING,
        diff_score=0.12,
        primary_diff_score=0.1,
        changed_ratio=0.6,
        is_loading_screen=False,
        aggressive_static=False,
    )
    blocked, info = coordinator._update_cinematic_state(active_detection)
    assert blocked is True
    assert info["kind"] == CinematicKind.CUTSCENE.value

    coordinator._turn_counter += 1
    paused_detection = CinematicDetectionResult(
        kind=CinematicKind.NONE,
        playback=PlaybackState.PAUSED,
        diff_score=0.001,
        primary_diff_score=0.001,
        changed_ratio=0.01,
        is_loading_screen=False,
        aggressive_static=False,
    )
    blocked, info = coordinator._update_cinematic_state(paused_detection)
    assert blocked is True
    assert info["low_motion_frames"] == 1

    coordinator._turn_counter += 1
    blocked, info = coordinator._update_cinematic_state(paused_detection)
    assert blocked is False
    assert info["released_via"] == "low_motion"
