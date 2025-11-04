from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from lluma_agent.cinematics import (
    CinematicDetector,
    CinematicKind,
    CinematicObservation,
    PlaybackState,
)
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


def test_primary_cutscene_skip_chip_detection(capture_config):
    detector = CinematicDetector(capture_config)
    result = detector.observe(
        CinematicObservation(
            image=_load_rgb_image("primary_cutscene_skip.png"),
            menu_is_usable=False,
        )
    )
    assert result.kind is CinematicKind.PRIMARY
    assert result.skip_hint_primary is True
    assert result.playback in {PlaybackState.PAUSED, PlaybackState.PLAYING}
