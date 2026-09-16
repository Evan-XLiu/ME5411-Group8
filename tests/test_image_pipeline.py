"""Focused regression tests for the deterministic Task 1-4 primitives."""

import pytest

np = pytest.importorskip("numpy")

from common.image_ops import connected_components, contrast_stretch, crop_components, otsu_threshold


def test_contrast_stretch_uses_full_uint8_range():
    result = contrast_stretch(np.array([[10, 20], [30, 40]], dtype=np.uint8))
    assert result.tolist() == [[0, 85], [170, 255]]


def test_otsu_separates_two_clear_modes():
    image = np.array([[10, 10, 10, 10], [240, 240, 240, 240]], dtype=np.uint8)
    threshold, mask = otsu_threshold(image)
    assert 10 <= threshold < 240
    assert mask.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1]]


def test_connected_components_supports_connectivity():
    diagonal = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    _, four = connected_components(diagonal, connectivity=4)
    _, eight = connected_components(diagonal, connectivity=8)
    assert len(four) == 2
    assert len(eight) == 1


def test_component_crops_accept_zero_or_255_masks():
    binary = np.zeros((4, 6), dtype=np.uint8)
    binary[1:3, 2:4] = 255
    _, components = connected_components(binary)
    crops = crop_components(binary, components, padding=0)
    assert len(crops) == 1
    assert int(crops[0].max()) == 255
