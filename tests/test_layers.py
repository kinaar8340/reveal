"""Layer names and residual_R helpers."""

from __future__ import annotations

import numpy as np

from reveal.layers import (
    LayerNames,
    circular_dispersion,
    geometric_residual,
    residual_R,
)


def test_layer_names_are_canonical():
    names = LayerNames()
    assert names.global_R == "global_R"
    assert names.frame_R == "frame_R"
    assert names.residual_R == "residual_R"


def test_residual_rms():
    obs = np.array([1.0, 2.0, 2.0])
    pred = np.array([1.0, 2.0, 1.0])
    assert residual_R(obs, pred) == np.sqrt(1.0 / 3.0)


def test_geometric_residual_small_for_angle_clusters():
    # Three tight clusters on the circle
    angles = np.array([0.0, 0.05, -0.05, 2.0, 2.04, 1.96, 4.0, 4.05, 3.95])
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert geometric_residual(labels, angles) < 0.1


def test_geometric_residual_inf_when_no_class_qualifies():
    angles = np.array([0.0, 1.0, 2.0])
    labels = np.array([0, 1, 2])
    assert geometric_residual(labels, angles) == float("inf")


def test_circular_dispersion_zero_for_identical():
    assert circular_dispersion(np.array([0.3, 0.3, 0.3])) == 0.0
