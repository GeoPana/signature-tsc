from __future__ import annotations

from math import factorial

import iisignature
import numpy as np
import pytest

from sigtsc.features.signature import (
    LogSigWindowConfig,
    logsignature_level_dims,
    logsig_features,
    path_signature_features,
    post_rescale_features,
    pre_rescale_path,
    signature_level_dims,
    validate_rescaling_mode,
)


def test_signature_level_dims() -> None:
    assert signature_level_dims(3, 3) == [3, 9, 27]


def test_post_rescaling_full_signature_blocks() -> None:
    features = np.ones(39, dtype=float)

    out = post_rescale_features(
        features,
        input_dim=3,
        level=3,
        transform="signature",
    )

    assert np.allclose(out[:3], 1.0)
    assert np.allclose(out[3:12], 2.0)
    assert np.allclose(out[12:39], 6.0)
    assert np.allclose(features, 1.0)


def test_pre_rescaling_does_not_mutate_input() -> None:
    X = np.arange(12, dtype=float).reshape(4, 3)
    original = X.copy()

    out = pre_rescale_path(X, level=3)

    alpha = factorial(3) ** (1 / 3)
    assert np.allclose(X, original)
    assert np.allclose(out, original * alpha)


def test_invalid_rescaling_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported rescaling mode"):
        validate_rescaling_mode("banana")


def test_rescaling_integration_signature_dimension_and_determinism() -> None:
    rng = np.random.default_rng(123)
    paths = [rng.normal(size=(18, 2)), rng.normal(size=(18, 2))]
    windowing = LogSigWindowConfig(type="global", aggregation="concat")

    none = path_signature_features(
        paths,
        level=3,
        transform_type="signature",
        with_time=True,
        windowing=windowing,
        rescaling="none",
    )
    pre_1 = path_signature_features(
        paths,
        level=3,
        transform_type="signature",
        with_time=True,
        windowing=windowing,
        rescaling="pre",
    )
    pre_2 = path_signature_features(
        paths,
        level=3,
        transform_type="signature",
        with_time=True,
        windowing=windowing,
        rescaling="pre",
    )
    post = path_signature_features(
        paths,
        level=3,
        transform_type="signature",
        with_time=True,
        windowing=windowing,
        rescaling="post",
    )

    assert none.ndim == 2
    assert pre_1.ndim == 2
    assert post.ndim == 2
    assert none.shape == pre_1.shape == post.shape
    assert np.allclose(pre_1, pre_2)


def test_default_rescaling_is_equivalent_to_none_for_logsig_wrapper() -> None:
    rng = np.random.default_rng(456)
    paths = [rng.normal(size=(16, 2)), rng.normal(size=(16, 2))]

    old_style = logsig_features(paths, level=2, with_time=True)
    explicit_none = logsig_features(
        paths,
        level=2,
        with_time=True,
        rescaling="none",
    )

    assert np.allclose(old_style, explicit_none)


def test_logsignature_post_rescaling_level_dims_and_shape() -> None:
    dims = logsignature_level_dims(3, 3)

    assert sum(dims) == iisignature.logsiglength(3, 3)

    rng = np.random.default_rng(789)
    paths = [rng.normal(size=(20, 2)), rng.normal(size=(20, 2))]
    none = path_signature_features(
        paths,
        level=3,
        transform_type="logsig",
        with_time=True,
        rescaling="none",
    )
    post = path_signature_features(
        paths,
        level=3,
        transform_type="logsig",
        with_time=True,
        rescaling="post",
    )

    assert post.shape == none.shape
