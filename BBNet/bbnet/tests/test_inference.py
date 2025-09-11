import numpy as np
import pytest

from bbnet import predict


def test_predict_pe_shape():
    """Check Parthenope (pe) mode output shape"""
    X = np.array([[6.1, 0.3, 1.0, 880.0]])  # [eta10, dn_nu, kappa, tau]
    y = predict(X, mode="pe")
    assert isinstance(y, np.ndarray), "Output should be a numpy array"
    assert y.shape == (1, 2), "Expected output shape (1, 2) for [H2/H, Yp]"


def test_predict_al_shape():
    """Check AlterBBN (al) mode output shape"""
    X = np.array([[1e-3, 1e-4, 880.0, 0.022]])  # [dd0, dd0_rad, tau, omegabn]
    y = predict(X, mode="al")
    assert isinstance(y, np.ndarray), "Output should be a numpy array"
    assert y.shape == (1, 2), "Expected output shape (1, 2) for [H2/H, Yp]"


def test_predict_values_non_nan():
    """Check predictions do not contain NaN/Inf"""
    X_pe = np.array([[6.1, 0.3, 1.0, 880.0]])
    X_al = np.array([[1e-3, 1e-4, 880.0, 0.022]])

    y_pe = predict(X_pe, mode="pe")
    y_al = predict(X_al, mode="al")

    assert np.all(np.isfinite(y_pe)), "Parthenope predictions contain NaN/Inf"
    assert np.all(np.isfinite(y_al)), "AlterBBN predictions contain NaN/Inf"
