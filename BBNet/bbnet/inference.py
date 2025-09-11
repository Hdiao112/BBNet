import numpy as np
import torch
import joblib
import importlib.resources as pkg_resources

from .model import ResMLPWithAttn

ARTIFACTS = {
    "pe": {
        "model": "model_pe.pth",
        "scaler": "scaler_pe.pkl",
        "hidden_dim": 4096,
        "min_key": "min_val_h2",
        "max_key": "max_val_h2",
    },
    "al": {
        "model": "model_al.pth",
        "scaler": "scaler_al.pkl",
        "hidden_dim": 4096,
        "min_key": "min_h2",
        "max_key": "max_h2",
    }
}


def load_scalers(path: str, mode: str):
    scalers = joblib.load(path)
    if mode == "pe":
        required = ["scaler_X", "scaler_kappa", "scaler_H2", "scaler_Y",
                    "lambda_h2", "shift", "min_val_h2", "max_val_h2"]
    elif mode == "al":
        required = ["scaler_X", "scaler_dd0", "scaler_H2", "scaler_Y",
                    "lambda_h2", "shift", "min_h2", "max_h2"]
    else:
        raise ValueError("Invalid mode, must be 'pe' or 'al'")

    missing = [k for k in required if k not in scalers]
    if missing:
        raise ValueError(f"Missing scalers for mode {mode}: {missing}")
    return scalers


def load_model(mode: str, model_path: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = ARTIFACTS[mode]["hidden_dim"]
    model = ResMLPWithAttn(hidden_dim=hidden_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _invert_h2(y_scaled_col0, scalers, mode):
    scaler_H2 = scalers["scaler_H2"]
    h2_scaled = scaler_H2.inverse_transform(y_scaled_col0.reshape(-1, 1)).flatten()
    lambda_h2 = scalers.get("lambda_h2", None)
    shift = scalers.get("shift", None)
    mn = scalers.get(ARTIFACTS[mode]["min_key"], None)
    mx = scalers.get(ARTIFACTS[mode]["max_key"], None)

    if scalers.get("use_boxcox", False) and lambda_h2 is not None:
        bc = mn + (h2_scaled / 0.01) * (mx - mn)
        inv_log = (bc * lambda_h2 + 1) ** (1.0 / lambda_h2) - shift
    elif scalers.get("use_scaling", False):
        inv_log = h2_scaled * 100.0
    else:
        inv_log = h2_scaled

    inv_log = np.nan_to_num(inv_log, nan=0.0, posinf=1e10, neginf=-1e10)
    inv_log = np.clip(inv_log, -308, 308)
    return 10.0 ** inv_log


def _preprocess_X(X, scalers, mode):
    X = X.astype(np.float32, copy=True)
    eps = 1e-10
    if mode == "al":
        log0 = np.log10(np.maximum(X[:, 0], eps)).reshape(-1, 1)
        X[:, 0] = scalers["scaler_dd0"].transform(log0).flatten()
    return scalers["scaler_X"].transform(X).astype(np.float32)


def predict(X, mode: str, batch_size=256, device=None):
    if mode not in ARTIFACTS:
        raise ValueError(f"Invalid mode {mode}, must be one of {list(ARTIFACTS.keys())}")

    with pkg_resources.path("bbnet.artifacts", ARTIFACTS[mode]["model"]) as model_path, \
         pkg_resources.path("bbnet.artifacts", ARTIFACTS[mode]["scaler"]) as scaler_path:

        scalers = load_scalers(str(scaler_path), mode)
        model = load_model(mode, str(model_path), device=device)

        X_scaled = _preprocess_X(X, scalers, mode)

        preds = []
        with torch.no_grad():
            for i in range(0, len(X_scaled), batch_size):
                xb = torch.from_numpy(X_scaled[i:i + batch_size]).to(next(model.parameters()).device)
                preds.append(model(xb).cpu().numpy())
        P = np.vstack(preds)

        H2_pred = _invert_h2(P[:, 0], scalers, mode)
        Yp_pred = scalers["scaler_Y"].inverse_transform(P[:, 1].reshape(-1, 1)).flatten()

        Y_pred = np.column_stack([H2_pred, Yp_pred])
        return np.nan_to_num(Y_pred, nan=0.0, posinf=1e10, neginf=-1e10)
