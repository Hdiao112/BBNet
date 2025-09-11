#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import boxcox
import os
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import joblib

np.set_printoptions(precision=10)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

class ResMLPWithAttn(nn.Module):
    def __init__(self,
                 in_dim: int = 4,
                 hidden_dim: int = 4096,
                 depth: int = 8,
                 dropout_p: float = 0.3,
                 n_heads: int = 8,
                 out_dim: int = 2):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=n_heads,
                                          dropout=dropout_p,
                                          batch_first=True)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
            )
            for _ in range(depth)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.input_lin(x))
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h = h + attn_out.squeeze(1)
        for blk in self.blocks:
            h = blk(h) + h
        return self.head(h)

class ResidualNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, depth=6, dropout_p=0.3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
        for _ in range(depth):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
'''
def load_data(path, input_cols, output_cols):
    df = pd.read_csv(path).head(10000)
    miss_in  = [c for c in input_cols  if c not in df.columns]
    miss_out = [c for c in output_cols if c not in df.columns]
    if miss_in or miss_out:
        raise ValueError(f"Missing columns: inputs {miss_in}, outputs {miss_out}")
    X = df[input_cols].values.astype(np.float32)
    Y = df[output_cols].values.astype(np.float32)
    return X, Y
'''
def load_data(path, input_cols, output_cols):
    df = pd.read_csv(path)
    dh_col = 'H2/H'
    df_filtered = df[(df[dh_col] > 1e-6) & (df[dh_col] < 1e-4)]

    miss_in  = [c for c in input_cols  if c not in df_filtered.columns]
    miss_out = [c for c in output_cols if c not in df_filtered.columns]
    if miss_in or miss_out:
        raise ValueError(f"Missing columns: inputs {miss_in}, outputs {miss_out}")
    
    X = df_filtered[input_cols].values.astype(np.float32)
    Y = df_filtered[output_cols].values.astype(np.float32)
    return X, Y, len(df_filtered)

def preprocess_data_corrected(X, Y, input_cols, output_cols,
                             idx_tr, idx_val, idx_te,
                             use_scaling=False, use_boxcox=False):
    eps = 1e-10

    X_tr, X_val, X_te = X[idx_tr].copy(), X[idx_val].copy(), X[idx_te].copy()
    Y_tr, Y_val, Y_te = Y[idx_tr], Y[idx_val], Y[idx_te]

    i_dd0 = input_cols.index('dd0')
    log_tr = np.log10(np.maximum(X_tr[:, i_dd0], eps)).reshape(-1,1)
    scaler_dd0 = MinMaxScaler().fit(log_tr)
    X_tr[:, i_dd0]  = scaler_dd0.transform(log_tr).squeeze()
    X_val[:, i_dd0] = scaler_dd0.transform(
        np.log10(np.maximum(X_val[:, i_dd0], eps)).reshape(-1,1)
    ).squeeze()
    X_te[:, i_dd0]  = scaler_dd0.transform(
        np.log10(np.maximum(X_te[:, i_dd0], eps)).reshape(-1,1)
    ).squeeze()

    scaler_X = StandardScaler().fit(X_tr)
    X_tr_s  = scaler_X.transform(X_tr)
    X_val_s = scaler_X.transform(X_val)
    X_te_s  = scaler_X.transform(X_te)

    i_H2 = output_cols.index('H2/H')
    H2_tr_log  = np.log10(np.maximum(Y_tr[:, i_H2], eps)).reshape(-1,1)
    H2_val_log = np.log10(np.maximum(Y_val[:, i_H2], eps)).reshape(-1,1)
    H2_te_log  = np.log10(np.maximum(Y_te[:, i_H2], eps)).reshape(-1,1)

    lambda_h2 = None
    shift = None
    mn = None
    mx = None

    if use_boxcox:
        shift = 0.01 - H2_tr_log.min()
        tr_flat = (H2_tr_log.flatten() + shift)
        H2_tr_trans, lambda_h2 = boxcox(tr_flat)
        H2_val_trans = boxcox(H2_val_log.flatten() + shift, lmbda=lambda_h2)
        H2_te_trans  = boxcox(H2_te_log.flatten() + shift, lmbda=lambda_h2)
        mn, mx = H2_tr_trans.min(), H2_tr_trans.max()
        H2_tr_in  = ((H2_tr_trans.reshape(-1,1) - mn)/(mx-mn+eps)) * 0.01
        H2_val_in = ((H2_val_trans.reshape(-1,1) - mn)/(mx-mn+eps)) * 0.01
        H2_te_in  = ((H2_te_trans.reshape(-1,1) - mn)/(mx-mn+eps)) * 0.01
    elif use_scaling:
        H2_tr_in  = H2_tr_log  / 100
        H2_val_in = H2_val_log / 100
        H2_te_in  = H2_te_log  / 100
    else:
        H2_tr_in  = H2_tr_log
        H2_val_in = H2_val_log
        H2_te_in  = H2_te_log

    scaler_H2 = StandardScaler().fit(H2_tr_in)
    H2_tr_s  = scaler_H2.transform(H2_tr_in)
    H2_val_s = scaler_H2.transform(H2_val_in)
    H2_te_s  = scaler_H2.transform(H2_te_in)

    i_Yp = output_cols.index('Yp')
    Yp_tr = Y_tr[:, i_Yp].reshape(-1,1)
    scaler_Y = StandardScaler().fit(Yp_tr)
    Yp_tr_s  = scaler_Y.transform(Yp_tr)
    Yp_val_s = scaler_Y.transform(Y_val[:, i_Yp].reshape(-1,1))
    Yp_te_s  = scaler_Y.transform(Y_te[:, i_Yp].reshape(-1,1))

    Y_tr_s  = np.hstack([H2_tr_s,  Yp_tr_s])
    Y_val_s = np.hstack([H2_val_s, Yp_val_s])
    Y_te_s  = np.hstack([H2_te_s,  Yp_te_s])

    return (
        X_tr_s, X_val_s, X_te_s,
        Y_tr_s, Y_val_s, Y_te_s,
        scaler_dd0, scaler_X,
        scaler_H2, scaler_Y,
        lambda_h2, shift, mn, mx
    )
'''
def train_epoch(model, loader, opt, device, weights,
                scaler_H2, idx_h2, lambda_h2, shift, delta=0.1):
    model.train()
    total, cnt = 0.0, 0
    scale_h2 = float(scaler_H2.scale_[0])
    mean_h2  = float(scaler_H2.mean_[0])
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(xb)
        base_loss = (weights * torch.abs(pred - yb)).mean()
        pred_log = pred[:, idx_h2] * scale_h2 + mean_h2
        true_log = yb[:, idx_h2]  * scale_h2 + mean_h2
        mape_h2 = F.l1_loss(pred_log, true_log)
        loss = base_loss + (lambda_h2 * mape_h2 if lambda_h2 else 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += loss.item() * xb.size(0)
        cnt   += xb.size(0)
    return total / cnt if cnt else 0.0
'''
'''
def train_epoch(model, loader, opt, device, weights,
                scaler_H2, idx_h2, lambda_h2, shift,
                lambda_std=0.5, eps=1e-10):
    model.train()
    total_loss = 0.0
    total_cnt  = 0

    scale_h2 = float(scaler_H2.scale_[0])
    mean_h2  = float(scaler_H2.mean_[0])

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        # forward
        pred = model(xb)                     # shape (B, 2)

        # weighted MAE loss for both outputs
        mae_loss = (weights * torch.abs(pred - yb)).mean()

        # recover log10(H2/H) from normalized prediction and target
        pred_log = pred[:, idx_h2] * scale_h2 + mean_h2
        true_log = yb[:,   idx_h2] * scale_h2 + mean_h2

        # MAPE on log-scale H2/H (as in original code)
        mape_h2 = F.l1_loss(pred_log, true_log)

        # std penalty on relative error: (pred_log - true_log) / true_log
        rel_err  = (pred_log - true_log) / (true_log + eps)
        std_term = rel_err.std()

        # total loss
        loss = mae_loss
        if lambda_h2:
            loss = loss + lambda_h2 * mape_h2
        if lambda_std:
            loss = loss + lambda_std * std_term

        # backward & step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item() * xb.size(0)
        total_cnt  += xb.size(0)

    return total_loss / total_cnt if total_cnt else 0.0
'''
def train_epoch(model, loader, opt, device, weights,
                scaler_H2, idx_h2, lambda_h2, shift,
                lambda_std=0.0, eps=1e-10):
    model.train()
    total_loss = 0.0
    noise_scale=0.02
    total_cnt  = 0

    scale_h2 = float(scaler_H2.scale_[0])
    mean_h2  = float(scaler_H2.mean_[0])

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()

        # forward pass
        pred = model(xb)                     # shape (B, 2)

        # weighted MAE loss for both outputs (no sample-weight sw)
        mae_loss = (weights * torch.abs(pred - yb)).mean()

        # recover log10(H2/H) from normalized prediction and target
        pred_log = pred[:, idx_h2] * scale_h2 + mean_h2
        true_log = yb[:,   idx_h2] * scale_h2 + mean_h2

        # MAPE on log-scale H2/H
        mape_h2 = F.l1_loss(pred_log, true_log)

        # std penalty on relative error: (pred_log - true_log) / true_log
        rel_err  = (pred_log - true_log) / (true_log + eps)
        std_term = rel_err.std()
        noise = torch.randn_like(xb) * noise_scale
        pred2 = model(xb + noise)
        smooth_loss = F.mse_loss(pred, pred2)

        # total loss
        loss = mae_loss + 0.1*smooth_loss
        if lambda_h2:
            loss = loss + lambda_h2 * mape_h2
        if lambda_std:
            loss = loss + lambda_std * std_term

        # backward & step with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total_loss += loss.item() * xb.size(0)
        total_cnt  += xb.size(0)

    return total_loss / total_cnt if total_cnt else 0.0

def train_residual(res_model, train_loader, main_model, opt2,
                   device, scaler_H2, idx_h2, scale_h2, mean_h2,
                   Y_raw_full, idxs_train, eps=1e-10):
    res_model.train()
    total, cnt, offset = 0.0, 0, 0
    for Xb_s, _ in train_loader:
        B = Xb_s.size(0)
        xb = Xb_s.to(device)
        batch_idxs = idxs_train[offset: offset + B]
        offset += B
        true_raw = torch.from_numpy(Y_raw_full[batch_idxs, idx_h2]).to(device)
        with torch.no_grad():
            pred_main = main_model(xb)
        pred_log = pred_main[:, idx_h2] * scale_h2 + mean_h2
        pred_raw = torch.pow(10.0, pred_log.clamp(min=-10))
        inp = torch.cat([xb, pred_raw.unsqueeze(1)], dim=1)
        resid_pred = res_model(inp).squeeze()
        pred_final = pred_raw + resid_pred
        loss = torch.mean(torch.abs((pred_final - true_raw) / (true_raw + eps)))
        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(res_model.parameters(), max_norm=1.0)
        opt2.step()
        total += loss.item() * B
        cnt   += B
    return total / cnt if cnt else 0.0

def compute_metrics(Yt, Yp):
    if Yt.size == 0 or Yt.shape != Yp.shape:
        return np.nan, np.nan, np.nan
    return (
        mean_squared_error(Yt, Yp),
        mean_absolute_error(Yt, Yp),
        r2_score(Yt, Yp)
    )

@torch.no_grad()
def test_sample(model, Xte_s, Y_raw_test,
                scaler_H2, scaler_Y, device, output_cols,
                use_residual=False, res_model=None,
                idx_h2=None, lambda_h2=None,
                use_scaling=False, shift=None, min_h2=None, max_h2=None):
    if len(Xte_s) == 0:
        print("\nNo test sample.")
        return
    idx = np.random.randint(len(Xte_s))
    x = Xte_s[idx:idx+1]
    xp = torch.from_numpy(x).to(device)
    model.eval()
    with torch.no_grad():
        pred_main = model(xp)
        if use_residual and res_model is not None and idx_h2 is not None:
            resid_in = torch.cat([xp, pred_main[:, idx_h2:idx_h2+1]], dim=1)
            pred_main[:, idx_h2] += res_model(resid_in).squeeze()
    yp_s = pred_main.cpu().numpy()
    h2_scaled = scaler_H2.inverse_transform(yp_s[:, [0]]).squeeze()
    if lambda_h2 is not None:
        bc = h2_scaled * (max_h2 - min_h2) / 0.01 + min_h2
        inv_log = (bc * lambda_h2 + 1)**(1/lambda_h2) - shift
    elif use_scaling:
        inv_log = h2_scaled * 100
    else:
        inv_log = h2_scaled
    inv_H2 = 10.0 ** inv_log
    inv_Yp = scaler_Y.inverse_transform(yp_s[:, [1]]).squeeze()
    y_all = np.hstack([inv_H2, inv_Yp])
    true = Y_raw_test[idx]
    print(f"\nSample {idx}:")
    for i, col in enumerate(output_cols):
        print(f" {col}: true={true[i]:.6e}, pred={y_all[i]:.6e}")

def main():
    set_seed(42)
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--hidden_dim', type=int, default=4096)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--dropout_p', type=float, default=0.3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--lambda_h2', type=float, default=0.0)
    p.add_argument('--use_residual', action='store_true')
    p.add_argument('--res_hidden_dim', type=int, default=512)
    p.add_argument('--res_depth', type=int, default=6)
    p.add_argument('--res_lr', type=float, default=1e-5)
    p.add_argument('--res_epochs', type=int, default=500)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--use_scaling', action='store_true')
    p.add_argument('--use_boxcox', action='store_true')
    args = p.parse_args()

    input_cols  = ['dd0', 'dd0_rad', 'tau', 'omegabn']
    output_cols = ['H2/H', 'Yp']
    idx_h2 = output_cols.index('H2/H')
    

    X_raw, Y_raw, lens = load_data(args.data, input_cols, output_cols)
   
    n = len(X_raw)
    test_sz, val_sz = 1000, 1000
    idx = np.arange(n)
    idx_rem, idx_te  = train_test_split(idx, test_size=test_sz,  random_state=42)
    idx_tr,  idx_val = train_test_split(idx_rem, test_size=val_sz, random_state=42)
    print(f"Sizes: train={len(idx_tr)}, val={len(idx_val)}, test={len(idx_te)}")

    df_test = pd.DataFrame(X_raw[idx_te], columns=input_cols)
    df_test = pd.concat([df_test, pd.DataFrame(Y_raw[idx_te], columns=output_cols)], axis=1)
    df_test.to_csv("test_split_alter_fin_org2.csv", index=False)

    (X_tr_s, X_val_s, X_te_s,
     Y_tr_s, Y_val_s, Y_te_s,
     scaler_dd0, scaler_X,
     scaler_H2, scaler_Y,
     lambda_h2, shift, min_h2, max_h2) = preprocess_data_corrected(
        X_raw, Y_raw, input_cols, output_cols,
        idx_tr, idx_val, idx_te,
        use_scaling=args.use_scaling,
        use_boxcox=args.use_boxcox
    )
    Y_raw_test = Y_raw[idx_te]
    
    dh_train_raw = Y_raw[idx_tr, 0]  
    bin_edges = [1e-6, 1e-5, 1e-4]
    dh_bins = np.digitize(dh_train_raw, bins=bin_edges)
    bin_counts = np.bincount(dh_bins, minlength=len(bin_edges)+1)
    sample_weights = 1.0 / bin_counts[dh_bins]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_ld = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(Y_tr_s)),
        batch_size=args.batch,
        sampler=sampler
    )

    val_ld   = DataLoader(TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(Y_val_s)),
                          batch_size=args.batch, shuffle=False)

    joblib.dump({
        'scaler_dd0':  scaler_dd0,
        'scaler_X':    scaler_X,
        'scaler_H2':   scaler_H2,
        'scaler_Y':    scaler_Y,
        'lambda_h2':   lambda_h2,
        'shift':       shift,
        'min_h2':      min_h2,
        'max_h2':      max_h2,
        'use_scaling': args.use_scaling,
        'use_boxcox':  args.use_boxcox
    }, "scalers_alterbbn_fin_org2.pkl")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResMLPWithAttn(len(input_cols), args.hidden_dim, args.depth,
                           args.dropout_p, args.n_heads, len(output_cols)).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10)
    weights = torch.tensor([20.0, 30.0], device=device)

    best_val, best_fp = float('inf'), 'best_model_alter_fin_org2.pth'
    train_losses, val_losses, lrs = [], [], []
    for ep in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_ld, opt, device,
                              weights, scaler_H2, idx_h2, args.lambda_h2, shift)
        model.eval(); v_loss=v_cnt=0
        with torch.no_grad():
            for xb,yb in val_ld:
                xb,yb = xb.to(device), yb.to(device)
                pv = model(xb)
                l = (weights * torch.abs(pv - yb)).mean()
                v_loss += l.item()*xb.size(0); v_cnt += xb.size(0)
        val_loss = v_loss/v_cnt if v_cnt else float('inf')
        train_losses.append(tr_loss); val_losses.append(val_loss)
        lrs.append(opt.param_groups[0]['lr'])
        sched.step(val_loss)
        if val_loss < best_val:
            best_val=val_loss; torch.save(model.state_dict(), best_fp)
            print(f"Ep{ep}: new best val {best_val:.6f}")
        if ep%10==0 or ep==args.epochs:
            print(f"Ep{ep}, Train={tr_loss:.6f}, Val={val_loss:.6f}")

    plt.figure()
    plt.plot(range(1, args.epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, args.epochs+1), val_losses,   label='Val Loss')
    plt.axhline(0, linestyle='--')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss',  fontsize=14)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('loss_curve_al_fin_org2.png', dpi=300)
    plt.close()
    plt.figure()
    plt.plot(range(1, args.epochs+1), lrs, label='Learning Rate')
    
    plt.xlabel('Epoch',        fontsize=14)
    plt.ylabel('Learning Rate',fontsize=14)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig('lr_schedule_al_fin_org2.png', dpi=300)
    plt.close()

    if args.use_residual:
        for p in model.parameters(): p.requires_grad=False
        res_model = ResidualNet(len(input_cols)+1,args.res_hidden_dim,args.res_depth).to(device)
        opt2 = torch.optim.Adam(res_model.parameters(),lr=args.res_lr,weight_decay=1e-5)
        for _ in range(args.res_epochs):
            train_residual(res_model, train_ld, model, opt2,
                           device, scaler_H2, idx_h2,
                           float(scaler_H2.scale_[0]), float(scaler_H2.mean_[0]),
                           Y_raw, idx_tr)
    else:
        res_model = None

    if os.path.exists(best_fp):
        model.load_state_dict(torch.load(best_fp, map_location=device))
        print(f"Loaded best model from {best_fp}")

    if len(X_te_s) > 0:
        print("\nTest Set Evaluation:")
        X_te_t = torch.from_numpy(X_te_s).to(device)
        with torch.no_grad():
            pm = model(X_te_t)
            if args.use_residual and res_model is not None:
                resid_in = torch.cat([X_te_t, pm[:, idx_h2:idx_h2+1]], dim=1)
                pm[:, idx_h2] += res_model(resid_in).squeeze()
        pm_np = pm.cpu().numpy()
        final_pred_raw = np.zeros_like(Y_raw_test)

        h2_scaled = scaler_H2.inverse_transform(pm_np[:, [0]]).squeeze()
        if lambda_h2 is not None:
            bc = h2_scaled * (max_h2 - min_h2) / 0.01 + min_h2
            inv_log_H2 = (bc * lambda_h2 + 1)**(1/lambda_h2) - shift
        elif args.use_scaling:
            inv_log_H2 = h2_scaled * 100
        else:
            inv_log_H2 = h2_scaled
        final_pred_raw[:, 0] = 10.0**inv_log_H2

        final_pred_raw[:, 1] = scaler_Y.inverse_transform(pm_np[:, [1]]).squeeze()

        print("\nPer-output raw-space metrics:")
        for i, c in enumerate(output_cols):
            mse_i, mae_i, r2_i = compute_metrics(Y_raw_test[:, i], final_pred_raw[:, i])
            rmse_i = np.sqrt(mse_i)
            print(f" {c}: MAE={mae_i:.6e}, RMSE={rmse_i:.6e}, R2={r2_i:.4f}")

        abs_err = final_pred_raw - Y_raw_test
        pd.DataFrame(abs_err, columns=output_cols).to_csv("absolute_errors_fin_org2.csv", index=False)
        
        eps = 1e-10
        abs_err = final_pred_raw - Y_raw_test
        signed = abs_err.mean(axis=0)
        ape    = np.abs(abs_err / (Y_raw_test + eps)) * 100
        smape  = np.abs(abs_err) / ((np.abs(final_pred_raw) + np.abs(Y_raw_test))/2 + eps) * 100

        print("\nMean Signed Error per output:")
        for c, v in zip(output_cols, signed):
            print(f" {c}: {v:.6e}")

        print("\nMAPE per output (%):")
        for c, v in zip(output_cols, np.nanmean(ape, axis=0)):
            print(f" {c}: {v:.4f}%")

        print("\nMdAPE per output (%):")
        for c, v in zip(output_cols, np.nanmedian(ape, axis=0)):
            print(f" {c}: {v:.4f}%")

        print("\nsMAPE per output (%):")
        for c, v in zip(output_cols, np.nanmean(smape, axis=0)):
            print(f" {c}: {v:.4f}%")

        test_sample(model, X_te_s, Y_raw_test,
                    scaler_H2, scaler_Y, device, output_cols,
                    use_residual=args.use_residual,
                    res_model=res_model,
                    idx_h2=idx_h2,
                    lambda_h2=lambda_h2,
                    use_scaling=args.use_scaling,
                    shift=shift,
                    min_h2=min_h2,
                    max_h2=max_h2)
    else:
        print("\nNo test data.")

if __name__ == "__main__":
    main()
