#!/usr/bin/env python3
"""
Build physics features from a RapidSim ROOT file (Uproot, pure Python).

Step 3:
  1) Read RapidSim TTree ("DecayTree") via Uproot.
  2) Use EXISTING two-body mass branches (mLcK, mDstK, mLcDst) and add m^2.
  3) Compute helicity angles cos(theta_H) for each pair:
       - If PX/PY/PZ/E exist, use them (scikit-hep 'vector').
       - Else, reconstruct from PT, ETA, PHI + PDG masses (scikit-hep 'particle' & 'vector').
       - If neither set exists, skip angles with a warning.
  4) Merge per-event resonance WEIGHTS (from Step 2, “weights” mode), default pair = mLcDst (Λc⁺D*⁰).
  5) Write two Parquet files: background (label=0) and signal-weighted (label=1, with 'weight').
  6) Quick validation plots: 1D mass checks, 2D Dalitz (phase), 2D Dalitz (weighted).

Usage (from repo root):
  python scripts/build_features.py \
      --infile  data/rapidsim/Lb_LcDst0K_phase.root \
      --weights data/rapidsim/Lb_LcDst0K_w_mLcDst.parquet \
      --outdir  data/processed \
      --figdir  figures \
      --mass-branch mLcDst

Notes:
- Assumes your RapidSim .config created mass branches: mLcK, mDstK, mLcDst.
- Units: GeV. (particle package returns MeV → convert to GeV.)
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import uproot

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt

# Optional deps for angles
try:
    import vector  # scikit-hep vector
except Exception:
    vector = None

try:
    from particle import Particle  # scikit-hep particle (PDG masses)
except Exception:
    Particle = None


# -------------------------- small utilities --------------------------

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def have_all(cols, needed):
    return all(c in cols for c in needed)


# ---------------------- 4-vector construction helpers ----------------------

def build_four_vectors_from_pxpypze(df: pd.DataFrame, prefix_map: dict):
    """
    Build Lorentz 4-vectors (vector.array) from PX, PY, PZ, E columns.

    prefix_map: {"Lc":"Lc_", "Dst":"Dst_", "K":"K_"} -> expects Lc_PX, Lc_PY, Lc_PZ, Lc_E, etc.
    """
    if vector is None:
        raise RuntimeError("Package 'vector' is required to build four-vectors from PX/PY/PZ/E.")

    vecs = {}
    for key, pfx in prefix_map.items():
        px = df[f"{pfx}PX"].to_numpy()
        py = df[f"{pfx}PY"].to_numpy()
        pz = df[f"{pfx}PZ"].to_numpy()
        E  = df[f"{pfx}E"].to_numpy()
        vecs[key] = vector.array({"px": px, "py": py, "pz": pz, "E": E})
    return vecs


def build_four_vectors_from_ptetaphi(df: pd.DataFrame, prefix_map: dict, mass_map_GeV: dict):
    """
    Build Lorentz 4-vectors from PT, ETA, PHI + PDG masses (GeV).

    prefix_map: {"Lc":"Lc_", "Dst":"Dst_", "K":"K_"}
    mass_map_GeV: {"Lc": m_Lc_GeV, "Dst": m_Dst_GeV, "K": m_K_GeV}
    """
    if vector is None:
        raise RuntimeError("Package 'vector' is required to build four-vectors from PT/ETA/PHI.")
    vecs = {}
    for key, pfx in prefix_map.items():
        pt  = df[f"{pfx}PT"].to_numpy()
        eta = df[f"{pfx}ETA"].to_numpy()
        phi = df[f"{pfx}PHI"].to_numpy()
        # PDG relations:
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        p2 = px*px + py*py + pz*pz
        m  = float(mass_map_GeV[key])
        E  = np.sqrt(p2 + m*m)
        vecs[key] = vector.array({"px": px, "py": py, "pz": pz, "E": E})
    return vecs


def cos_angle(u, v):
    """Cosine of the angle between two 3-vectors (spatial parts of 4-vectors)."""
    dot = (u.x * v.x) + (u.y * v.y) + (u.z * v.z)
    nu  = np.sqrt(u.x*u.x + u.y*u.y + u.z*u.z)
    nv  = np.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)
    eps = 1e-12
    return dot / (np.maximum(nu*nv, eps))


def compute_helicity_angles(vecs: dict):
    """
    cos(theta_H) for each pair:
      1) X=(Lc K),   a=Lc,  bachelor B=Dst
      2) X=(Dst K),  a=Dst, bachelor B=Lc
      3) X=(Lc Dst), a=Lc,  bachelor B=K
    """
    pLc  = vecs["Lc"]
    pDst = vecs["Dst"]
    pK   = vecs["K"]

    # 1) X=(Lc K)
    pX1 = pLc + pK
    a1, B1 = pLc, pDst

    # 2) X=(Dst K)
    pX2 = pDst + pK
    a2, B2 = pDst, pLc

    # 3) X=(Lc Dst)
    pX3 = pLc + pDst
    a3, B3 = pLc, pK

    # Boost into X rest frame
    a1_X = a1.boostCM_of_p4(pX1);  B1_X = B1.boostCM_of_p4(pX1)
    a2_X = a2.boostCM_of_p4(pX2);  B2_X = B2.boostCM_of_p4(pX2)
    a3_X = a3.boostCM_of_p4(pX3);  B3_X = B3.boostCM_of_p4(pX3)

    return {
        "costh_LcK":   cos_angle(a1_X, -B1_X),
        "costh_DstK":  cos_angle(a2_X, -B2_X),
        "costh_LcDst": cos_angle(a3_X, -B3_X),
    }


# ------------------------------- main --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build Dalitz features + helicity angles from RapidSim ROOT (Uproot).")
    ap.add_argument("--infile",   required=True, help="Input ROOT file (phase-space sample from RapidSim).")
    ap.add_argument("--weights",  required=True, help="Parquet with per-event weights (Step 2, 'weights' mode).")
    ap.add_argument("--outdir",   default="data/processed", help="Output directory for features parquet files.")
    ap.add_argument("--figdir",   default="figures", help="Output directory for validation plots.")
    ap.add_argument("--mass-branch", default="mLcDst",
                    help="Which two-body mass was reweighted (mLcK, mDstK, or mLcDst). Default: mLcDst (ΛcD*0).")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir(args.figdir)

    # --- Read ROOT tree with Uproot ---
    with uproot.open(args.infile) as f:
        if "DecayTree" not in f:
            raise SystemExit(f"[ERROR] 'DecayTree' not found in {args.infile}.")
        t = f["DecayTree"]
        branches = list(t.keys())

    print(f"[INFO] Found {len(branches)} branches in 'DecayTree'.")

    # --- Load all branches to Pandas ---
    df = t.arrays(library="pd").reset_index(drop=True)
    df["event"] = df.index.astype(np.int64)

    # --- Expected existing mass branches ---
    expected_masses = ["mLcK", "mDstK", "mLcDst"]
    missing = [b for b in expected_masses if b not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] Missing expected mass branches: {missing}. "
                         f"Add 'param' lines in RapidSim .config or adapt names here.")

    # --- Add m^2 (Dalitz) ---
    df["mLcK2"]   = df["mLcK"]**2
    df["mDstK2"]  = df["mDstK"]**2
    df["mLcDst2"] = df["mLcDst"]**2

    # --- Helicity angles ---
    angles = {}
    have_cartesian = have_all(df.columns, [
        "Lc_PX","Lc_PY","Lc_PZ","Lc_E",
        "Dst_PX","Dst_PY","Dst_PZ","Dst_E",
        "K_PX","K_PY","K_PZ","K_E"
    ])
    have_ptetaphi = have_all(df.columns, [
        "Lc_PT","Lc_ETA","Lc_PHI",
        "Dst_PT","Dst_ETA","Dst_PHI",
        "K_PT","K_ETA","K_PHI"
    ])

    if have_cartesian:
        if vector is None:
            warnings.warn("Optional package 'vector' not found -> skipping helicity angles.")
        else:
            vecs = build_four_vectors_from_pxpypze(df, {"Lc":"Lc_", "Dst":"Dst_", "K":"K_"})
            angles = compute_helicity_angles(vecs)

    elif have_ptetaphi:
        if (vector is None) or (Particle is None):
            warnings.warn("Optional packages 'vector' and/or 'particle' not found -> skipping helicity angles.")
        else:
            # PDG masses (MeV) -> GeV
            m_Lc  = Particle.from_pdgid(4122).mass / 1000.0  # Λc+
            m_Dst = Particle.from_pdgid(423).mass  / 1000.0  # D*0
            m_K   = Particle.from_pdgid(-321).mass / 1000.0  # K-
            vecs  = build_four_vectors_from_ptetaphi(
                df, {"Lc":"Lc_", "Dst":"Dst_", "K":"K_"},
                {"Lc": m_Lc, "Dst": m_Dst, "K": m_K}
            )
            angles = compute_helicity_angles(vecs)
    else:
        warnings.warn("No PX/PY/PZ/E or PT/ETA/PHI branches found; helicity angles will be omitted.")

    for k, v in angles.items():
        df[k] = v

    # --- Merge per-event WEIGHTS (signal shaping) ---
    if not os.path.exists(args.weights):
        raise SystemExit(f"[ERROR] Weights file not found: {args.weights}")
    wdf = pd.read_parquet(args.weights)  # columns: event, weight
    if not {"event","weight"}.issubset(wdf.columns):
        raise SystemExit("[ERROR] Weights parquet must contain columns: 'event', 'weight'.")

    feat_cols = ["event",
                 "mLcK","mDstK","mLcDst","mLcK2","mDstK2","mLcDst2"]
    feat_cols += [c for c in ["costh_LcK","costh_DstK","costh_LcDst"] if c in df.columns]

    features = df[feat_cols].merge(wdf, on="event", how="left").fillna({"weight":0.0})

    # Labeled outputs:
    # BACKGROUND: weight must be 1.0 (NOT the merged signal weights)
    bg = features.copy()
    bg["label"]  = 0
    bg["weight"] = 1.0

    # SIGNAL (weighted): keep the merged BW weights
    sg = features.copy()
    sg["label"]  = 1
    # 'weight' already present from merge


    # --- Write parquet ---
    out_bg = os.path.join(args.outdir, "features_bg.parquet")
    # be careful with the dash in CLI arg name; use the attribute name
    mass_branch_tag = args.mass_branch.replace("*","star").replace("/","_")
    out_sg = os.path.join(args.outdir, f"features_sg_weighted_{mass_branch_tag}.parquet")

    bg.to_parquet(out_bg)
    sg.to_parquet(out_sg)
    print(f"[OK] Wrote background features : {out_bg} (rows={len(bg)})")
    print(f"[OK] Wrote signal-weighted    : {out_sg} (rows={len(sg)})")

    # --- Validation plots ---
    ensure_dir(args.figdir)

    # 1) 1D mass checks
    plt.figure(figsize=(12,3))
    for i,(col,lbl) in enumerate([("mLcK","m(ΛcK) [GeV]"),
                                  ("mDstK","m(D*0K) [GeV]"),
                                  ("mLcDst","m(ΛcD*0) [GeV]")]):
        ax = plt.subplot(1,3,i+1)
        ax.hist(df[col], bins=80, histtype="step")
        ax.set_xlabel(lbl); ax.set_ylabel("Events")
    plt.tight_layout()
    fig1 = os.path.join(args.figdir, "mass_checks.png")
    plt.savefig(fig1, dpi=150); plt.close()
    print(f"[OK] Saved {fig1}")

    # 2) Dalitz phase-space (unweighted)
    plt.figure(figsize=(7,5))
    plt.hist2d(df["mLcK2"], df["mDstK2"], bins=400)
    plt.xlabel(r"$m^2(\Lambda_c K)$ [GeV$^2$]")
    plt.ylabel(r"$m^2(D^{*0} K)$ [GeV$^2$]")
    plt.colorbar(label="Entries")
    plt.tight_layout()
    fig2 = os.path.join(args.figdir, "dalitz_phase.png")
    plt.savefig(fig2, dpi=400); plt.close()
    print(f"[OK] Saved {fig2}")

    # 3) Weighted Dalitz (should show band for chosen resonant pair)
    plt.figure(figsize=(7,5))
    plt.hist2d(sg["mLcK2"], sg["mDstK2"], bins=400, weights=sg["weight"])
    plt.xlabel(r"$m^2(\Lambda_c K)$ [GeV$^2$]")
    plt.ylabel(r"$m^2(D^{*0} K)$ [GeV$^2$]")
    plt.colorbar(label="Weighted entries")
    plt.tight_layout()
    fig3 = os.path.join(args.figdir, f"dalitz_signal_weighted_{mass_branch_tag}.png")
    plt.savefig(fig3, dpi=400); plt.close()
    print(f"[OK] Saved {fig3}")

    # Summary
    print("\n[SUMMARY]")
    print(f"  Infile         : {args.infile}")
    print(f"  Weights        : {args.weights}  (resonant pair: {args.mass_branch})")
    print(f"  Features (bg)  : {out_bg}")
    print(f"  Features (sg)  : {out_sg}")
    print(f"  Figures        : {fig1}, {fig2}, {fig3}")
    print("  Next: use these parquet files for XGBoost + SHAP.")


if __name__ == "__main__":
    if vector is None:
        warnings.warn("Optional package 'vector' not found. Helicity angles will be skipped if needed.")
    if Particle is None:
        warnings.warn("Optional package 'particle' not found. If no PX/PY/PZ/E, helicity angles will be skipped.")
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.", file=sys.stderr)
        sys.exit(130)
