import numpy as np, uproot, pandas as pd, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--infile",  required=True)
ap.add_argument("--outfile", required=True)
ap.add_argument("--mass-branch", default="mLcK")   # choose your pair: mLcK, mDstK, or mLcDst
ap.add_argument("--M", type=float, required=True)  # pole mass in GeV
ap.add_argument("--G", type=float, required=True)  # width in GeV
ap.add_argument("--mode", choices=["weights","thin"], default="weights")
ap.add_argument("--thin-scale", type=float, default=1.0, help="scale acceptance prob if needed")
args = ap.parse_args()

with uproot.open(args.infile) as f:
    t = f["DecayTree"]
    # Pull only the mass branch to build w(m); we can copy other branches later if needed
    m = t[args.mass_branch].array(library="np")  # GeV
    # Relativistic Breit–Wigner (constant width toy): w(m) ∝ 1 / [(m^2 - M^2)^2 + M^2 Γ^2]
    M2 = args.M**2
    w  = 1.0 / ((m*m - M2)**2 + (M2 * args.G*args.G))
    # Normalize for convenience
    w /= w.max()

    if args.mode == "weights":
        # Save event indices and weights to a lightweight file for training
        out = pd.DataFrame({"event": np.arange(len(m), dtype=np.int64), "weight": w})
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        out.to_parquet(args.outfile)
        print("Wrote weights to", args.outfile, "with max(w)=1")
    else:
        # Thinning: keep with prob = thin-scale * w, clipped to [0,1]
        p = np.clip(args.thin_scale * w, 0.0, 1.0)
        keep = np.random.random(size=len(p)) < p
        idx  = np.nonzero(keep)[0].astype(np.int64)
        out  = pd.DataFrame({"event": idx})
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        out.to_parquet(args.outfile)
        print("Wrote kept indices to", args.outfile, "kept", keep.sum(), "of", len(keep))