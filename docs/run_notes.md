# RapidSim run log

## Λb⁰ → Λc⁺ D*⁰ K⁻ (phase-space generation)

### Command used
```bash
conda activate lb_lcDstarK
RapidSim.exe configs/Lb_LcDst0K 300000 1
mv configs/Lb_LcDst0K_tree.root data/rapidsim/Lb_LcDst0K_tree.root

## Resonant reweighting (m(Λc⁺D*⁰))

### Command used
```bash
python scripts/make_resonant.py
--infile data/rapidsim/Lb_LcDst0K_phase.root
--outfile data/rapidsim/Lb_LcDst0K_w_mLcDst.parquet
--mass-branch mLcDst --M 4.44 --G 0.04 --mode weights

### Details
- Input: phase-space Λb⁰ → Λc⁺ D*⁰ K⁻ sample (300k events)
- Resonance injected: m(Λc⁺D*⁰), M = 4.44 GeV, Γ = 0.04 GeV
- Output: weights file `data/rapidsim/Lb_LcDst0K_w_mLcDst.parquet`
- Purpose: emulate narrow pentaquark-like signal for ML classification