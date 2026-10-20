# From benchmark to field: weather-hazard image recognition (code and data)

Code, split files and result tables for the manuscript
*"From benchmark to field: auditing image-based recognition of agricultural weather hazards on a real frost-damage event"*
(Chandiraprakash N and A. Chinnasamy, submitted to *Biosystems Engineering*).

Every table and figure in the paper can be reproduced from the public datasets with the notebooks below.
The notebooks were run on Kaggle (2 × NVIDIA T4, Python 3.10, PyTorch 2.x, timm).

## Datasets (not redistributed)
| Dataset | Source | Used for |
|---|---|---|
| WEAPD (6,862 images, 11 classes) | Kaggle `jehanbhathena/weather-dataset`; original: github.com/haixiaxiao/A-database-WEAPD | Duplicate audit, benchmark, conformal prediction |
| PlantDoc (classification) | Singh et al. (2020) | Generality of the duplicate audit |
| PlantVillage (colour) | Hughes & Salathé (2015) | Generality of the duplicate audit |
| EV3: frost-damaged and healthy grapevines (909 images, Manisa, Turkey, April 2025) | Arık (2025), IEEE DataPort, doi:10.21227/wns0-np29; version used: Roboflow Universe "Grape-Segmentation – yedek – kendi verim", CC BY 4.0 | Independent field validation |

## Folder layout
```
splits/     leakage-free WEAPD split, EV3 image labels and EV3 pool/calibration/test split
notebooks/  all experiments (see order below)
scripts/    statistics and conformal analysis run on the result files
results/    result tables (CSV) behind every table and figure
```

## Order of the experiments
| Step | Notebook / script | Paper section | Output |
|---|---|---|---|
| 1 | `notebooks/weapd_leakage_experiment` and `results/neardup/*.py` | 2.2, 3.1 | WEAPD hashes, duplicate groups, cross-label rate by distance band |
| 2 | `notebooks/n4v2_calibrated_duplicate_audit.ipynb` | 2.2, 3.1 | PlantDoc / PlantVillage audit (Table 4, Fig. 3, Fig. 4) |
| 3 | `notebooks/weapd_baselines_v2_nodup` + `weapd_v2_seeds234` | 2.3, 3.2 | 7 baselines × 5 seeds on the leakage-free split |
| 4 | `notebooks/weapd_c3_round2`, `weapd_c3_round3_5seeds` | 2.3, 3.2 | Swin-T + KAN and DINOv2 variants × 5 seeds |
| 5 | `scripts/stats_corrected.py` | 2.4, 3.2 | Majority-vote exact McNemar, bootstrap CI, Holm (Table 6, Fig. 5, Table S1) |
| 6 | `scripts/action_conformal.py` | 2.5, 3.3 | Class- and action-level conformal sets (Table 7, Fig. 6) |
| 7 | `notebooks/ev3_frost_grape_external_validation.ipynb` | 2.6, 3.4 | Zero-shot, few-shot probe, binary conformal, cost thresholds (Tables 8–9, Figs. 7–9) |
| 8 | `notebooks/ev3_finetune_no_cal_reuse.ipynb` | 2.6, 3.4 | Fine-tuning without use of the calibration set (Table 9) |
| 9 | `notebooks/gradcam_ev3_weapd.ipynb` | 2.7, 3.5 | Grad-CAM attention ratio (Table 10, Fig. 10, Fig. S1) |

## Data hygiene
* WEAPD: duplicate groups (MD5 + pHash, Hamming distance ≤ 8) are never split across training, validation and test (`splits/weapd_leakage_free_split.csv`, column `group`).
* EV3: pool (454), calibration (184) and test (271) sets are disjoint and keep near-duplicate groups together (`splits/ev3_split.csv`).
  The calibration set is used only for conformal and cost-threshold calibration; the test set is evaluated once.
* Seeds: WEAPD split seed 42, training seeds 0–4; EV3 seeds 0–2.

## Licence
Code: MIT (see `LICENSE`). Split files and result tables: CC BY 4.0. The images remain under the licences of their original datasets.

## Citation
See `CITATION.cff`.
