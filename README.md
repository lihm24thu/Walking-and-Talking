# Walking and Talking: Neural Representations of Speech under In-bed and Freely-moving Settings

Code repository for:

"**Speech-related Neural Representations Are Modulated by Behavioral Contexts while Maintaining Population-level Organization**"

This repository contains the analysis and modeling code used in the manuscript submitted to **Nature Communications**.

<img width="822" height="639" alt="截屏2026-08-12 16 40 02" src="https://github.com/user-attachments/assets/6cf1affd-fd64-4987-ae27-f60a21c83556" />


## Reproducing the main figures

| Figure | Analysis and Visualization | Script / Notebook |
|---|---|---|
| Fig. 2 | A neckband wireless neural recording platform for multiple behavior contexts | `figures/` |
| Fig. 3 | Substantial local neural variability across behavioral contexts for different brain regions and participants revealed by single-channel linear decoding | `figures/figure_3.py`, `src/linear_decoding.py` |
| Fig. 4 | Consistent population-level neural organization across behavioral contexts revealed by deep learning | `figures/figure_4.py`, `src/models.py`, `src/Run_detection_model.py`, `src/Run_mel_decoder.py` |
| Fig. 5 | Spatial mapping of cross-context consistency reveals distributed cortical networks with preserved functional organization | `figures/figure_5.py`, `src/SCI_define.py` |
| Fig. 6 | Shared neural subspace underlying speech representations across contexts and brain regional contributions | `figures/` |

## Data

The data that support the findings of this study are available on request from the corresponding author. Data are not publicly available due to ethical restrictions. Requests will be considered on a case-by-case basis, and data access may require signing a data access agreement.
