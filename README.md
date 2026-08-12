# Walking and Talking: Neural Representations of Speech under In-bed and Freely-moving Settings

Code repository for:

"**Speech-related Neural Representations Are Modulated by Behavioral Contexts while Maintaining Population-level Organization**"

This repository contains the analysis and modeling code used in the manuscript submitted to **Nature Communications**.

## Reproducing the main figures

| Figure | Analysis and Visualization | Script / Notebook |
|---|---|---|
| Fig. 2 | A neckband wireless neural recording platform for multiple behavior contexts | `figures/` |
| Fig. 3 | Substantial local neural variability across behavioral contexts for different brain regions and participants revealed by single-channel linear decoding | `figures/figure_3.py`, 'src/linear_decoding.py' |
| Fig. 4 | Consistent population-level neural organization across behavioral contexts revealed by deep learning | `figures/figure_4.py`, 'src/models.py', 'src/Run_detection_model.py', 'src/Run_mel_decoder.py' |
| Fig. 5 | Spatial mapping of cross-context consistency reveals distributed cortical networks with preserved functional organization | `figures/figure_5.py`, 'SCI_define.py' |
| Fig. 6 | Shared neural subspace underlying speech representations across contexts and brain regional contributions | `figures/` |

## Data

Raw neural recordings are not included in this repository because of patient privacy and data-sharing restrictions.
The data that support the findings of this study are available on request from the corresponding author. Data are not publicly available due to ethical restrictions. Requests will be considered on a case-by-case basis, and data access may require signing a data access agreement.
