# Kibria–Lukman Regularization in Extreme Learning Machines for Ill-Conditioned Models

This repository contains the R code used in the experimental study of Kibria–Lukman regularized extreme learning machines (KL-ELM and MKL-ELM) for ill-conditioned regression problems.

## Authors
- Murat Genç (Çukurova University)
- Adewale F. Lukman (University of North Dakota)
- Ömer Özbilen (Mersin University)

## Repository Structure
```text
.
├── README.md
├── code/
│   ├── funs-gh.R                 # ELM and regularized ELM functions
│   ├── funs_DUNN-gh.R            # Dunn post-hoc test function
│   └── main.R                    # Main experiment script
```

## Requirements
- R version 4.0 or higher
- The following R packages:
  - `MASS`
  - `dplyr`
  - `ggplot2`
  - `dunn.test`
  - `writexl`
  - `xtable`

Install them with:
```r
install.packages(c("MASS", "dplyr", "ggplot2", "dunn.test", "writexl", "xtable"))
```

## Running the Experiments
1. Clone this repository or download all files.
2. Set the working directory to the repository root (the folder containing `code/` and `data/`).
3. Run the script `code/main.R`. The following outputs will be generated:
   - `elm_results.csv`: summary of train/test RMSE, SD, reduction rate, lambda, and computation time.
   - `individual_test_rmse.csv`: individual test RMSE values for each trial.
   - `dunn_<dataset>_<n_hidden>.xlsx`: Dunn post-hoc test results.
   - Boxplots for 50 and 100 hidden neurons.


## Methods
The following estimators are implemented:
- `classic_elm`: standard extreme learning machine
- `ridge_elm`: ridge-regularized ELM (KHM tuning)
- `liu_elm`: Liu-regularized ELM (DOPT tuning)
- `aur_elm`: almost unbiased ridge ELM (KHM tuning)
- `kl_elm`: Kibria–Lukman ELM (HKM and KMIN tuning)
- `mkl_elm`: modified Kibria–Lukman ELM (HKM and KMIN tuning)

## Reproducibility
All experiments use 100 repeated random 70/30 train–test splits. Input weights are generated from a uniform distribution on [-1, 1] and regenerated independently for each repetition. The hyperbolic tangent function is used as the activation function. 

## License
This project is licensed under the GNU General Public License v3.0.
