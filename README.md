# Sobol-ADR
Files supporting the paper "Global Sensitivity Analysis of Reactive Transport Models in Groundwater Systems: Controls at Early Breakthrough, Maximum Concentration, and Late-Time Tailing"

## Requirements
Numpy, pandas, scipy, mpmath, SALib, seaborn, matplotlib

# Steps
- All results files supporting the manuscript are located in the following Zenodo archive:
- If you would like to explore the results previously generated, download the "Results" folder from Zenodo and copy it to the results directory in your clone of this repository. You should be able to generate all of the figures using the scripts in the figure_scripts folder, and perform additional analysis as desired.
- The same set of results can be generated via the scripts in the folder "sensitivity_analyses". SA_full_range.py performs the sensitivity analysis as outlined in Experiment 1, while the other four scripts in the "sensitivity_analyses" folder perform the analysis in Experiment 2. Runtime will vary by sample size and individual computer specifications.
- If you would like to perform your own sensitivity analysis, with different parameter ranges or static parameter inputs, please follow the guidance outlined in the notebook: experiment_tutorial.
