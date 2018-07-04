This is the accompany code for the paper "A machine learning approach to preterm infants survival estimation: the PISA algorithm".

# Data
We used two datasets:
- Data containing infants born in INN centers within 2008-2014 (the training set).
- Data containing infants born in INN centers within 2015-2016 (the test set).

For privacy concerns, data is not available.

# Experiments

The experiments are basically two:
- firstly, there is a model selection phase where we searched for an optimal classifier for the problem using only the training data. We selected this best model from a pool of candidate models, which include Logistic Regression, K-Nearest Neighbors, Random Forests, Gradient Boosting Machines, Support Vector Machines and Neural Networks.
- secondly, we compared the best model with various benchmark models, including:
  - a Logistic Regression model that uses only birth weight as predictor;
  - a Logistic Regression model that uses birth weight + gestational age as predictor;
  - a Logistic Regression model from the study by Manktelow et al.;
  - a Logistic Regression model from the study by Tyson el al.;
  - a Logistic Regression model that uses the same predictors as the model used by VON;
  - the candidate Logistic Regression model which was trained in the previous phase.

As regards the second experiment, all the models were evaluated on the test set (2015-2016), which was not used for training, in order to understand the generalization capabilities of the selected model with respect to the benchmarks on unseen data.

We evaluated the resulting models using the ROC AUC scoring metric, as well as the Brier loss (for goodness of fit).
The evaluation was performed on:
1. the full test set;
and some subsets of interest (scenarios), such as:
2. infants born within the 25th gestational week, whose weight is above and included 400 g and below and included 999 g;
3. infants whose birth weight is above and included 1000 g and below and included 1500 g;
4. infants born within the 23rd and the 32nd (included) gestational weeks, who are singletons (i.e. not twins).

# Code Notation

If you skim through the files, be aware of some notation which helps you understand code and results.

Models in the code and in the results tables are referred to by identifying strings:
- `bw` is the benchmark model built with birth weight alone;
- `bwga` is the benchmark model built with birth weight and gestational age;
- `logreg` is the model built using the same preditors as the VON model;
- `mankt` is the model by Manktelow et al.;
- `tyson` is the model by Manktelow et al.;
- `lr` is the candidate Logistic Regression model;
- `knn` is the candidate K-Nearest Neighbor model;
- `rf` is the candidate Random Forest;
- `xgb` is the candidate Gradient Boosting Machine;
- `svm` is the candidate Support Vector Machine;
- `nn` is the candidate Neural Network;

The same goes for each evaluation scenario:
- `full` is the identifier of scenario 1
- `elbwi` (standing for Extremely Low Birth Weight Infants) is the identifier of scenario 2
- `vlbwi` (standing for Very Low Birth Weight Infants) is the identifier of scenario 3
- `singletons` is the identifier of scenario 4.


# Code files of interest in the repository

- file `train_models.py` is used to perform the model selection on both the candidate models and the benchmark models that were trained from scratch.
- file `predict_models.py` generates predictions for each evaluated model.
- file `analyze_results.py` computes both ROC AUC and Brier loss scores for the evaluated models in every analyzed scenarios.
- file `plot_figures.py` generates Figure 1 of the paper.
- file `utils/grids.py` contains the grids of hyper-parameters that were searched during model selection, plus some additional information about the model selection phase itself.