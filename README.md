# Biometric-based on EEG

This repository corresponds to the code used to develop the study presented in the article entitled "Analysis of Factors that Influence the Performance of Biometric Systems Based on EEG Signals."

**Authors**:

- Dustin Carrión-Ojeda (dustin.carrion@gmail.com)
- Rigoberto Fonseca-Delgado (rfonseca@inaoep.mx)
- Israel Pineda (ipineda@yachaytech.edu.ec)

## Repository organization

This repository is made up of eleven python scripts and two folders containing the data and results. The scripts can be divided into three categories:

1. Data Preparation.
2. Hyperparameter Optimization.
3. Experiment.

On the other hand, the folders [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and [BIOMEX-DB](https://inaoe.repositorioinstitucional.mx/jspui/handle/1009/1604) correspond to the datasets used in this study, and each one will have three subfolders:

1. `data` Contains all the information necessary for the study.
2. `hyperparameter_optimization` Contains the specific data used for hyperparameter optimization and its results.
3. `experiment` Contains the specific data used for the experiment and its results.

Each of the scripts categories and how to use them are detailed below.

**Note**: The order of execution of the scripts corresponds to the order presented below.

### Data Preparation

This category is composed of the following scripts:

1. `create_feature_matrices.py`
2. `split_feature_matrices.py`

The first script is responsible for preprocessing the electroencephalograms (EEG) using the discrete wavelet transform (DWT), and also it generates the feature matrices. These matrices are composed of the relative wavelet energy (RWE) of all detail coefficients and the last approximation coefficient. To run this script, it is necessary to download the datasets used in this study.

The DEAP dataset download process is detailed on its [page](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html). This study used the preprocessed version of this dataset. In the case of BIOMEX-DB, as it is not publicly accessible, to be able to access it, it must be requested directly from its [authors](https://eegbiometryexperiment.blogspot.com/2019/08/edf-content.html).

Once the data is downloaded, the following command can be executed:
```bash
python create_feature_matrices.py input_data_path dataset_name
```

`input_data_path` corresponds to the relative path to the folder where the data downloaded from each of the databases is located, while` dataset_name` can only take two values: `deap` or` biomex-db`.

After the feature matrices are generated, they must be divided into data for hyperparameter optimization and data for the experiment. The percentage of data used for optimization was 20%, while the remaining 80% was used in the experiment. To perform this division, the following command is used:

```bash
python split_feature_matrices.py dataset_name
```
**Note:** Due to possible complications getting the original datasets, the repository provides the feature matrices (result of `create_feature_matrices.py`). For this reason, the script `split_feature_matrices.py` can be run without any problem.

### Hyperparameter Optimization

This study uses six classifiers:

- Support Vector Machine (SVM)
- K-nearest Neighbors (KNN)
- Random Forest (RF)
- AdaBoost (AB)
- Gaussian Naïve Bayes (GNB)
- Multilayer Perceptron (MLP)

To obtain the best results, greedy search optimization was applied based on a ten-fold-cross validation with overlapping between folds to find the best hyperparameters for each classifier. For running the scripts in this category, it is necessary to have run the **Data Preparation** scripts. All hyperparameter optimization scripts can be executed with the following command:

```bash
python run_optimization.py
```

This command is equivalent to running the following:

```bash
python create_optimization_fold_files.py deap
python create_optimization_fold_files.py biomex-db
python grid_search.py deap
python grid_search.py biomex-db
python grid_search_neural_network.py deap
python grid_search_neural_network.py biomex-db
python read_optimization_results.py deap
python read_optimization_results.py biomex-db
```

The functionality of each script is detailed below:

- `create_optimization_fold_files.py` Generates the training and testing data of each fold.
- `grid_search.py` Executes hyperparameter optimization for SMV, KNN, RF, AB, and GNB. This script generates two `.sav` files for each classifier. One file corresponds to the evaluated hyperparameters, and the other contains the average accuracy reached by the classifier using these hyperparameters.
- `grid_search_neural_network.py` The functionality of this script is the same as the previous one, but in this case, the classifier is MLP.
- `read_optimization_results.py` Reads the results generated with` grid_search.py` and `grid_search_neural_network.py` and selects the best set of hyperparameters for each classifier based on the accuracy achieved.

### Experiment

The objective of the experiment was to assess the impact of the duration of EEG recordings and the levels of decomposition of the DWT on the performance of the classifiers.  In both datasets, each signal was segmented into the following times:  0.25, 0.5, 0.75, 1, 1.25,1.5, 1.75, 2, 2.25, and 2.5 seconds.  To simulate the differences that may exist between the recordings in a real scenario, the start of the segmentation was randomly taken. This work uses ten-fold-cross validation to increase the reliability of the experimental results. The performance metrics used to evaluate the classifiers are the  Average  accuracy  *(Acc)*,  Macro-averaging  Sensitivity  *(Se)*,  and  Macro-averaging Specificity *(Sp)* :

![equation](https://latex.codecogs.com/gif.latex?Acc%3D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%7B%5Cfrac%7BTp_i&plus;Tn_i%7D%7BTp_i&plus;Fn_i&plus;Fp_i&plus;Tn_i%7D%7D%5Cright%29/l)

![equation](https://latex.codecogs.com/gif.latex?Se%3D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%7B%5Cfrac%7BTp_i%7D%7BTp_i&plus;Fn_i%7D%7D%5Cright%29/l)

![equation](https://latex.codecogs.com/gif.latex?Sp%3D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7Bl%7D%7B%5Cfrac%7BTn_i%7D%7BTn_i&plus;Fp_i%7D%7D%5Cright%29/l)

To run the scripts in this category, it is necessary to have run the **Data Preparation** scripts. The execution of the **Hyperparameter Optimization** scripts is not required because it was already executed during the development of the study. Thus, the **Experiment** scripts were coded with the best set of hyperparameters for each classifier. All experiment scripts can be executed with the following command:

```bash
python run_experiment.py
```

This command is equivalent to running the following:

```bash
python create_experiment_fold_files.py deap
python create_experiment_fold_files.py biomex-db
python k_cross.py deap
python k_cross.py biomex-db
python k_cross_neural_network.py deap
python k_cross_neural_network.py biomex-db
```
The functionality of each script is detailed below:

- `create_experiment_fold_files.py` Generates the training and testing data for each fold.
- `k_cross.py` Executes ten-fold-cross validation for SMV, KNN, RF, AB, and GNB classifiers with each time segment. This script generates three `.xlsx` files for each time segment. Each file corresponds to a metric (accuracy, sensitivity, and specificity) and contains the results of all classifiers at each fold.
- `k_cross_neural_network.py` The functionality of this script is the same as the previous one, but in this case, the classifier is MLP.
