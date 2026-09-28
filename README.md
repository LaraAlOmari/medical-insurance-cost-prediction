# Medical Insurance Cost Prediction

Term project for **ENGR 202 — Data Science and AI**, Department of Computer Science, Fall 2024.  
Presented 29 November 2024.

**Author:** Lara Al Omari

## Problem

The goal is to predict the medical insurance charge for a person from demographic and lifestyle information. The inputs are age, sex, BMI, number of children, smoker status, and region. The label is `charges`, a dollar amount, so this is a regression problem. Performance is measured with mean absolute error. There is no class label and no confusion matrix.

## Dataset

The project uses a medical insurance table with **1,137 records** and 7 columns. The file is `notebooks/insurance_dataset.csv`.

Exploration of that file found:

- 57 missing values in `region`
- Invalid negative values in `age` and `children`
- Inconsistent text in `sex`
- A right-skewed `charges` column, with high outliers up to about 63,770
- More non-smokers than smokers
- Four regions with a fairly even count

Smokers and people with higher BMI tend to have higher charges. Age and BMI are positively associated with charges.

## Methods

The notebook splits the rows before cleaning, so the test set is not used to fit the cleaners:

- 70% training
- 20% validation
- 10% test

Preprocessing fit on the training split, then applied to validation and test:

- Impute missing values
- Remove outliers with the interquartile range and correct invalid entries
- Label-encode `sex` and `smoker`
- One-hot encode `region`
- Scale features with `MinMaxScaler`

Several dense networks were trained in TensorFlow/Keras. The first network has hidden layers of 64, 32, and 16 units, ReLU activations, Adam, and 500 epochs.

Later networks use RMSprop, batch size 32, and mean absolute error as the loss. The configuration kept in the project is `model_rmsprop2_600`:

- Hidden layers of 512, 256, 128, and 32, then one linear output
- No dropout
- 600 epochs

Training MAE was still falling at 400 epochs, so training was extended to 600, where validation MAE leveled off. The candidates were compared with 5-fold cross-validation. The saved file name in the notebook is `trained_model.keras`.

## Results

Five-fold cross-validation on the training split, from the notebook outputs:

| Model | Epochs | Average MAE | MAE spread |
| --- | --- | --- | --- |
| RMSprop, 4 hidden layers | 400 | 1,937.14 | ± 237.46 |
| RMSprop, 4 hidden layers (`model_rmsprop2_600`) | 600 | 1,886.83 | ± 159.42 |
| RMSprop, 5 hidden layers | 500 | 1,864.05 | ± 192.24 |
| RMSprop, 5 hidden layers, dropout 0.1 | 500 | 1,803.82 | ± 320.71 |

The notebook and the course slides select `model_rmsprop2_600`. Its average MAE is 1,886.83 dollars, and the fold-to-fold spread is smaller than the deeper dropout network. The model is saved for the test notebooks.

Plots from the notebook are in `results/`:

- `charges_histogram.png` and `charges_boxplot.png` show the skewed label and high outliers.
- `bmi_vs_charges.png` and `smoker_vs_charges.png` show the two strongest cost patterns.
- `correlation_before_cleaning.png` and `correlation_after_cleaning.png` compare feature relationships.
- `charges_after_cleaning.png` shows the label after preprocessing.
- `adam_3layer_mae.png` is the 64–32–16 Adam network.
- `selected_model_mae.png` is the training and validation MAE of the selected 600-epoch model.

## Contribution

Lara Al Omari cleaned the insurance table, compared Adam and RMSprop networks, ran the cross-validation, and selected the 600-epoch RMSprop model. The slide report is `docs/ENGR202-Medical-Insurance-Dataset.pdf`. The training notebook, the held-out test notebook, and the notebook for a new CSV are under `notebooks/`.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`notebooks/insurance_dataset.csv` is already next to `Final_Project_Insurance_dataset.ipynb`. Start Jupyter from the `notebooks` folder and run that notebook from top to bottom. It writes the test split, the preprocessing objects, and `trained_model.keras` into the same folder:

- `X_test.pkl`, `y_test.pkl`
- `scaler.pkl`, `numerical_imputer.pkl`, `categorical_imputer.pkl`, `medians.pkl`
- `label_encoder_sex.pkl`, `label_encoder_smoker.pkl`, `one_hot_encoder_region.pkl`
- `remove_outliers.pkl`, `clean_categorical_typos.pkl`, `binary_encoding.pkl`, `one_hot_encoding.pkl`
- `trained_model.keras`

Then run `notebooks/Project_Test.ipynb` in that same folder. It loads those files and prints test MAE.

`notebooks/Real_data_Test.ipynb` scores a new file. Name that file `filename.csv`, give it the same columns including `charges`, and place it in `notebooks/` with the saved `.pkl` files and `trained_model.keras`.

## Repository layout

```
notebooks/insurance_dataset.csv
notebooks/Final_Project_Insurance_dataset.ipynb
notebooks/Project_Test.ipynb
notebooks/Real_data_Test.ipynb
docs/ENGR202-Medical-Insurance-Dataset.pdf
results/
requirements.txt
.gitignore
```
