# Medical Insurance Cost Prediction

Term project for **ENGR 202 — Data Science and AI**, Department of Computer Science, Fall 2024.  
Presented 29 November 2024.

The project predicts medical insurance charges from age, sex, BMI, number of children, smoker status, and region. The dataset has **1,137 records** and 7 columns. The label is `charges`.

## What the data needed

Exploration of `insurance_dataset.csv` found:

- 57 missing values in `region`
- Invalid negative values in `age` and `children`
- Inconsistent entries in `sex`
- A right-skewed `charges` column, with high outliers (up to about 63,770)
- More non-smokers than smokers, and a fairly even split across four regions

Smoking and higher BMI are associated with higher charges.

## Preprocessing

The data is split before cleaning so the test set stays untouched:

- 70% training
- 20% validation
- 10% test

Training preprocessing, saved and reused on validation and test:

- Impute missing values
- Remove outliers with the interquartile range and correct invalid entries
- Label-encode binary columns (`sex`, `smoker`)
- One-hot encode `region`
- Scale numeric features with `MinMaxScaler`

## Models

Several dense networks were trained in TensorFlow/Keras and compared with validation MAE and 5-fold cross-validation.

The first network matches the initial design: three hidden layers of 64, 32, and 16 units, ReLU activations, Adam, and 500 epochs.

The selected model is `model_rmsprop2_600`:

- Hidden layers: 512, 256, 128, 32, then a single output unit
- ReLU activations, no dropout
- RMSprop, batch size 32, mean absolute error
- 600 epochs

Training and validation MAE kept falling through 400 epochs, so training was extended to 600, where validation MAE leveled off. Cross-validation selected this model over deeper networks and dropout variants. It is saved as `trained_model.keras`.

## Repository layout

```
Final_Project_Insurance_dataset.ipynb   Full analysis, training, and model selection
Project_Test.ipynb                      Evaluate the saved model on the held-out test split
Real_data_Test.ipynb                    Score a new CSV with the saved preprocessing and model
docs/ENGR202-Medical-Insurance-Dataset.pdf
```

## How to run

Install the libraries, then open the notebooks from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`Final_Project_Insurance_dataset.ipynb` reads `insurance_dataset.csv` from the same folder. That CSV is not in this repository. The test notebooks also expect the files the training notebook writes:

- `X_test.pkl`, `y_test.pkl`
- `scaler.pkl`, `numerical_imputer.pkl`, `categorical_imputer.pkl`, `medians.pkl`
- `label_encoder_sex.pkl`, `label_encoder_smoker.pkl`, `one_hot_encoder_region.pkl`
- `remove_outliers.pkl`, `clean_categorical_typos.pkl`, `binary_encoding.pkl`, `one_hot_encoding.pkl`
- `trained_model.keras`

`Real_data_Test.ipynb` reads a new file named `filename.csv` with the same columns, including `charges`.
