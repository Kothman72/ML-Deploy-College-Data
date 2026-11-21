# College Data Classification

An educational machine-learning project that predicts whether a college is
private (`Yes`) or public (`No`) from numeric college statistics. It includes a
Jupyter notebook for learning and evaluation, a saved scikit-learn pipeline, and
a small Streamlit app for trying predictions.

## What you can learn

- How to prepare a labeled dataset and separate features (`X`) from a target
  (`y`).
- Why a stratified train/test split preserves the class proportions.
- Why imputation and scaling belong inside a scikit-learn `Pipeline`.
- How to evaluate a binary classifier with accuracy, precision, recall, a
  confusion matrix, and cross-validation.
- How to save and reload a fitted pipeline with Joblib and use it in a simple
  app.

KMeans appears in the notebook as an example of unsupervised clustering. Since
this dataset already has `private` labels, the final evaluated and deployed
model is a supervised Logistic Regression classifier instead.

## Project files

- `data.csv` — college data; `private` is the target column and the other
  numeric columns are model features.
- `Untitled-1.ipynb` — data exploration, model training and evaluation, plus
  code to save the production pipeline and metadata.
- `model/college_private_pipeline.joblib` — fitted preprocessing and classifier
  pipeline used by the app.
- `model/metadata.json` — feature names, label mapping, default form values,
  and relevant software versions.
- `app.py` — Streamlit interface for entering college features and viewing a
  prediction.
- `requirements.txt` — pinned Python dependencies.

## Set up the environment

Use Python 3.13. With Conda installed:

```powershell
conda create --name college-data python=3.13
conda activate college-data
python -m pip install -r requirements.txt
```

In VS Code, open the notebook and select the `college-data` Python environment
as its kernel.

## Run the notebook

Run `Untitled-1.ipynb` from top to bottom. Its final cell evaluates the model on
a held-out test set, runs stratified five-fold cross-validation, refits the
pipeline on all rows with recognized target labels, and writes the model and
metadata into `model/`.

The pipeline handles missing numeric feature values with median imputation,
scales the features, and fits a class-weighted Logistic Regression model. The
test-set metrics are reproducible for the notebook's fixed random seed, but
they are estimates from this dataset and are not a guarantee of performance on
new colleges.

## Run the Streamlit app

From the project directory, with the environment activated:

```powershell
python -m streamlit run app.py
```

The app opens in a browser. It loads the saved pipeline and metadata from
`model/`, displays one input for each required numeric feature, and reports the
predicted college type and estimated probability of `Yes` (private). The
probability is a model estimate, not a guarantee.

If the files in `model/` are missing or you retrain the model, run the final
notebook cell again before starting the app.

## Use the saved pipeline in Python

The artifact is a scikit-learn pipeline, so it includes the imputer, scaler, and
classifier. Supply a pandas DataFrame with the feature columns listed in
`model/metadata.json`:

```python
import json
import joblib
import pandas as pd

with open("model/metadata.json", encoding="utf-8") as file:
    metadata = json.load(file)

pipeline = joblib.load("model/college_private_pipeline.joblib")
new_college = pd.DataFrame(
    [metadata["default_values"]],
    columns=metadata["feature_columns"],
)

prediction = pipeline.predict(new_college)[0]
label = metadata["target_mapping"][str(prediction)]
print(label)
```

Only load Joblib artifacts from trusted sources. For deployment, install the
versions in `requirements.txt` to match the environment used to save the
pipeline.
