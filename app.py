import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model" / "college_private_pipeline.joblib"
METADATA_PATH = PROJECT_DIR / "model" / "metadata.json"

st.set_page_config(page_title="College Type Prediction", page_icon="🎓")
st.title("College Type Prediction")
st.write("Enter the college statistics to predict whether it is private.")

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    st.error(
        "Model files are missing. Run the final training cell in Untitled-1.ipynb "
        "to create the model folder and artifacts."
    )
    st.stop()

with METADATA_PATH.open(encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)


@st.cache_resource
def load_pipeline(model_path: str):
    return joblib.load(model_path)


model = load_pipeline(str(MODEL_PATH))
feature_columns = metadata["feature_columns"]
model_features = model.feature_names_in_.tolist()
if model_features != feature_columns:
    raise ValueError("The model feature schema does not match model/metadata.json.")

default_values = metadata.get("default_values", {})
percentage_features = {
    "top10perc",
    "top25perc",
    "phd",
    "terminal",
    "perc_alumni",
    "grad_rate",
}

with st.form("college_features"):
    st.subheader("College statistics")
    input_columns = st.columns(2)
    feature_values = {}

    for index, feature in enumerate(feature_columns):
        column = input_columns[index % len(input_columns)]
        label = feature.replace("_", " ").title()
        default_value = float(default_values.get(feature, 0))

        if feature == "s_f_ratio":
            feature_values[feature] = column.number_input(
                label,
                min_value=0.0,
                value=default_value,
                step=0.1,
                format="%.1f",
            )
        else:
            maximum = 100 if feature in percentage_features else None
            feature_values[feature] = column.number_input(
                label,
                min_value=0,
                max_value=maximum,
                value=min(int(round(default_value)), maximum)
                if maximum is not None
                else max(int(round(default_value)), 0),
                step=1,
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame([feature_values], columns=feature_columns)
    prediction = int(model.predict(input_data)[0])
    positive_class_index = list(model.classes_).index(1)
    private_probability = float(
        model.predict_proba(input_data)[0, positive_class_index]
    )
    predicted_label = metadata["target_mapping"][str(prediction)]

    st.subheader("Prediction")
    st.metric("College type", predicted_label)
    st.metric("Probability of being private", f"{private_probability:.1%}")
    st.caption("The probability is a model estimate, not a guarantee.")
