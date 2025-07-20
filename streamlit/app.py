import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

@st.cache_data
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🌸 Iris Species Classifier")
st.write("Choose measurements to predict the iris species.")

sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    features = np.array([[sl, sw, pl, pw]])
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    st.success(f"**Prediction:** {pred}")
    st.write("**Probabilities:**")
    st.write({species: f"{prob*100:.1f} %" for species, prob in zip(model.classes_, probs)})

    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    fig = px.scatter(df, x="petal_length", y="petal_width", color="species",
                     title="Petal length vs width")
    fig.add_scatter(x=[pl], y=[pw], mode="markers",
                    marker=dict(color="red", size=15), name="Your input")
    st.plotly_chart(fig)
