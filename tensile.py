import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("🔬 Material Property Predictor")

uploaded_file = st.file_uploader("Upload CSV file with material properties", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", data.head())

    target_column = st.selectbox("Select target column to predict", data.columns)
    input_features = st.multiselect("Select feature columns", [col for col in data.columns if col != target_column])

    model = None

    if st.button("Train Model"):
        if not input_features:
            st.error("Please select at least one feature column.")
        else:
            X = data[input_features]
            y = data[target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            st.session_state.model = model
            st.session_state.features = input_features
            st.session_state.target = target_column
            st.success(f"Model trained successfully. R² score: {score:.2f}")

    # Prediction Section
    if "model" in st.session_state:
        st.write("### Predict New Sample")
        sample = []
        for feature in st.session_state.features:
            value = st.number_input(f"Enter value for {feature}", format="%.4f")
            sample.append(value)

        if st.button("Predict"):
            input_array = np.array(sample).reshape(1, -1)
            prediction = st.session_state.model.predict(input_array)
            st.write(f"🔍 Predicted {st.session_state.target}: **{prediction[0]:.2f}**")
