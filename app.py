import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.exceptions import NotFittedError

st.set_page_config(page_title="My Portfolio with Streamlit", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Projects", "Data Viz", "Predict"])

if page == "About":
    st.title("My Portfolio with Streamlit")
    st.markdown("""
    - **Name:** Rusdi Ahmad
    - **Background:** S2 Matematika UNAND, Guru Matematika
    - **Skills:** Data Analysis, Machine Learning, Streamlit
    """)
    st.write("This portfolio uses the UTBK dataset and includes a simple prediction pipeline.")

elif page == "Projects":
    st.header("Projects")
    st.write("1. Analysis of UTBK Scores — EDA and prediction")
    st.write("2. Streamlit Portfolio App")
    st.write("3. Data Science Teaching Materials")
    st.write("You can add images or buttons here.")

elif page == "Data Viz":
    st.header("Data Visualization")
    uploaded = st.file_uploader("Upload a CSV or Excel file for visualization", type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if str(uploaded.name).lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.write("Dataset preview:", df.head())
            st.write("Numeric description:")
            st.write(df.describe())
            st.bar_chart(df.select_dtypes(include=[np.number]).iloc[:, :5].mean())
        except Exception as e:
            st.error("Error reading file: " + str(e))

elif page == "Predict":
    st.header("Predict Lulus/Tidak (simple model)")
    st.write("Upload dataset CSV/Excel in the same format used for training (numeric features).")
    uploaded = st.file_uploader("Upload data for prediction", type=['csv','xlsx'])
    model_path = "model/utbk_model.pkl"
    if uploaded is not None:
        try:
            if str(uploaded.name).lower().endswith('.csv'):
                data = pd.read_csv(uploaded)
            else:
                data = pd.read_excel(uploaded)
            st.write("Preview:", data.head())
            # load model
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                # select numeric cols used in model
                num_cols = model.named_steps['preprocessor'].transformers[0][2]
                X = data[num_cols]
                preds = model.predict(X)
                data['pred_lulus'] = preds
                st.write(data.head())
                st.success("Predictions added as column 'pred_lulus'.")
            except FileNotFoundError:
                st.error("Model file not found. Make sure 'model/utbk_model.pkl' exists.")
            except NotFittedError:
                st.error("Model not fitted.")
            except Exception as e:
                st.error("Prediction error: " + str(e))
        except Exception as e:
            st.error("File read error: " + str(e))