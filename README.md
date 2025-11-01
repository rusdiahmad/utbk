# UTBK Streamlit Portfolio

This project is a Streamlit portfolio app built to satisfy the "Portfolio Building with Streamlit" assignment.
It uses a UTBK dataset and includes a simple logistic regression model for predicting "lulus" (pass).

## Files
- app.py : main Streamlit app
- model/utbk_model.pkl : pretrained model pipeline (if available)
- pipeline.py : helper to retrain the model locally
- requirements.txt : Python dependencies

## How to run
1. Create virtualenv and install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run app:
   ```
   streamlit run app.py
   ```