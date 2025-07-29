# batch_predict.py
import pandas as pd
import torch
import joblib
import numpy as np

# Load model & scaler
scaler = joblib.load("model/scaler.pkl")
model = ...  # same ANNModel
model.load_state_dict(torch.load("model/Ann_exam_score_model.pkl"))
model.eval()

# Load new data
new_data = pd.read_csv("input_students.csv", header=0)  # with column names
# Preprocess (same mappings)
# Predict
# Save to output.csv
