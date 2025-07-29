import torch
import numpy as np
import joblib
import os

# -------------------------------
# 1. Define Model Architecture (must match training)
# -------------------------------
class ANNModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------------
# 2. Mapping Dictionaries (must match training)
# -------------------------------
lowmed_map = {'Low': 0, 'Medium': 1, 'High': 2}
yes_no_map = {'Yes': 1, 'No': 0}
school_type_map = {'Public': 0, 'Private': 1}
influence_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
education_map = {'None': 0, 'Primary': 1, 'High School': 2, 'College': 3, 'Postgraduate': 4}
distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
gender_map = {'Female': 1, 'Male': 0}

# Reverse maps for display
rev_education = {v: k for k, v in education_map.items()}
rev_distance = {v: k for k, v in distance_map.items()}

# -------------------------------
# 3. Load Model and Scaler
# -------------------------------
model_path = "model/Ann_exam_score_model.pkl"
scaler_path = "model/scaler.pkl"

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    print("Make sure you've trained the model and saved it as 'model/Ann_exam_score_model.pkl'")
    exit()

if not os.path.exists(scaler_path):
    print(f"❌ Scaler file not found: {scaler_path}")
    exit()

try:
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading scaler:", e)
    exit()

input_dim = 19
model = ANNModel(input_dim=input_dim)
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# -------------------------------
# 4. Feature Definitions & Prompts
# -------------------------------
features = [
    {"name": "Age", "type": "float", "prompt": "Enter student's age"},
    {"name": "Hours_Studied", "type": "float", "prompt": "Enter hours studied per week"},
    {"name": "Parental_Involvement", "type": "cat", "map": lowmed_map, "options": "Low/Medium/High"},
    {"name": "Access_to_Resources", "type": "cat", "map": lowmed_map, "options": "Low/Medium/High"},
    {"name": "Extracurricular_Activities", "type": "cat", "map": yes_no_map, "options": "Yes/No"},
    {"name": "Sleep_Hours", "type": "float", "prompt": "Enter average sleep hours per night"},
    {"name": "Previous_Scores", "type": "float", "prompt": "Enter previous exam score (0-100)"},
    {"name": "Motivation_Level", "type": "cat", "map": lowmed_map, "options": "Low/Medium/High"},
    {"name": "Internet_Access", "type": "cat", "map": yes_no_map, "options": "Yes/No"},
    {"name": "Tutoring_Sessions", "type": "float", "prompt": "Enter weekly tutoring sessions"},
    {"name": "Family_Income", "type": "cat", "map": lowmed_map, "options": "Low/Medium/High"},
    {"name": "Teacher_Quality", "type": "cat", "map": lowmed_map, "options": "Low/Medium/High"},
    {"name": "School_Type", "type": "cat", "map": school_type_map, "options": "Public/Private"},
    {"name": "Peer_Influence", "type": "cat", "map": influence_map, "options": "Positive/Negative/Neutral"},
    {"name": "Physical_Activity", "type": "float", "prompt": "Enter physical activity hours per week"},
    {"name": "Learning_Disabilities", "type": "cat", "map": yes_no_map, "options": "Yes/No"},
    {"name": "Parental_Education_Level", "type": "cat", "map": education_map, "options": "None/Primary/High School/College/Postgraduate"},
    {"name": "Distance_from_Home", "type": "cat", "map": distance_map, "options": "Near/Moderate/Far"},
    {"name": "Gender", "type": "cat", "map": gender_map, "options": "Male/Female"}
]

print("\n" + "="*50)
print("🎓 STUDENT EXAM SCORE PREDICTION")
print("="*50)
print("Enter the following details. Options are shown in brackets.\n")

user_input = []

for feat in features:
    while True:
        if feat["type"] == "float":
            prompt = f"{feat['prompt']} (e.g., 15): "
        else:
            prompt = f"{feat['name']} [{feat['options']}]: "

        value = input(prompt).strip()

        if not value:
            print(f"❌ Input required. Please enter a value.")
            continue

        try:
            if feat["type"] == "float":
                val = float(value)
                if val < 0:
                    raise ValueError
                user_input.append(val)
                break

            elif feat["name"] == "Parental_Education_Level":
                if value not in education_map:
                    print(f"❌ Invalid input. Choose from: {feat['options']}")
                    continue
                user_input.append(education_map[value])
                break

            elif feat["name"] == "Distance_from_Home":
                if value not in distance_map:
                    print(f"❌ Invalid input. Choose from: {feat['options']}")
                    continue
                user_input.append(distance_map[value])
                break

            elif feat["name"] == "Gender":
                if value not in gender_map:
                    print(f"❌ Invalid input. Choose from: {feat['options']}")
                    continue
                user_input.append(gender_map[value])
                break

            else:
                # Other categorical mappings
                if 'Yes' in feat["map"] and value in ['Yes', 'No']:
                    user_input.append(feat["map"][value])
                    break
                elif value in feat["map"]:
                    user_input.append(feat["map"][value])
                    break
                else:
                    print(f"❌ Invalid input. Choose from: {feat['options']}")
                    continue

        except ValueError:
            print(f"❌ Invalid format. Please enter a valid number or option.")

# -------------------------------
# 5. Make Prediction
# -------------------------------
input_array = np.array([user_input], dtype=np.float32)
input_scaled = scaler.transform(input_array)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    predicted_score = model(input_tensor).item()

# Clamp score between 0 and 100
predicted_score = max(0, min(100, predicted_score))

print("\n" + "-"*40)
print(f"🎉 Predicted Exam Score: {predicted_score:.2f}")
print("-"*40)