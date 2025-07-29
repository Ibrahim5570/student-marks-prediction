import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------
data = pd.read_csv("data/StudentPerformanceFactors.csv")

# Mapping dictionaries
lowmed_map = {'Low': 0, 'Medium': 1, 'High': 2}
yes_no_map = {'Yes': 1, 'No': 0}
school_type_map = {'Public': 0, 'Private': 1}
influence_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
education_map = {'None': 0, 'Primary': 1, 'High School': 2, 'College': 3, 'Postgraduate': 4}
distance_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
gender_map = {'Female': 1, 'Male': 0}

# Fill missing values
data['Teacher_Quality'] = data['Teacher_Quality'].fillna(data['Teacher_Quality'].mode()[0])
data['Parental_Education_Level'] = data['Parental_Education_Level'].fillna(data['Parental_Education_Level'].mode()[0])

# Apply mappings
data['Parental_Involvement'] = data['Parental_Involvement'].map(lowmed_map)
data['Access_to_Resources'] = data['Access_to_Resources'].map(lowmed_map)
data['Extracurricular_Activities'] = data['Extracurricular_Activities'].map(yes_no_map)
data['Motivation_Level'] = data['Motivation_Level'].map(lowmed_map)
data['Internet_Access'] = data['Internet_Access'].map(yes_no_map)
data['Family_Income'] = data['Family_Income'].map(lowmed_map)
data['Teacher_Quality'] = data['Teacher_Quality'].map(lowmed_map)
data['School_Type'] = data['School_Type'].map(school_type_map)
data['Peer_Influence'] = data['Peer_Influence'].map(influence_map)
data['Learning_Disabilities'] = data['Learning_Disabilities'].map(yes_no_map)
data['Parental_Education_Level'] = data['Parental_Education_Level'].map(education_map)
data['Distance_from_Home'] = data['Distance_from_Home'].map(distance_map)
data['Gender'] = data['Gender'].map(gender_map)

# Drop any remaining rows with NaN
data.dropna(inplace=True)

# Features and target
X = data.drop('Exam_Score', axis=1).values
y = data['Exam_Score'].values.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# -------------------------------
# 2. Define the Model
# -------------------------------
class ANNModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, optimizer
input_dim = X.shape[1]  # Should be 19
model = ANNModel(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 3. Train the Model
# -------------------------------
epochs = 1000
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# -------------------------------
# 4. Evaluate the Model
# -------------------------------
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

print("\n--- Evaluation on Test Set ---")
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Show first 20 predictions
comparison = pd.DataFrame({
    'Actual': y_test.flatten(),
    'Predicted': y_pred.flatten()
})
print("\nFirst 20 Actual vs Predicted Scores:")
print(comparison.head(20).to_string(index=False))

# -------------------------------
# 5. Save Model and Scaler
# -------------------------------
import os
os.makedirs("model", exist_ok=True)

torch.save(model.state_dict(), "model/Ann_exam_score_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model and scaler saved successfully!")
print("→ Model: model/Ann_exam_score_model.pkl")
print("→ Scaler: model/scaler.pkl")