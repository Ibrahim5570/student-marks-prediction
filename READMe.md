# 📊 Student Exam Score Prediction using PyTorch ANN

A machine learning model that predicts student exam scores based on 19 socio-academic and behavioral factors using an Artificial Neural Network (ANN) built with **PyTorch**.

This project demonstrates end-to-end workflow: data preprocessing, model training, evaluation, and real-time prediction.

---

## 🚀 Features

- Trains a custom ANN to predict continuous exam scores (regression).
- Uses real-world student performance data with categorical & numerical features.
- Preprocesses data with `StandardScaler` and proper encoding.
- Saves trained model and scaler for reuse.
- Interactive prediction script for real-time inference.
- Clean, modular, and well-documented code.

---

## 📁 Project Structure
Student-Performance-Predictor/
│
├── data/
│ └── StudentPerformanceFactors.csv ← Input dataset
│
├── model/
│ ├── Ann_exam_score_model.pkl ← Trained PyTorch model
│ └── scaler.pkl ← Fitted StandardScaler
│
├── notebooks/
│ ├── train_student_model.ipynb ← Training notebook
│ └── predict_exam_score.ipynb ← Prediction notebook
│
├── train_student_model.py ← Training script
├── predict_exam_score.py ← Prediction script
├── README.md ← This file
└── requirements.txt ← Python dependencies

---

## ⚙️ Setup & Installation

### 1. Clone the repo
```bash
  git clone https://github.com/yourusername/Student-Performance-Predictor.git
  cd Student-Performance-Predictor
```
### 2. Install Dependencies
```bash
  pip install -r requirements.txt
```
Required packages: torch, scikit-learn, pandas, numpy, joblib

▶️ Usage 
---
## 1. Train the Model
```bash
  python train_student_model.py
```
* Trains the ANN and saves:
* model/Ann_exam_score_model.pkl
* model/scaler.pkl

## 2. Predict Marks
```bash
    python predict_exam_score.py
```
* Prompts for 19 student attributes.
* Outputs predicted exam score (0–100).

📈 Model Performance (Example)
---
| Metric       | Value |
|--------------|-------|
| R² Score     | 0.85+ |
| RMSE         | ~5.0  |


Note: Performance depends on data quality and preprocessing. 

---

🧪 Dataset Info
---
* **Source: data/StudentPerformanceFactors.csv**
* **Features: 19 (e.g., Hours_Studied, Parental_Involvement, Sleep_Hours)**
* **Target: Exam_Score (continuous, 0–100)**
* **Samples: ~1000 (based on input)**

---
📚 Technologies Used
---

* **Python**
* **PyTorch – Deep learning**
* **Scikit-learn – Preprocessing & metrics**
* **Pandas/Numpy – Data handling**

---
🤝 Contributing
---
### Pull requests are welcome! For major changes, please open an issue first.

# 📄 License

MIT

---

## Author
# Connect with me

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Ibrahim5570)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ibrahim-abdullah-220917319)
