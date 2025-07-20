# 🧠 Agile Project Estimator (Streamlit App)

This Streamlit application uses machine learning to provide **early effort estimations (in man-months)** for Agile software projects. It allows users to input project parameters and select a trained model to predict how much time the project may take.

---

## 🚀 Features

- Estimate man-months using trained machine learning models.
- Real-time prediction based on:
  - Year of the Project
  - Industry Type
  - Primary Programming Language
  - Project Complexity
  - Team Experience
  - Number of Requirements
  - Maximum Team Size
  - Technology Stack Complexity
...
---

## 📁 Folder Structure
agileee_app/
├── streamlit_app.py        # Streamlit Community Cloud Launcher
├── agilee
|   ├── main.py             # Entry point of the Streamlit app
|   ├──  models.py          # Model creation, loading.prediction logic
|   ├── ui.py               # Streamlit UI components
├── models/                 # Folder where pickled models and scaler are saved
│   ├── top1_.pkl
│   ├── top2_.pkl
│   └── top3_.pkl
├── requirements.txt         # Python dependencies for the app
└── README.md                # Documentation on how to run and use the app

---

## 🛠️ Setup Instructions

1. **Clone the repo** or copy the files into a folder:
   ```bash
   git clone https://github.com/yourname/agileee-pro.git
   cd agileee-prod
2. Install dependencies:
    pip install -r requirements.txt
3. Run the app:
    streamlit run main.py

