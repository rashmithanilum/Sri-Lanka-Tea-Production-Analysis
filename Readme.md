# Sri Lankan Tea Production Analysis & Yield Prediction

This repository contains the code, data, and reports for a data science project analyzing Sri Lankan tea production. It includes both comprehensive Exploratory Data Analysis (EDA) and predictive modeling to classify and forecast high tea yields.

This work was completed as part of the ST 3011 assignment (Group 08).

## 📁 Project Structure

* `app.py`: The main Python application script (Streamlit/Flask) for the interactive model interface.
* `Tea_EDA_Part.ipynb`: Jupyter Notebook containing the Exploratory Data Analysis.
* `tea_combined_preprocessed.csv`: The cleaned and preprocessed dataset used for model training.
* `total_sarimax.joblib` / `share_catboost.joblib`: Exported machine learning and time-series models (CatBoost and SARIMAX).
* **Reports**: Includes the main Group 08 report and individual contribution documents detailing the methodology behind the high tea yield classification.

## 🎯 Objectives
1.  Perform EDA to uncover trends and patterns in Sri Lankan tea production data over time.
2.  Implement machine learning classification models to accurately predict periods of high tea yields.
3.  Deploy an interactive application (`app.py`) to demonstrate the model's predictive capabilities.

## 🚀 How to Run the Project

**1. Clone the repository:**
```bash
git clone https://github.com/rashmithanilum/Sri-Lanka-Tea-Production-Analysis.git
cd Sri-Lanka-Tea-Production-Analysis
```

**2. Install dependencies:**
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit App:**
```bash
streamlit run app.py
```