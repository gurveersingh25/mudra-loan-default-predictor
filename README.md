# 🚀 Mudra Loan Default Prediction System

This project is a full-fledged **AI-powered loan default prediction system** built using a real-world Mudra loan dataset. The solution is deployed using **Streamlit**, integrates a trained machine learning model, and allows users to interactively input loan applicant details to get an instant prediction on whether the loan is likely to **default**.

> ⚠️ The dataset used is **confidential and proprietary**. It is not included in this repository to protect sensitive financial data. A sanitized sample version can be shared **privately upon request**.

---

## 📊 Project Features

- ✅ Real-world Mudra loan dataset (1+ lakh records)
- ✅ Cleaned, feature-engineered data used for model training
- ✅ Machine Learning classification model (sklearn)
- ✅ All categorical columns handled with saved LabelEncoders
- ✅ Fully interactive **Streamlit UI** with dropdowns and form inputs
- ✅ Inputs dynamically encoded and aligned to model expectations
- ✅ Model, encoders, and feature order saved using `joblib`

---

## 🧠 Tech Stack

| Layer           | Tools Used                          |
|----------------|--------------------------------------|
| Backend ML      | Scikit-learn, Pandas, NumPy         |
| Deployment      | Streamlit                           |
| Serialization   | Joblib (for model & encoders)       |
| Frontend UI     | Streamlit Widgets                   |
| Environment     | Python 3.10+                         |

---

## 🛠️ Files Included

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `app.py`               | Streamlit application code                                                  |
| `mudra_model.pkl`      | Final trained classification model                                          |
| `encoder_label.pkl`    | Dictionary of fitted LabelEncoders for each categorical column              |
| `feature_order.pkl`    | Feature column order required by the model                                  |
| `requirements.txt`     | List of all Python packages required to run the project                     |
| `README.md`            | Project documentation                                                       |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone this repo
git clone https://github.com/gurveersingh25/mudra-loan-default-predictor.git
cd mudra-loan-default-predictor

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

---

⚙️ How It Works
User fills out form details (loan, borrower, geography, etc.)

App uses stored encoders (encoder_label.pkl) to convert inputs to model-ready values

Final input is ordered to match feature_order.pkl

Model (mudra_model.pkl) returns prediction: ✅ Not Default / ❌ Default

---

📌 Important Notes
This app uses LabelEncoders for categorical data. It is crucial that any new input value must already be known to the encoder (i.e., it was present in training). Unseen values will cause the app to throw a ValueError.

To avoid this, all dropdowns are dynamically populated using encoder classes, ensuring only valid options are shown in the UI.

---

✅ Model Details
Model Type: Logistic Regression / RandomForestClassifier (based on best performance during experimentation)

Training Size: ~100,000+ rows

Features Used: 19 core features including business type, loan amount, state, city, credit line, and more

Categorical Features: Encoded using LabelEncoder

Numeric Features: Used as-is after cleaning and scaling

Accuracy (test set): ~93% (Fill with actual score)

---

🔒 Data Confidentiality
Due to the nature of the data:

The original dataset is not included in this repository.

If you're an employer, recruiter, or evaluator and would like to review the working system, a sanitized sample can be shared privately on request.

---

🙋‍♂️ Author
Gurveer Singh

📧 indian.army25ff@gmail.com

💼 LinkedIn:https://www.linkedin.com/in/gurveer-singh-184627283/

🌐 GitHub:https://github.com/gurveersingh25/mudra-loan-default-predictor

---

🏁 Final Note
This project demonstrates end-to-end data science + full-stack deployment skills using real-world financial data. Ideal for showcasing in portfolios, interviews, or resume links for high-paying data science/ML roles.


