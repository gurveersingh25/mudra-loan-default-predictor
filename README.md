# 🚀 Mudra Loan Default Prediction System

**[Live Demo on Streamlit](https://mudra-loan-default-predictor.streamlit.app/)**  
📦 Predict loan defaults using a machine learning model trained on real Mudra loan data.

---

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

## ⚙️ How It Works

- User fills out loan and borrower details via the form.
- App uses stored encoders (`encoder_label.pkl`) to convert values.
- Inputs are ordered using `feature_order.pkl` to match training.
- The trained model (`mudra_model.pkl`) outputs the prediction.

---

## 📌 Important Notes

- Categorical inputs are encoded using `LabelEncoder`.
- **Unseen values** (not present during training) will raise a `ValueError`.
- **All dropdowns are dynamically generated** from the encoder to prevent this issue.

---

## ✅ Model Details

- **Model**: Logistic Regression / Random Forest (best performance retained)
- **Dataset Size**: ~100,000+ records
- **Features Used**: 19 core variables including:
  - Business type
  - City, state, bank name
  - Loan amount, credit line, loan guarantee
- **Categorical Handling**: LabelEncoded
- **Numeric Handling**: Cleaned and scaled
- **Accuracy**: ~93% (based on test set performance)

---

## 🔐 Data Confidentiality

- Due to sensitive financial data, the **original dataset is excluded** from this repository.
- Recruiters or evaluators may request a **sanitized sample privately** for demo purposes.

---

## 🙋‍♂️ Author
Gurveer Singh

- 📧 Email:indian.army25ff@gmail.com

- 💼 LinkedIn:https://www.linkedin.com/in/gurveer-singh-184627283/

- 🌐 GitHub:https://github.com/gurveersingh25/mudra-loan-default-predictor

---

## 🏁 Final Note
- This project demonstrates end-to-end data science + full-stack deployment skills using real-world financial data. Ideal for showcasing in portfolios, interviews, or resume links for high-paying data science/ML roles.


