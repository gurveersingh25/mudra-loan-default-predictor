import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AI-Powered Mudra Loan Risk Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: #121212 !important;  /* Dark background */
    }
    .main::before {
        background: none !important;  /* Remove overlay */
    }
    .stApp > header {
        background-color: transparent;
    }
    .main-header {
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.8);
        border-radius: 10px;
    }
    .approved-result, .rejected-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        background: rgba(255, 255, 255, 0.1); /* Semi-transparent background */
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    .input-section {
        background: transparent !important;  /* Remove white box */
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .section-header {
        color: white !important;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("mudra_model.pkl")
    encoders = joblib.load("encoder_label.pkl")
    feature_order = joblib.load("feature_order.pkl")
    return model, encoders, feature_order

model, encoders, feature_order = load_model_and_encoders()

# Main UI
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI-Powered Mudra Loan Risk Classifier</h1>
        <h3>Powered by Machine Learning</h3>
        <p>An intelligent loan evaluation tool that uses historical Mudra loan data to predict whether a loan is likely to default or be repaid</p>
        <p style="margin-top: 1rem; font-size: 1rem; color: #ccc;"><em>By Gurveer Singh</em></p>
    </div>
    """, unsafe_allow_html=True)


    with st.form("loan_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🔹 Business Details</div>', unsafe_allow_html=True)

            business = st.selectbox("Type of Business", encoders['business'].classes_.tolist())
            classification_code = st.selectbox("Loan Classification Code", ['MUDRA-SHISHU', 'MUDRA-KISHORE', 'MUDRA-TARUN'])
            count_employees = st.number_input("Number of Employees", 0, 100, 5)
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])
            loan_approved_gross = st.number_input("Total Loan Amount (₹)", 10000.0, 10000000.0, 100000.0, 1000.0)
            gross_amount_disbursed = st.number_input("Amount Disbursed (₹)", 0.0, 10000000.0, 90000.0, 1000.0)
            gross_amount_balance = st.number_input("Remaining Balance (₹)", 0.0, 10000000.0, 50000.0, 1000.0)
            chargedoff_amount = st.number_input("Charged Off Amount (₹)", 0.0, loan_approved_gross, 0.0, 1000.0)

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">👤 Borrower Demographics</div>', unsafe_allow_html=True)

            demography = st.selectbox("Area Type", encoders['demography'].classes_.tolist())
            borrower_city = st.selectbox("Borrower City", encoders['borrower_city'].classes_.tolist())
            borrower_state = st.selectbox("Borrower State", encoders['borrower_state'].classes_.tolist())

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🏦 Bank Information</div>', unsafe_allow_html=True)

            name_of_bank = st.selectbox("Lending Bank", encoders['name_of_bank'].classes_.tolist())
            state_of_bank = st.selectbox("Bank State", encoders['state_of_bank'].classes_.tolist())
            code_franchise = st.number_input("Franchise Code", 1000, 9999, 5000)

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">📄 Loan Parameters</div>', unsafe_allow_html=True)

            guaranteed_approved_loan = st.selectbox("Guaranteed Loan", encoders['guaranteed_approved__loan'].classes_.tolist())
            low_documentation_loan = st.selectbox("Low Documentation Loan", encoders['low_documentation_loan'].classes_.tolist())
            revolving_credit_line = st.selectbox("Revolving Credit Line", encoders['revolving_credit_line'].classes_.tolist())

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">👨‍💼 Employment Impact</div>', unsafe_allow_html=True)

            jobs_created = st.number_input("Jobs Created", 0, 50, 2)
            jobs_retained = st.number_input("Jobs Retained", 0, 100, 5)

            st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔮 Predict Loan Default Risk", use_container_width=True)

        if submitted:
            try:
                # Classification Code fallback
                if 'classification_code' in encoders:
                    classification_encoded = encoders['classification_code'].transform([classification_code])[0]
                else:
                    classification_map = {'MUDRA-SHISHU': 0, 'MUDRA-KISHORE': 1, 'MUDRA-TARUN': 2}
                    classification_encoded = classification_map.get(classification_code, 0)

                input_data = pd.DataFrame({
                    'business': [encoders['business'].transform([business])[0]],
                    'jobs_reatained': [jobs_retained],  # keep typo
                    'jobs_created': [jobs_created],
                    'guaranteed_approved__loan': [encoders['guaranteed_approved__loan'].transform([guaranteed_approved_loan])[0]],
                    'low_documentation_loan': [encoders['low_documentation_loan'].transform([low_documentation_loan])[0]],
                    'demography': [encoders['demography'].transform([demography])[0]],
                    'state_of_bank': [encoders['state_of_bank'].transform([state_of_bank])[0]],
                    'chargedoff_amount': [chargedoff_amount],
                    'borrower_city': [encoders['borrower_city'].transform([borrower_city])[0]],
                    'borrower_state': [encoders['borrower_state'].transform([borrower_state])[0]],
                    'gross_amount_balance': [gross_amount_balance],
                    'count_employees': [count_employees],
                    'classification_code': [classification_encoded],
                    'loan_approved_gross': [loan_approved_gross],
                    'gross_amount_disbursed': [gross_amount_disbursed],
                    'loan_term': [loan_term],
                    'code_franchise': [code_franchise],
                    'name_of_bank': [encoders['name_of_bank'].transform([name_of_bank])[0]],
                    'revolving_credit_line': [encoders['revolving_credit_line'].transform([revolving_credit_line])[0]]
                })

                input_data = input_data[feature_order]
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]

                if prediction == 0:
                    st.markdown(f"""
                        <div class="approved-result">
                            ✅ LOAN APPROVED<br>
                            <small>Confidence: {prediction_proba[0] * 100:.1f}%</small><br>
                            <small>This loan is likely to be repaid successfully</small>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="rejected-result">
                            ❌ HIGH DEFAULT RISK<br>
                            <small>Confidence: {prediction_proba[1] * 100:.1f}%</small><br>
                            <small>This loan has a high probability of default</small>
                        </div>
                    """, unsafe_allow_html=True)

            except ValueError as ve:
                st.error(f"Value Error: {ve}")
                st.info("Please check that all dropdown values are selected correctly.")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.info("Please ensure all fields are filled correctly and the model file is compatible.")

if __name__ == "__main__":
    main()
