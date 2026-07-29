import os
# 1. تعطيل خيوط المعالجة المتعددة في TensorFlow التي تسبب الشاشة السوداء في ويندوز
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 1. Page Configuration (تأكيد تحميل الواجهة فوراً)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 30px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Lazy Loading Function (تحميل TensorFlow بدون تجميد)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    try:
        # استدعاء TensorFlow داخل الدالة فقط لمنع تجميد السيرفر
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        from keras.models import load_model

        model = load_model('bank_churn_nn.keras', compile=False)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        return None, None

# ---------------------------------------------------------
# 3. Application Header
# ---------------------------------------------------------
st.markdown("<div class='main-header'>🏦 Bank Customer Churn Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict customer retention and churn risk using Artificial Neural Networks (ANN)</div>", unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------
# 4. Input Form Layout
# ---------------------------------------------------------
st.sidebar.header("📋 Customer Profile")
st.sidebar.write("Configure customer features for evaluation:")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics & Basic Info")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 38)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    st.subheader("💳 Financials & Account Details")
    balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
    num_products = st.radio("Number of Products", [1, 2, 3, 4], horizontal=True)
    card_type = st.selectbox("Card Type", ["DIAMOND", "GOLD", "SILVER", "PLATINUM"])
    point_earned = st.number_input("Points Earned", min_value=0, max_value=1000, value=450)

with col3:
    st.subheader("📊 Engagement & Feedback")
    has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
    complain = st.selectbox("Has Complaint?", ["No", "Yes"])
    satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)

st.divider()

## ---------------------------------------------------------
# 5. Preprocessing & Encoding (Matching Training Features)
# ---------------------------------------------------------
# One-hot encoding / Binary encoding matching exact model features
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

card_diamond = 1 if card_type == "DIAMOND" else 0
card_gold = 1 if card_type == "GOLD" else 0
card_platinum = 1 if card_type == "PLATINUM" else 0
card_silver = 1 if card_type == "SILVER" else 0

complain_val = 1 if complain == "Yes" else 0
has_card_val = 1 if has_card == "Yes" else 0
is_active_val = 1 if is_active == "Yes" else 0

# Input DataFrame Construction with EXACT trained features
input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_card_val,
    'IsActiveMember': is_active_val,
    'EstimatedSalary': salary,
    'Satisfaction Score': satisfaction_score,
    'Point Earned': point_earned,
    'Complain': complain_val,
    'Geography_France': 1 if geography == "France" else 0,
    'Geography_Germany': geo_germany,
    'Geography_Spain': geo_spain,
    'Gender_Female': gender_female,
    'Gender_Male': gender_male,
    'Card Type_DIAMOND': card_diamond,
    'Card Type_GOLD': card_gold,
    'Card Type_PLATINUM': card_platinum,
    'Card Type_SILVER': card_silver
}

input_df = pd.DataFrame([input_dict])

# ---------------------------------------------------------
# 6. Prediction Logic & Output
# ---------------------------------------------------------
st.subheader("🔍 Prediction Results")

if st.button("Calculate Churn Risk"):
    with st.spinner("Loading Model & Calculating..."):
        model, scaler = load_model_and_scaler()
        
        if model is not None and scaler is not None:
            try:
                # Reorder features to match scaler fit order automatically if needed
                if hasattr(scaler, 'feature_names_in_'):
                    input_df_reordered = input_df[scaler.feature_names_in_]
                else:
                    input_df_reordered = input_df
                
                scaled_input = scaler.transform(input_df_reordered)
                raw_pred = model.predict(scaled_input, verbose=0)
                probability = float(raw_pred[0][0])
                churn_risk_percent = probability * 100
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.metric(
                        label="Churn Probability", 
                        value=f"{churn_risk_percent:.1f}%"
                    )
                    
                with res_col2:
                    st.write("### Risk Assessment:")
                    st.progress(probability)
                    
                    if probability >= 0.5:
                        st.error("⚠️ **High Churn Risk:** This customer is likely to leave the bank. Immediate retention actions are recommended.")
                    else:
                        st.success("✅ **Low Churn Risk:** This customer is stable and likely to stay with the bank.")
                        
            except Exception as err:
                st.error(f"Error during prediction: {err}")
        else:
            st.error("Model assets could not be loaded. Check file names and directory.")