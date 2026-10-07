from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pickle
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key = 'homecredit_secret_key_2024'

# Load model and medians
with open('home_credit_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("MODEL LOADED:", type(model))

feature_medians = pd.read_csv('feature_medians.csv', index_col=0).squeeze()

def safe_float(val, default=0):
    try:
        return float(val) if val != '' and val is not None else default
    except (ValueError, TypeError):
        return default

def calculate_features(d):

    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # =========================
    # BASIC NUMERIC EXTRACTION
    # =========================
    
    amt_income = safe_float(d.get('AMT_INCOME_TOTAL'))
    amt_credit = safe_float(d.get('AMT_CREDIT'))
    amt_annuity = safe_float(d.get('AMT_ANNUITY'))
    amt_goods = safe_float(d.get('AMT_GOODS_PRICE'))

    amt_payment_sum = safe_float(d.get('AMT_PAYMENT_SUM'))
    amt_instalment_sum = safe_float(d.get('AMT_INSTALMENT_SUM'))

    amt_balance_sum = safe_float(d.get('AMT_BALANCE_SUM'))
    amt_limit_sum = safe_float(d.get('AMT_CREDIT_LIMIT_ACTUAL_SUM'))

    credit_sum = safe_float(d.get('AMT_CREDIT_SUM_SUM'))
    debt_sum = safe_float(d.get('AMT_CREDIT_SUM_DEBT_SUM'))
    overdue_sum = safe_float(d.get('AMT_CREDIT_SUM_OVERDUE_SUM'))

    obs_60 = safe_float(d.get('OBS_60_CNT_SOCIAL_CIRCLE'))
    def_60 = safe_float(d.get('DEF_60_CNT_SOCIAL_CIRCLE'))

    sk_dpd = safe_float(d.get('SK_DPD_DEF_MAX'))  # ✅ corrected column

    # =========================
    # EXT SOURCE
    # =========================
    ext1 = safe_float(d.get('EXT_SOURCE_1'))
    ext2 = safe_float(d.get('EXT_SOURCE_2'))
    ext3 = safe_float(d.get('EXT_SOURCE_3'))

    d['EXT_SOURCE_MEAN'] = (ext1 + ext2 + ext3) / 3
    d['EXT_SOURCE_STD'] = np.std([ext1, ext2, ext3], ddof=1)

    # =========================
    # CORE RATIOS (FIXED)
    # =========================
    d['INCOME_CREDIT_RATIO'] = amt_income / (amt_credit + 1)
    d['ANNUITY_INCOME_RATIO'] = amt_annuity / (amt_income + 1)

    # =========================
    # CREDIT STRESS
    # =========================
    d['DEBT_TO_INCOME'] = debt_sum / (amt_income + 1)
    d['CREDIT_OVERDUE_RATIO'] = overdue_sum / (credit_sum + 1)
    d['TOTAL_DEBT_BURDEN'] = amt_annuity / (amt_income + 1)

    # =========================
    # PAYMENT BEHAVIOR
    # =========================
    d['PAYMENT_COMPLETION_RATE'] = amt_payment_sum / (amt_instalment_sum + 1)
    payment_delay_mean = safe_float(d.get('PAYMENT_DELAY_MEAN'))
    d['PAYMENT_DELAY_MEAN'] = payment_delay_mean
    num_approved = safe_float(d.get('num_approved'))
    num_refused = safe_float(d.get('num_refused'))

    total = max(num_approved + num_refused, 1)

    d['APPROVED_MEAN'] = num_approved / total
    d['REFUSED_MEAN'] = num_refused / total
    d['HAS_ANY_DELAY'] = 1 if payment_delay_mean > 0 else 0
    d['IS_CHRONICALLY_LATE'] = 1 if payment_delay_mean > 60 else 0
    # =========================
    # SOCIAL RISK
    # =========================
    if obs_60 > 0:
        d['SOCIAL_CIRCLE_DEFAULT_RATE'] = def_60 / obs_60
    else:
        d['SOCIAL_CIRCLE_DEFAULT_RATE'] = 0

    d['NO_SOCIAL_DATA'] = 1 if obs_60 == 0 else 0
    d['AGE_YEARS'] = safe_float(d.get('AGE_YEARS'))
    d['EMPLOYMENT_YEARS'] = safe_float(d.get('EMPLOYMENT_YEARS'))

    # =========================
    # UTILIZATION
    # =========================
    d['UTIL_BALANCE_RATIO'] = amt_balance_sum / (amt_limit_sum + 1)

    # =========================
    # STABILITY FLAGS
    # =========================
    d['IS_EMPLOYED'] = 1 if safe_float(d.get('DAYS_EMPLOYED')) < 0 else 0
    d['RECENTLY_CHANGED_PHONE'] = 1 if safe_float(d.get('PHONE_CHANGE_YEARS')) < 1 else 0

    d['HAS_CAR_AND_REALTY'] = 1 if (
        d.get('FLAG_OWN_CAR') == 'Y' and d.get('FLAG_OWN_REALTY') == 'Y'
    ) else 0

    return d

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/applicant', methods=['POST'])
def applicant():
    session['applicant_data'] = dict(request.form)
    return redirect(url_for('officer'))

@app.route('/officer', methods=['GET'])
def officer():
    return render_template('officer.html')

@app.route('/predict', methods=['POST'])
def predict():

    print("PREDICT ROUTE HIT", flush=True)

    # -----------------------------
    # DEMO MODE
    # -----------------------------
    selected_index = request.form.get('demoIndex')

    if selected_index and selected_index.isdigit() and 'demo_data' in session:
        print("USING DEMO DATA", flush=True)

        demo_rows = session.get('demo_data')
        combined = demo_rows[int(selected_index)]

        flat_applicant = combined

    else:
        print("USING USER INPUT", flush=True)

        applicant_data = session.get('applicant_data', {})
        officer_data = dict(request.form)

        flat_applicant = {
            k: v[0] if isinstance(v, list) else v
            for k, v in applicant_data.items()
        }

        combined = {**flat_applicant, **officer_data}

    # -----------------------------
    # DEBUG
    # -----------------------------
    print("OCCUPATION:", combined.get('OCCUPATION_TYPE'), flush=True)

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    combined = calculate_features(combined)

    input_df = pd.DataFrame([combined])

    input_df = pd.get_dummies(input_df)

    # align with model
    for col in model.feature_name_:
        if col not in input_df.columns:
            input_df[col] = feature_medians.get(col, 0)

    input_df = input_df.fillna(feature_medians)

    input_df = input_df[model.feature_name_]

    print("INPUT SHAPE:", input_df.shape, flush=True)

    # -----------------------------
    # PREDICT
    # -----------------------------
    proba = model.predict_proba(input_df)[0][1]

    print("PRED:", proba, flush=True)

    # continue your existing result logic...
    prediction = 'HIGH RISK' if proba >= 0.5 else 'LOW RISK'
    percentage = round(proba * 100, 1)

    # Risk level — LOWERCASE for CSS classes
    if proba >= 0.7:
        risk_level = 'critical'
        risk_message = 'Loan application should be carefully reviewed or declined.'
    elif proba > 0.5:
        risk_level = 'high'
        risk_message = 'Elevated risk detected. Additional verification recommended.'
    elif proba > 0.3:
        risk_level = 'medium'
        risk_message = 'Moderate risk. Standard approval process applies.'
    else:
        risk_level = 'low'
        risk_message = 'Low default risk. Applicant appears creditworthy.'

    # =======================================
    # Reasons and strengths (IMPROVED FINAL) 
    # =======================================

    reasons = []
    strengths = []

    income = safe_float(combined.get('AMT_INCOME_TOTAL'))
    credit = safe_float(combined.get('AMT_CREDIT'))
    annuity = safe_float(combined.get('AMT_ANNUITY'))
    ext2 = safe_float(combined.get('EXT_SOURCE_2'))
    debt = safe_float(combined.get('AMT_CREDIT_SUM_DEBT_SUM'))
    overdue = safe_float(combined.get('AMT_CREDIT_SUM_OVERDUE_SUM'))
    payment_delay = safe_float(combined.get('PAYMENT_DELAY_MEAN'))
    completion = safe_float(combined.get('PAYMENT_COMPLETION_RATE'))
    employment = safe_float(combined.get('EMPLOYMENT_YEARS'))

    # =========================
    # CREDIT SCORE
    # =========================
    if ext2 > 0:
        if ext2 < 0.3:
            reasons.append("Very poor external credit score")
        elif ext2 < 0.5:
            reasons.append("Below-average external credit score")
        elif ext2 < 0.7:
            strengths.append("Decent external credit profile")
        else:
            strengths.append("Strong external credit profile")

    # =========================
    # LOAN VS INCOME
    # =========================
    if income > 0:
        if credit > income * 5:
            reasons.append("Loan size is extremely high relative to income")
        elif credit > income * 3:
            reasons.append("Loan amount is high relative to income")
        elif credit < income * 1.5:
            strengths.append("Loan size is reasonable relative to income")

        if annuity > income * 0.5:
            reasons.append("Very high EMI burden")
        elif annuity > income * 0.35:
            reasons.append("Moderately high EMI burden")
        elif annuity < income * 0.25:
            strengths.append("Comfortable EMI-to-income ratio")

    # =========================
    # DEBT & OVERDUE
    # =========================
    if income > 0 and debt > income * 1.5:
        reasons.append("Debt level is significantly high compared to income")

    if overdue > 0:
        if overdue > 50000:
            reasons.append("Large overdue obligations detected")
        else:
            reasons.append("Minor overdue amounts present")

    # =========================
    # PAYMENT BEHAVIOR
    # =========================
    if payment_delay > 90:
        reasons.append("Severe and prolonged payment delays")
    elif payment_delay > 30:
        reasons.append("Frequent delays in repayments")
    elif payment_delay > 10:
        reasons.append("Occasional repayment delays")
    elif payment_delay <= 2:
        if ext2 > 0.6:
            strengths.append("Consistent and timely repayment behavior")
        else:
            strengths.append("Limited but recent repayment stability observed")

    # =========================
    # COMPLETION RATE
    # =========================
    if completion > 0:
        if completion < 0.6:
            reasons.append("Poor payment completion record")
        elif completion < 0.85:
            reasons.append("Moderate repayment consistency")
        elif completion > 0.98:
            if ext2 > 0.6:
                strengths.append("Very high repayment completion rate")
            else:
                strengths.append("Good recent repayment completion, though overall credit profile is mixed")
        elif completion > 0.9:
            strengths.append("Strong repayment consistency")

    # =========================
    # EMPLOYMENT
    # =========================
    if employment > 0:
        if employment < 1:
            reasons.append("Very short employment history")
        elif employment < 3:
            strengths.append("Moderate employment stability")
        elif employment > 6:
            strengths.append("Stable long-term employment")

    # =========================
    # CONSISTENCY LAYER (IMPORTANT FIX)
    # =========================
    if ext2 < 0.5:
        strengths = [
            s for s in strengths
            if not any(word in s.lower() for word in ["repayment", "completion", "discipline"])
            or "recent" in s.lower()
        ]
    
    if proba >= 0.7:
        # remove overly strong positives
        strengths = [s for s in strengths if "strong" not in s.lower()]
        strengths = [s for s in strengths if "very high" not in s.lower()]

        reasons.append("Overall profile indicates a high likelihood of default")

    elif proba <= 0.3:
        # remove overly harsh negatives
        reasons = [r for r in reasons if "extremely" not in r.lower()]

    # =========================
    # CLEANUP
    # =========================
    reasons = list(dict.fromkeys(reasons))
    strengths = list(dict.fromkeys(strengths))

    if not reasons:
        reasons.append("No major risk factors identified")

    if not strengths:
        strengths.append("Limited positive indicators available")

    return render_template('result.html',
                        prediction=prediction,
                        percentage=percentage,
                        risk_level=risk_level,
                        risk_message=risk_message,
                        applicant_name=combined.get('applicant_name', 'Applicant'),
                        reasons=reasons,
                        strengths=strengths)

@app.route('/get_demo_data')
def get_demo_data():
    try:
        with open(os.path.join(BASE_DIR, 'demo_data.json'), 'r') as f:
            data = json.load(f)
        print("DEMO DATA FILE LOADED:", data)
        session['demo_data'] = data
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_demo_data_officer')
def get_demo_data_officer():
    data = session.get('demo_data')
    if data:
        return jsonify(data)
    try:
        with open(os.path.join(BASE_DIR, 'demo_data.json'), 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': 'Demo data not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
