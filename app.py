import streamlit as st
import joblib
import numpy as np

model_path = 'model_o_0.5_u_0.7.joblib'
model = joblib.load(model_path)

def predict_fraud_proba(features):
    input_data = np.array(features).reshape(1, -1)
    proba = model.predict_proba(input_data)[:, 1]

    return proba[0]

def main():
    st.title("Fraud Detection App - CyberTrove Politeknik Negeri Bandung")

    features = {
        'income': st.slider('Income - Annual income of the applicant (in decile form). Ranges between 0.1-0.9', min_value=0.0, max_value=1.0, step=0.01, key='income'),
        'name_email_similarity': st.slider('Name-Email Similarity - Metric of similarity between email and applicants name. Higher values represent higher similarity. Ranges between [0, 1].', min_value=0.0, max_value=1.0, step=0.01, key='name_email_similarity'),
        'prev_address_months_count': st.slider('Previous Address Months Count - Number of months in previous registered address of the applicant, i.e. the applicants previous residence, if applicable. Ranges between [−1, 380] months (-1 is a missing value)', min_value=-1, max_value=380, step=1, key='prev_address_months_count'),
        'current_address_months_count': st.slider('Current Address Months Count - Months in currently registered address of the applicant. Ranges between [−1, 429] months (-1 is a missing value)', min_value=-1, max_value=429, step=1, key='current_address_months_count'),
        'customer_age': st.slider('Customer Age - Applicant’s age in years, rounded to the decade. Ranges between [10, 90] years.', min_value=10, max_value=90, step=1, key='customer_age'),
        'days_since_request': st.slider('Days Since Request -  Number of days passed since application was done. Ranges between [0, 79] days.', min_value=0, max_value=79, step=1, key='days_since_request'),
        'intended_balcon_amount': st.slider('Intended Balcon Amount - Initial transferred amount for application. Ranges between [−16, 114] (negatives are missing values)', min_value=-16, max_value=114, step=1, key='intended_balcon_amount'),
        'zip_count_4w': st.slider('Zip Count 4 Weeks - Number of applications within same zip code in last 4 weeks. Ranges between [1, 6830]', min_value=1, max_value=6830, step=1, key='zip_count_4w'),
        'velocity_6h': st.slider('Velocity 6 Hours - Velocity of total applications made in last 6 hours i.e., average number of applications per hour in the last 6 hours. Ranges between [−175, 16818].', min_value=-175, max_value=16818, step=1, key='velocity_6h'),
        'velocity_24h': st.slider('Velocity 24 Hours - Velocity of total applications made in last 24 hours i.e., average number of applications per hour in the last 24 hours. Ranges between [1297, 9586]', min_value=1297, max_value=9586, step=1, key='velocity_24h'),
        'velocity_4w': st.slider('Velocity 4 Weeks - Velocity of total applications made in last 4 weeks, i.e., average number of applications per hour in the last 4 weeks. Ranges between [2825, 7020]', min_value=2825, max_value=7020, step=1, key='velocity_4w'),
        'bank_branch_count_8w': st.slider('Bank Branch Count 8 Weeks -  Number of total applications in the selected bank branch in last 8 weeks. Ranges between [0, 2404]', min_value=0, max_value=2404, step=1, key='bank_branch_count_8w'),
        'date_of_birth_distinct_emails_4w': st.slider('DOB Distinct Emails 4 Weeks - Number of emails for applicants with same date of birth in last 4 weeks. Ranges between [0, 39].', min_value=0, max_value=39, step=1, key='date_of_birth_distinct_emails_4w'),
        'credit_risk_score': st.slider('Credit Risk Score -  Internal score of application risk. Ranges between [−191, 389]', min_value=-191, max_value=389, step=1, key='credit_risk_score'),
        'email_is_free': st.slider('Email is Free - Domain of application email (either free or paid)', min_value=0, max_value=1, step=1, key='email_is_free'),
        'phone_home_valid': st.slider('Phone Home Valid -  Validity of provided home phone', min_value=0, max_value=1, step=1, key='phone_home_valid'),
        'phone_mobile_valid': st.slider('Phone Mobile Valid - Validity of provided mobile phone', min_value=0, max_value=1, step=1, key='phone_mobile_valid'),
        'bank_months_count': st.slider('Bank Months Count - : How old is previous account (if held) in months. Ranges between [−1, 32] months (-1 is a missing value).', min_value=-1, max_value=32, step=1, key='bank_months_count'),
        'has_other_cards': st.slider('Has Other Cards - If applicant has other cards from the same banking company', min_value=0, max_value=1, step=1, key='has_other_cards'),
        'proposed_credit_limit': st.slider('Proposed Credit Limit - Applicant’s proposed credit limit. Ranges between [200, 2000]', min_value=200, max_value=2000, step=1, key='proposed_credit_limit'),
        'foreign_request': st.slider('Foreign Request - : If origin country of request is different from bank’s country.', min_value=0, max_value=1, step=1, key='foreign_request'),
        'session_length_in_minutes': st.slider('Session Length (Minutes) - : Length of user session in banking website in minutes. Ranges between [−1, 107] minutes (-1 is a missing value).', min_value=-1, max_value=107, step=1, key='session_length_in_minutes'),
        'keep_alive_session': st.slider('Keep Alive Session -  User option on session logout', min_value=0, max_value=1, step=1, key='keep_alive_session'),
        'device_distinct_emails_8w': st.slider('Device Distinct Emails 8 Weeks - : Number of distinct emails in banking website from the used device in last 8 weeks. Ranges between [−1, 2] emails (-1 is a missing value)', min_value=-1.0, max_value=2.0, step=0.1, key='device_distinct_emails_8w'),
        'device_fraud_count': st.slider('Device Fraud Count - Number of fraudulent applications with used device. Ranges between [0, 1].', min_value=0, max_value=1, step=1, key='device_fraud_count'),
        'month': st.slider('Month - : Month where the application was made. Ranges between [0, 7].', min_value=0, max_value=7, step=1, key='month')
    }

    input_values = [value for value in features.values()]

    if st.button("Predict Fraud"):
        proba = predict_fraud_proba(input_values)
        st.write(f"Probability of Fraud: {proba*100:.3f} %")
        if 0.4 <= proba <= 0.6:
            st.warning("Model Prediction: **Like Fraud** (Probability in the range 0.4 to 0.6)!")
        elif proba < 0.4:
            st.success("Model Prediction: **No Fraud Detected** (Probability less than 0.4).")
        else:
            st.error("Model Prediction: **Fraud Detected** (Probability greater than 0.6).")


if __name__ == "__main__":
    main()