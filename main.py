import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
import mysql.connector
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)

# SingleStore database credentials
db_user = "admin"
db_password = "Rules123"
db_host = "svc-538d9a8e-2af0-4a3c-abf5-f797b0d6b9f0-dml.aws-virginia-6.svc.singlestore.com"
db_port = 3306
db_name = "loan_prediction"


def preprocess_dependents(value):
    if value == '3+':
        return 4  # Interpret '3+' as 4 dependents
    return int(value)


# Function to train the model using the sample CSV file
def train_model():
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("Loan_Applications_Sample_All_Accepted.csv")

    # Preprocess 'dependents' data
    df['dependents'] = df['dependents'].apply(preprocess_dependents)

    # One-hot encoding for categorical columns
    categorical_columns = ['gender', 'married', 'education', 'self_employed', 'property_area']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Define the features and target columns
    X = df_encoded.drop("loan_status", axis=1)
    y = df_encoded["loan_status"]

    # Train a logistic regression model using the data
    model = LogisticRegression()
    model.fit(X, y)
    return model, X.columns


def create_loan_vector(loan_data):
    dependents = preprocess_dependents(loan_data['dependents'])
    gender_vector = 1 if loan_data["gender"] == "Male" else 0
    married_vector = 1 if loan_data["married"] == "Yes" else 0
    education_vector = 1 if loan_data["education"] == "Graduate" else 0
    self_employed_vector = 1 if loan_data["self_employed"] == "Yes" else 0
    property_area_vector = 1 if loan_data["property_area"] == "Urban" else 0
    credit_history_vector = 1 if loan_data["credit_history"] == "1" else 0

    loan_vector = [
        gender_vector, married_vector, dependents, education_vector, self_employed_vector,
        loan_data["applicant_income"], loan_data["coapplicant_income"], loan_data["loan_amount"],
        loan_data["loan_amount_term"], credit_history_vector, property_area_vector
    ]

    return np.array(loan_vector).reshape(1, -1)


def predict_loan_approval(model, feature_columns, user_input):
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_df['dependents'] = user_input_df['dependents'].apply(preprocess_dependents)

    # Ensure all required columns are present
    user_input_encoded = pd.get_dummies(user_input_df, drop_first=True)
    # Align the columns with the training data columns
    user_input_encoded = user_input_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(user_input_encoded)[0]
    return prediction


def update_database_loan_application(loan_data):
    connection = mysql.connector.connect(
        user=db_user,
        password=db_password,
        host=db_host,
        database=db_name,
        port=db_port
    )
    cursor = connection.cursor()

    insert_query = """INSERT INTO loan_applications (loan_id, name, gender, married, dependents, education, self_employed, 
                     applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, 
                     property_area, loan_status) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    cursor.execute(insert_query, (
        loan_data["loan_id"], loan_data["name"], loan_data["gender"], loan_data["married"], loan_data["dependents"],
        loan_data["education"], loan_data["self_employed"], loan_data["applicant_income"],
        loan_data["coapplicant_income"], loan_data["loan_amount"], loan_data["loan_amount_term"],
        loan_data["credit_history"], loan_data["property_area"], loan_data["loan_status"]))

    loan_vector = create_loan_vector(loan_data)

    insert_vector_query = "INSERT INTO loan_vectors (loan_id, loan_vector) VALUES (%s, %s)"
    cursor.execute(insert_vector_query, (loan_data["loan_id"], loan_vector.tobytes()))

    connection.commit()
    cursor.close()
    connection.close()


def customer_main():
    st.title("Loan Approval Predictive Model")

    model, feature_columns = train_model()

    loan_id = st.number_input("Loan ID", min_value=0, step=1, value=0)
    name = st.text_input("Name", "John Doe")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Dependents", min_value=0, max_value=3, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, step=1)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=1)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=1)
    loan_amount_term = st.number_input("Loan Amount Term", min_value=0, step=1)
    credit_history = st.selectbox("Credit History", ["1", "0"])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    user_input = {
        "loan_id": loan_id,
        "name": name,
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_amount_term": loan_amount_term,
        "credit_history": credit_history,
        "property_area": property_area
    }

    if st.button("Predict Loan Approval and Submit Data"):
        prediction = predict_loan_approval(model, feature_columns, user_input)
        if prediction == "Y":
            st.success("Loan approved!")
        else:
            st.error("Loan denied.")

        # Visualize prediction
        probabilities = model.predict_proba(create_loan_vector(user_input))[0]
        labels = ['Denied', 'Approved']
        plt.bar(labels, probabilities, color=['red', 'green'])
        plt.xlabel('Loan Status')
        plt.ylabel('Probability')
        plt.title('Loan Approval Probability')
        st.pyplot(plt)

        loan_data = {**user_input, "loan_status": prediction}
        try:
            update_database_loan_application(loan_data)
            st.write("Data submitted to the database.")
        except Exception as e:
            st.error(f"Error submitting data to the database: {str(e)}")
            logging.error(f"Error submitting data to the database: {str(e)}")


if __name__ == "__main__":
    customer_main()
