import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging
import mysql.connector
import numpy as np
import torch
import transformers
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)

# SingleStore database credentials
db_user = "admin"
db_password = "Rules123"
db_host = "svc-538d9a8e-2af0-4a3c-abf5-f797b0d6b9f0-dml.aws-virginia-6.svc.singlestore.com"
db_port = 3306
db_name = "loan_prediction"

# Function to train the model using the sample CSV file
def train_model():
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("Loan_Applications_Sample.csv")

    # One-hot encoding for categorical columns
    categorical_columns = ['gender', 'married', 'education', 'self_employed', 'property_area']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Define the features and target columns
    X = df_encoded.drop("loan_status", axis=1)
    y = df_encoded["loan_status"]

    # Train a logistic regression model using the data
    model = LogisticRegression()
    model.fit(X, y)
    return model, X.columns

# Function to create a vector representation of loan application data using BERT
def create_loan_vector(loan_data):
    # Load the pre-trained BERT model and tokenizer
    model = transformers.BertModel.from_pretrained('bert-base-uncased')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the loan application data
    tokens = tokenizer.encode(str(loan_data), add_special_tokens=True)

    # Create a tensor from the tokenized data
    input_ids = torch.tensor([tokens])

    # Generate a vector representation of the loan application data using BERT
    with torch.no_grad():
        outputs = model(input_ids)
        last_hidden_states = outputs.last_hidden_state
        loan_vector = last_hidden_states[:, 0, :].numpy().flatten()

    return loan_vector

# Function to convert bytes data back to numpy array
def bytes_to_array(bytes_data):
    return np.frombuffer(bytes_data, dtype=np.float64)

# Function to predict loan approval based on user inputs
def predict_loan_approval(model, feature_columns, user_input):
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame(user_input, index=[0])

    # One-hot encode the user input
    user_input_encoded = pd.get_dummies(user_input_df)

    # Align the columns of user_input_encoded with the trained model's feature columns
    user_input_encoded = user_input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Make a prediction using the trained model
    prediction = model.predict(user_input_encoded)[0]

    return prediction

# Function to update database with loan application data and vector representation
def update_database_loan_application(loan_data):
    # Connect to SingleStore database
    connection = mysql.connector.connect(
        user=db_user,
        password=db_password,
        host=db_host,
        database=db_name,
        port=db_port
    )
    cursor = connection.cursor()

    # Insert loan application data
    insert_query = """INSERT INTO loan_applications (loan_id, gender, married, dependents, education, self_employed, 
                     applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, 
                     property_area, loan_status) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    cursor.execute(insert_query, (
        loan_data["loan_id"], loan_data["gender"], loan_data["married"], loan_data["dependents"],
        loan_data["education"], loan_data["self_employed"], loan_data["applicant_income"],
        loan_data["coapplicant_income"], loan_data["loan_amount"], loan_data["loan_amount_term"],
        loan_data["credit_history"], loan_data["property_area"], loan_data["loan_status"]))

    # Create loan vector
    loan_vector = create_loan_vector(loan_data)

    # Encode loan vector as base64
    loan_vector_base64 = base64.b64encode(loan_vector).decode("utf-8")

    # Insert loan vector into loan_vectors table
    insert_vector_query = "INSERT INTO loan_vectors (loan_id, loan_vector) VALUES (%s, %s)"
    cursor.execute(insert_vector_query, (loan_data["loan_id"], loan_vector_base64))

    connection.commit()
    cursor.close()
    connection.close()

# Function to retrieve loan vector from database
def retrieve_loan_vector(loan_id):
    # Connect to SingleStore database
    connection = mysql.connector.connect(
        user=db_user,
        password=db_password,
        host=db_host,
        database=db_name,
        port=db_port
    )
    cursor = connection.cursor()

    # Retrieve loan vector for the given loan_id
    select_vector_query = "SELECT loan_vector FROM loan_vectors WHERE loan_id = %s"
    cursor.execute(select_vector_query, (loan_id,))
    result = cursor.fetchone()

    cursor.close()
    connection.close()

    if result:
        # Decode loan vector from base64
        loan_vector_base64 = result[0]
        loan_vector = base64.b64decode(loan_vector_base64)
        return loan_vector
    else:
        return None

# Function to display loan approval prediction and submit data to the database
def customer_main():
    st.title("Loan Approval Predictive Model")

    # Train the model using the sample CSV file
    model, feature_columns = train_model()

    # Get user input from Streamlit interface
    loan_id = st.number_input("Loan ID", min_value=0, step=1, value=0)
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

    # Add a button to trigger the loan approval prediction and submit data to the database
    if st.button("Predict Loan Approval and Submit Data"):
        # Make a prediction using the trained model
        prediction = predict_loan_approval(model, feature_columns, user_input)

        # Display the prediction result
        if prediction == "Y":
            st.success("Loan approved!")
        else:
            st.error("Loan denied.")

        # Add loan application data to SingleStore database
        loan_data = {**user_input, "loan_status": prediction}
        try:
            update_database_loan_application(loan_data)
            st.write("Data submitted to the database.")
        except Exception as e:
            st.error(f"Error submitting data to the database: {str(e)}")
            logging.error(f"Error submitting data to the database: {str(e)}")

if __name__ == "__main__":
    customer_main()
