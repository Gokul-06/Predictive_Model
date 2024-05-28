import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd

from langchain_community.llms import GooglePalm
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from few_shots import few_shots

import os
from dotenv import load_dotenv

load_dotenv()

def get_loan_approval_chain():
    db_user = "admin"
    db_password = "Rules123"
    db_host = "svc-538d9a8e-2af0-4a3c-abf5-f797b0d6b9f0-dml.aws-virginia-6.svc.singlestore.com"
    db_port = 3306
    db_name = "loan_prediction"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
       Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
       Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
       Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
       Pay attention to use CURDATE() function to get the current date, if the question involves "today".

       Use the following format:

       Question: Question here
       SQLQuery: Query to run with no pre-amble
       SQLResult: Result of the SQLQuery
       Answer: Final answer here

       No pre-amble.
       """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult", "Answer", ],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],  # These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain



def main():
    st.title("Loan Approval Predictive Model")

    question = st.text_input("Question: ")

    if question:
        # Fetch the response from the database
        chain = get_loan_approval_chain()
        response = chain.run(question)

        st.header("Answer")
        st.write(response)

        # Generate SHAP explanations
        # Assuming you have access to your trained model (`model`) and data (`X_train`, `X_test`)
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

        # Create DataFrame for SHAP values
        shap_df = pd.DataFrame({"Feature": X.columns, "SHAP Value": shap_values[0]})

        # Sort the DataFrame by absolute SHAP values (for importance ranking)
        shap_df["Absolute SHAP"] = abs(shap_df["SHAP Value"])
        shap_df = shap_df.sort_values(by="Absolute SHAP", ascending=False).drop(columns=["Absolute SHAP"])

        # Display the top N features
        N = 10
        top_features = shap_df.head(N)
        st.subheader("Top Features and SHAP Values")
        st.table(top_features)

        # Plot the summary plot
        st.subheader("SHAP Summary Plot")
        shap.summary_plot(shap_values, X_test)
        st.pyplot()

        # Create the force plot
        st.subheader("SHAP Force Plot")
        shap.force_plot(
            explainer.expected_value,   # The expected value of the model
            shap_values[0],             # SHAP values for the chosen prediction
            X_test,                     # Features for the chosen prediction
            feature_names=X.columns,    # List of feature names
            matplotlib=True
        )
        plt.gcf().set_size_inches(10, 7)  # Control the size of the figure using Matplotlib parameters
        plt.rcParams.update({"font.size": 18})
        st.pyplot(plt)

if __name__ == "__main__":
    main()