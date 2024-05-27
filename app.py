import streamlit as st
from admin import get_loan_approval_chain

# Load the background image
st.image('/Users/gokul/Desktop/Believe/Projects/langchain/Main/images/main.jpg', caption='')

# Text input for the admin to enter the question
question = st.text_input("Question: ")

# If a question is provided, run the language model chain to get the response
if question:
    # Initialize the language model chain
    chain = get_loan_approval_chain()

    # Run the chain to get the response
    response = chain.run(question)

    # Display the response
    st.header("Answer")
    st.write(response)
