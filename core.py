import streamlit as st
import main as customer
import admin
import time

def main():
    st.title("LoanPredict+")

    # Create a spinner and a message
    with st.spinner("Welcome to the World of AI in Approval of Loans 🚀"):

        # Simulate loading data for 2 seconds
        time.sleep(2)

    st.sidebar.header("Navigation")
    st.sidebar.markdown("""
        This application is designed to provide users with predictive analysis on loan approvals using real-time data from Singlestore DB and Google Palm. Created by Gokul Palanisamy.
        """)
    st.sidebar.header("About Us")
    st.sidebar.markdown("""
        Developed by Gokul Palanisamy, this tool helps users make informed decisions on loan approvals by analyzing historical data and predicting future trends.
        """)
    st.sidebar.header("Contact Us")
    st.sidebar.markdown("""
        Email: [gokulp@bu.edu](mailto:gokulp@bu.edu)

        Phone: +1 (857) 832-0441

        More Information: [Gokul Palanisamy](https://www.google.com/search?q=Gokul+Palanisamy)
        """)

    selected_option = st.sidebar.selectbox("Select Role:", ("Customer", "Admin"))

    if selected_option == "Customer":
        st.subheader("Welcome, Customer!")
        customer.customer_main()
    elif selected_option == "Admin":
        st.subheader("Welcome, Admin!")
        admin.main()

if __name__ == "__main__":
    main()
