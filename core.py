import streamlit as st
import main as customer
import admin

def main():
    st.title("Loan Application System")

    st.sidebar.header("Options")
    selected_option = st.sidebar.radio("Select Role:", ("Customer", "Admin"))

    if selected_option == "Customer":
        st.subheader("Welcome, Customer!")
        customer.customer_main()
    elif selected_option == "Admin":
        st.subheader("Welcome, Admin!")
        admin.main()

if __name__ == "__main__":
    main()
