# sidebar.py

import streamlit as st

def sidebar_inputs():
    st.sidebar.header("Redshift Input")
    z_input = st.sidebar.text_input("If set to 0, data will be in BCG-frame.", value="0.00", key="z_input")
    try:
        redshift = float(z_input.replace(",", "."))
    except ValueError:
        st.sidebar.error("Please enter a valid redshift.")
        redshift = None
    
    #inputting data
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload FITS data cube", type=["fits"], key="uploaded_file")

    return uploaded_file, redshift