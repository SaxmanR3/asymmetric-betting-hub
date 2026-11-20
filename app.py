import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ODDSPAPI_API_KEY = (
    st.secrets.get("ODDSPAPI_API_KEY")
    if hasattr(st, "secrets") and "ODDSPAPI_API_KEY" in st.secrets
    else os.getenv("ODDSPAPI_API_KEY")
)