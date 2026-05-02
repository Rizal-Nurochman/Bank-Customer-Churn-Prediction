import joblib
import streamlit as st
import os
import pandas as pd

@st.cache_data
def load_data(path):
  if os.path.exists:
    df = pd.read_csv(path)
    return df
  return print(f'Data tidak ditemukan')

@st.cache_resource
def load_assets():
    model = joblib.load("notebooks/model.pkl")
    scaler = joblib.load("notebooks/scaler.pkl")
    encoder = joblib.load("notebooks/encoder.pkl")
    num_cols = joblib.load("notebooks/numerical_cols.pkl")
    return model, scaler, encoder, num_cols