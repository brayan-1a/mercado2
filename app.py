import streamlit as st
import pandas as pd
from prediction import make_prediction, load_model, get_new_data_from_supabase
from config import get_supabase_client

def show_prediction_page():
    st.title('Predicción de Ventas')
    
    model = load_model()  # Cargar el modelo previamente entrenado
    new_data = get_new_data_from_supabase()  # Obtener los datos de Supabase
    
    st.write('Datos obtenidos de Supabase:')
    st.dataframe(new_data)
    
    if st.button('Hacer predicción'):
        predictions = make_prediction(model, new_data)
        new_data['prediccion'] = predictions
        st.write('Predicciones realizadas:')
        st.dataframe(new_data)

def main():
    show_prediction_page()

if __name__ == '__main__':
    main()


