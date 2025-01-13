import streamlit as st
import pandas as pd
from model_train import train_model
from prediction import make_prediction
from utils import load_model, save_model, plot_feature_importance
from preprocess import preprocess_data
from config import get_supabase_client
import joblib

# Título de la aplicación
st.title('Predicción de Compras y Control de Desperdicios')

# Definir la barra lateral para la navegación
st.sidebar.title('Opciones')

# Opción para entrenar el modelo
if st.sidebar.button('Entrenar modelo'):
    st.write("Entrenando modelo, por favor espera...")
    model = train_model()  # Entrena el modelo
    save_model(model, 'modelo_entrenado.pkl')  # Guarda el modelo entrenado
    st.write("Modelo entrenado y guardado exitosamente!")

# Opción para cargar un modelo previamente entrenado
if st.sidebar.button('Cargar modelo'):
    try:
        model = load_model('modelo_entrenado.pkl')  # Carga el modelo guardado
        st.write("Modelo cargado exitosamente!")
    except FileNotFoundError:
        st.write("No se encontró el modelo entrenado. Por favor, entrena un modelo primero.")

# Opción para predecir y mostrar resultados
st.sidebar.title('Realizar predicción')

# Entrada de datos de predicción
producto = st.text_input('Producto', 'Tomates')
cantidad = st.number_input('Cantidad', min_value=0, value=10)
precio = st.number_input('Precio por unidad', min_value=0.01, value=0.5)

if st.sidebar.button('Hacer predicción'):
    if 'model' not in locals():
        st.write("Por favor, carga o entrena un modelo primero.")
    else:
        # Realizar predicción
        data = pd.DataFrame({'producto': [producto], 'cantidad': [cantidad], 'precio': [precio]})
        data = preprocess_data(data)  # Preprocesar los datos
        prediction = make_prediction(model, data)  # Realizar predicción
        st.write(f"Predicción de desperdicio: {prediction[0]} unidades de {producto}")

# Mostrar importancia de características
if st.sidebar.button('Mostrar importancia de características'):
    try:
        model = load_model('modelo_entrenado.pkl')  # Cargar el modelo guardado
        plot_feature_importance(model, ['producto', 'cantidad', 'precio'])
    except Exception as e:
        st.write(f"Error al cargar el modelo: {e}")

# Sección de cargar y mostrar datos de Supabase (si es necesario)
if st.sidebar.button('Mostrar datos de Supabase'):
    client = get_supabase_client()
    # Suponiendo que tengas una tabla en Supabase llamada 'productos'
    productos = client.table('productos').select('*').execute()
    df_productos = pd.DataFrame(productos['data'])
    st.write("Datos de productos desde Supabase")
    st.write(df_productos)

# Mostrar información adicional en la parte principal de la página
st.header('Información adicional')
st.write("""
    Este es un sistema para predecir la cantidad de productos a comprar, 
    minimizando desperdicios y optimizando el inventario.
""")

