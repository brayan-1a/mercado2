import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from supabase import create_client
import pickle
import streamlit as st

# URL y API Key de Supabase
URL = 'https://odlosqyzqrggrhvkdovj.supabase.co'
KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Cky'

def get_supabase_client():
    """Crea y devuelve una instancia del cliente Supabase."""
    return create_client(URL, KEY)

# Conectar con la base de datos de Supabase
supabase = get_supabase_client()

# Cargar las tablas necesarias desde Supabase
df_venta = supabase.table('ventas').select('*').execute().data
df_producto = supabase.table('productos').select('*').execute().data
df_cliente = supabase.table('clientes').select('*').execute().data
df_promociones = supabase.table('promociones').select('*').execute().data
df_condiciones_climaticas = supabase.table('condiciones_climaticas').select('*').execute().data

# Convertir a DataFrames de Pandas
df_venta = pd.DataFrame(df_venta)
df_producto = pd.DataFrame(df_producto)
df_cliente = pd.DataFrame(df_cliente)
df_promociones = pd.DataFrame(df_promociones)
df_condiciones_climaticas = pd.DataFrame(df_condiciones_climaticas)

# Verificar las columnas de cada DataFrame
st.write("Columnas de df_venta:", df_venta.columns)
st.write("Columnas de df_producto:", df_producto.columns)
st.write("Columnas de df_cliente:", df_cliente.columns)

# Preprocesamiento de datos
# Aseguramos que las fechas estén en el formato correcto
df_venta['fecha_venta'] = pd.to_datetime(df_venta['fecha_venta'])
df_condiciones_climaticas['fecha'] = pd.to_datetime(df_condiciones_climaticas['fecha'])

# Mostrar algunas filas de cada DataFrame para verificar que los datos sean correctos
st.write("Primeras 5 filas de df_venta:", df_venta.head())
st.write("Primeras 5 filas de df_producto:", df_producto.head())
st.write("Primeras 5 filas de df_cliente:", df_cliente.head())

# Hacer el merge de la tabla ventas con otras tablas relevantes
df_venta = df_venta.merge(df_producto, on='producto_id', how='left')
st.write("Después de merge con df_producto:", df_venta.head())

df_venta = df_venta.merge(df_cliente, on='cliente_id', how='left')
st.write("Después de merge con df_cliente:", df_venta.head())

df_venta = df_venta.merge(df_promociones, on='producto_id', how='left')
st.write("Después de merge con df_promociones:", df_venta.head())

# Ahora necesitamos una columna adicional para las predicciones, por ejemplo: cantidad_vendida
df_venta['cantidad_vendida'] = df_venta['cantidad_vendida'].astype(float)

# Verificar las columnas disponibles después de todos los merges
st.write("Columnas finales de df_venta:", df_venta.columns)

# Supongamos que queremos predecir la cantidad a comprar para evitar el desperdicio
# Utilizaremos columnas como fecha_venta, descuento_aplicado, etc., como características de entrada
X = df_venta[['fecha_venta', 'descuento_aplicado', 'cantidad_vendida']]
X['fecha_venta'] = X['fecha_venta'].map(lambda x: x.timestamp())  # Convertir fecha a timestamp para usarla en el modelo

# Variable objetivo (target), que será la cantidad de productos a comprar
y = df_venta['cantidad_vendida']

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Error absoluto medio (MAE) en las predicciones: {mae}")
