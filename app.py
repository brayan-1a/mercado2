import streamlit as st
from supabase_connector import fetch_data_from_supabase

# Configuración de la página
st.set_page_config(page_title="Dashboard de Compras y Ventas", layout="wide")

# Títulos
st.title("Dashboard de Predicción de Compras y Reducción de Desperdicios")

# Cargar datos
ventas = fetch_data_from_supabase('ventas')

# Mostrar datos
st.subheader("Vista General de Ventas")
st.dataframe(ventas)

# Ejemplo de visualización
st.subheader("Visualización de Tendencias")
if not ventas.empty:
    ventas['fecha'] = pd.to_datetime(ventas['fecha'])
    ventas_grouped = ventas.groupby('fecha')['cantidad_vendida'].sum().reset_index()
    st.line_chart(data=ventas_grouped, x='fecha', y='cantidad_vendida')



