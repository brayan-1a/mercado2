import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from config import Config
from supabase_connector import SupabaseConnector
from preprocessing import DataPreprocessor
from models import PredictionModel

def main():
    st.title("Sistema Predictivo de Stock para Verduras 🥬")
    
    # Configuración
    config = Config(
        SUPABASE_URL=st.secrets["SUPABASE_URL"],
        SUPABASE_KEY=st.secrets["SUPABASE_KEY"]
    )
    
    # Inicializar componentes
    connector = SupabaseConnector(config)
    preprocessor = DataPreprocessor(config)
    model = PredictionModel(config)
    
    # Obtener lista de productos
    products_df = connector.get_products()
    
    # Sidebar para controles
    st.sidebar.header("Configuración")
    selected_product = st.sidebar.selectbox(
        "Seleccionar Producto",
        products_df['nombre_producto'].tolist()
    )
    
    selected_product_id = products_df[
        products_df['nombre_producto'] == selected_product
    ]['producto_id'].iloc[0]
    
    # Pestañas principales
    tabs = st.tabs(["Predicciones", "Análisis", "Configuración"])
    
    with tabs[0]:
        st.header("Predicciones de Demanda")
        
        if st.button("Actualizar Predicciones"):
            with st.spinner("Obteniendo datos..."):
                # Obtener datos de los últimos 6 meses
                start_date = datetime.now() - timedelta(days=180)
                
                # Obtener todos los datos necesarios
                sales_data = connector.get_sales_data(start_date)
                inventory_data = connector.get_inventory_data(start_date)
                waste_data = connector.get_waste_data(start_date)
                promos_data = connector.get_promotions()
                weather_data = connector.get_weather_data(start_date)
                
                # Preprocesar datos
                X, y = preprocessor.preprocess_data(
                    sales_data,
                    inventory_data,
                    waste_data,
                    promos_data,
                    weather_data
                )
                
                # Entrenar modelo y obtener métricas
                mse, mape = model.train(X, y, selected_product_id)
                
                st.info(f"""
                    Métricas del modelo:
                    - Error cuadrático medio: {mse:.2f}
                    - Error porcentual medio absoluto: {mape:.2%}
                """)
                
                # Generar predicciones
                future_dates = pd.date_range(
                    start=datetime.now(),
                    periods=config.FORECAST_DAYS,
                    freq='D'
                )



