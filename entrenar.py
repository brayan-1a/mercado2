import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import streamlit as st
import config  # Importamos el archivo config.py

# Conectar con Supabase usando la función get_supabase_client de config.py
supabase = config.get_supabase_client()

# Función para cargar las tablas desde Supabase
def cargar_tablas():
    try:
        df_venta = supabase.table('ventas').select('*').execute().data
        df_producto = supabase.table('productos').select('*').execute().data
        df_cliente = supabase.table('clientes').select('*').execute().data
        df_promociones = supabase.table('promociones').select('*').execute().data
        df_condiciones_climaticas = supabase.table('condiciones_climaticas').select('*').execute().data
        
        # Convertir las respuestas en DataFrames de pandas
        df_venta = pd.DataFrame(df_venta)
        df_producto = pd.DataFrame(df_producto)
        df_cliente = pd.DataFrame(df_cliente)
        df_promociones = pd.DataFrame(df_promociones)
        df_condiciones_climaticas = pd.DataFrame(df_condiciones_climaticas)

        return df_venta, df_producto, df_cliente, df_promociones, df_condiciones_climaticas
    
    except Exception as e:
        print(f"Error al cargar las tablas desde Supabase: {e}")
        return None

# Cargar las tablas
df_venta, df_producto, df_cliente, df_promociones, df_condiciones_climaticas = cargar_tablas()

# Asegurarse de que las tablas no son None
if df_venta is not None:
    # Limpiar y estandarizar nombres de columnas (si es necesario)
    df_venta.columns = df_venta.columns.str.strip().str.lower()  # Estandarizar nombres de columnas
    
    # Asegurarse de que 'cantidad_vendida' y 'descuento_aplicado' sean numéricos
    df_venta['cantidad_vendida'] = pd.to_numeric(df_venta['cantidad_vendida'], errors='coerce')
    df_venta['descuento_aplicado'] = pd.to_numeric(df_venta['descuento_aplicado'], errors='coerce')
    
    # Asegurarse de que 'fecha_venta' sea de tipo datetime antes de convertir a timestamp
    df_venta['fecha_venta'] = pd.to_datetime(df_venta['fecha_venta'], errors='coerce')
    
    # Verificar las primeras filas del DataFrame para confirmar que se carga correctamente
    print("Primeras filas de df_venta:", df_venta.head())
    
    # Hacer el merge de la tabla ventas con otras tablas relevantes
    df_venta = df_venta.merge(df_producto, on='producto_id', how='left')
    df_venta = df_venta.merge(df_cliente, on='cliente_id', how='left')
    df_venta = df_venta.merge(df_promociones, on='producto_id', how='left')

    # Verificar si las columnas necesarias existen
    if all(col in df_venta.columns for col in ['fecha_venta', 'descuento_aplicado', 'cantidad_vendida']):
        # Selección de características para el modelo
        X = df_venta[['fecha_venta', 'descuento_aplicado']]  # Aquí seleccionamos las columnas que necesitas
        
        # Convertir 'fecha_venta' a timestamp
        X['fecha_venta'] = X['fecha_venta'].map(lambda x: x.timestamp() if pd.notna(x) else 0)
        
        y = df_venta['cantidad_vendida']  # Variable objetivo

        # Dividir los datos en entrenamiento (80%) y prueba (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar el modelo
        modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Predicciones y evaluación
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)

        # Mostrar el error en Streamlit
        st.write(f"Error absoluto medio (MAE) en las predicciones: {mae}")

        # Guardar el modelo entrenado
        with open('random_forest_model.pkl', 'wb') as f:
            pickle.dump(modelo, f)

        # Función de predicción
        def predecir_cantidad_a_comprar(fecha_venta, descuento_aplicado):
            fecha_venta_timestamp = datetime.strptime(fecha_venta, '%Y-%m-%d').timestamp()
            datos_entrada = pd.DataFrame({
                'fecha_venta': [fecha_venta_timestamp],
                'descuento_aplicado': [descuento_aplicado]
            })
            cantidad_predicha = modelo.predict(datos_entrada)
            return cantidad_predicha[0]

        # Mostrar en Streamlit
        st.title("Predicción de Cantidad a Comprar")
        fecha_venta_input = st.date_input("Fecha de la venta", datetime.today())
        descuento_aplicado_input = st.slider("Descuento aplicado (%)", 0.0, 20.0, 10.0)

        if st.button("Predecir cantidad a comprar"):
            cantidad = predecir_cantidad_a_comprar(str(fecha_venta_input), descuento_aplicado_input)
            st.write(f"La cantidad recomendada a comprar es: {cantidad} unidades")

        # Funcionalidad para descargar el modelo
        @st.cache
        def descargar_modelo():
            return 'random_forest_model.pkl'

        if st.button("Descargar modelo entrenado"):
            st.download_button(
                label="Descargar modelo",
                data=open(descargar_modelo(), "rb").read(),
                file_name="random_forest_model.pkl",
                mime="application/octet-stream"
            )
    else:
        print("Faltan algunas columnas necesarias en df_venta")
        print("Columnas disponibles:", df_venta.columns)
else:
    st.write("Hubo un error al cargar las tablas desde Supabase.")




