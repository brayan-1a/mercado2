import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

def clean_data(df):
    """Elimina valores nulos y realiza limpieza básica de los datos."""
    # Eliminar filas con valores nulos
    df = df.dropna()
    return df

def feature_engineering(df):
    """Crea nuevas características o transforma las existentes."""
    # Ejemplo: convertir la fecha en variables de año y mes
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
    df['año_venta'] = df['fecha_venta'].dt.year
    df['mes_venta'] = df['fecha_venta'].dt.month
    
    # Puede agregar otras transformaciones si es necesario
    
    return df

def encode_categorical_features(df):
    """Codifica variables categóricas."""
    le = LabelEncoder()
    df['metodo_pago'] = le.fit_transform(df['metodo_pago'])  # Ejemplo de codificación
    
    # Otras columnas categóricas que necesiten codificación podrían añadirse aquí
    return df

def normalize_data(df):
    """Normaliza las características numéricas."""
    scaler = StandardScaler()
    
    # Normalizamos las características numéricas (ej. precio, cantidad)
    df[['precio_total', 'cantidad_vendida']] = scaler.fit_transform(df[['precio_total', 'cantidad_vendida']])
    
    return df

def preprocess(df):
    """Realiza todas las tareas de preprocesamiento en el dataframe."""
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categorical_features(df)
    df = normalize_data(df)
    
    return df

def main():
    """Función principal para preprocesar los datos."""
    # Simulando que obtienes los datos desde Supabase
    # Si los datos vienen de la base de datos, harías algo como:
    # df = obtener_datos_desde_supabase()

    # Como ejemplo, creamos un DataFrame simple
    df = pd.DataFrame({
        'venta_id': [1, 2, 3],
        'producto_id': [101, 102, 103],
        'cliente_id': [1001, 1002, 1003],
        'fecha_venta': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'cantidad_vendida': [10, 15, 7],
        'precio_total': [50, 75, 35],
        'descuento_aplicado': [5, 7, 3],
        'metodo_pago': ['Tarjeta', 'Efectivo', 'Tarjeta'],
        'hora_venta': ['10:00', '11:30', '14:00'],
        'canal_venta': ['Online', 'Tienda', 'Online'],
        'ubicacion': ['Lima', 'Arequipa', 'Cusco']
    })
    
    # Preprocesar los datos
    df_cleaned = preprocess(df)
    print(df_cleaned)

if __name__ == '__main__':
    main()
