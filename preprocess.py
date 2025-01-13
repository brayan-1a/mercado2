import pandas as pd

def preprocess_data(df):
    """Preprocesa los datos antes de entrenar el modelo"""
    # Realiza las transformaciones necesarias, como:
    # - Eliminar columnas no relevantes
    # - Rellenar valores nulos
    # - Convertir columnas categóricas en numéricas, etc.
    df = df.dropna()  # Ejemplo simple
    return df

