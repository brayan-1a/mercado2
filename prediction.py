import pickle
import pandas as pd
from config import get_supabase_client

def load_model():
    """Carga el modelo previamente entrenado desde el archivo .pkl"""
    with open('modelo_predictivo.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def get_new_data_from_supabase():
    """Obtiene nuevos datos de Supabase para hacer predicciones"""
    supabase = get_supabase_client()
    nuevas_ventas = supabase.table('nuevas_ventas').select('*').execute().data
    df = pd.DataFrame(nuevas_ventas)
    return df

def make_prediction(model, data):
    """
    Hace la predicción usando el modelo cargado y los datos de entrada.
    El modelo toma los datos de entrada y realiza la predicción de acuerdo con las características esperadas.
    """
    if 'precio_total' not in data.columns or 'descuento_aplicado' not in data.columns or 'metodo_pago' not in data.columns:
        raise ValueError("Las columnas necesarias no están presentes en los datos de entrada.")
    
    X_new = data[['precio_total', 'descuento_aplicado', 'metodo_pago']]  # Selección de columnas
    prediction = model.predict(X_new)  # Predicción
    return prediction

def main():
    model = load_model()  # Cargar el modelo
    new_data = get_new_data_from_supabase()  # Obtener nuevos datos de Supabase
    predictions = make_prediction(model, new_data)  # Hacer la predicción
    
    # Agregar las predicciones a los datos
    new_data['prediccion'] = predictions
    print(new_data)

if __name__ == '__main__':
    main()



