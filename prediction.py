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
    # Suponiendo que tienes una tabla 'nuevas_ventas' con las columnas necesarias
    nuevas_ventas = supabase.table('nuevas_ventas').select('*').execute().data
    df = pd.DataFrame(nuevas_ventas)
    return df

def predict(model, new_data):
    """Realiza la predicción con los nuevos datos"""
    X_new = new_data[['precio_total', 'descuento_aplicado', 'metodo_pago']]  # Variables independientes
    predictions = model.predict(X_new)
    return predictions

def main():
    model = load_model()
    new_data = get_new_data_from_supabase()
    predictions = predict(model, new_data)
    
    # Mostrar las predicciones o hacer algo con ellas
    new_data['prediccion'] = predictions
    print(new_data)

if __name__ == '__main__':
    main()
