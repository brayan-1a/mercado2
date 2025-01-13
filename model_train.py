import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # O el modelo que decidas usar
from config import get_supabase_client

def get_data_from_supabase():
    """Obtiene datos de Supabase para entrenar el modelo."""
    supabase = get_supabase_client()
    # Suponiendo que tienes una tabla llamada 'ventas_historicas' con las columnas necesarias
    ventas = supabase.table('ventas_historicas').select('*').execute().data
    df = pd.DataFrame(ventas)
    return df

def train_model(df):
    """Entrena un modelo con los datos de ventas históricos."""
    X = df[['precio_total', 'descuento_aplicado', 'metodo_pago']]  # Variables independientes
    y = df['ventas']  # La variable dependiente (lo que queremos predecir)

    model = RandomForestRegressor()  # Usamos RandomForest como ejemplo
    model.fit(X, y)  # Entrenamos el modelo con los datos
    return model

def save_model(model):
    """Guarda el modelo entrenado en un archivo .pkl."""
    with open('modelo_predictivo.pkl', 'wb') as f:
        pickle.dump(model, f)

def main():
    df = get_data_from_supabase()  # Obtén los datos desde Supabase
    model = train_model(df)  # Entrena el modelo
    save_model(model)  # Guarda el modelo entrenado

if __name__ == '__main__':
    main()


