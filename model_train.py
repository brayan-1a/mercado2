import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from config import get_supabase_client
import pandas as pd

def get_data_from_supabase():
    """Obtiene los datos necesarios desde Supabase"""
    supabase = get_supabase_client()
    # Suponiendo que tienes una tabla 'ventas' con las columnas necesarias
    ventas_data = supabase.table('ventas').select('*').execute().data
    df = pd.DataFrame(ventas_data)
    return df

def preprocess_data(df):
    """Realiza el preprocesamiento de los datos"""
    # Aquí puedes limpiar o modificar los datos según sea necesario
    df = df.dropna()  # Ejemplo de eliminar filas con valores faltantes
    # Suponiendo que 'cantidad_vendida' es lo que quieres predecir
    X = df[['precio_total', 'descuento_aplicado', 'metodo_pago']]  # Variables independientes
    y = df['cantidad_vendida']  # Variable dependiente
    return X, y

def train_model(X, y):
    """Entrena el modelo y lo guarda en un archivo"""
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio: {mse}")
    
    # Guardar el modelo entrenado
    with open('modelo_predictivo.pkl', 'wb') as f:
        pickle.dump(model, f)

def main():
    df = get_data_from_supabase()
    X, y = preprocess_data(df)
    train_model(X, y)

if __name__ == '__main__':
    main()
