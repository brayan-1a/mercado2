from supabase import fetch_data_from_supabase
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Obtener los datos
ventas = fetch_data_from_supabase('ventas')
inventarios = fetch_data_from_supabase('inventarios')
desperdicio = fetch_data_from_supabase('desperdicio')
promociones = fetch_data_from_supabase('promociones')
clima = fetch_data_from_supabase('condiciones_climaticas')

# Entrenamiento del modelo
def train_model():
    try:
        # Ejemplo de creación del conjunto de datos para el modelo
        X = ventas[['cantidad_vendida', 'precio_unitario']]
        y = ventas['cantidad_a_comprar']

        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entrenar modelo RandomForest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluación
        y_pred = model.predict(X_test)
        error = mean_squared_error(y_test, y_pred)
        print(f"Error cuadrático medio: {error}")

        return model
    except Exception as e:
        print(f"Error durante el entrenamiento del modelo: {e}")
        return None

# Entrenar el modelo
trained_model = train_model()



