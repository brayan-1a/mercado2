import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from config import get_supabase_client  # Importar la función para obtener la conexión

# Obtener el cliente de Supabase
supabase = get_supabase_client()

# Cargar los datos de ventas desde Supabase
def load_data_from_supabase():
    """Cargar datos de ventas desde Supabase"""
    ventas_data = supabase.table('venta').select('*').execute()
    ventas_df = pd.DataFrame(ventas_data.data)
    return ventas_df

# Cargar los datos
df_ventas = load_data_from_supabase()

# Preprocesamiento de los datos
def preprocess_data(df):
    """Realizar el preprocesamiento de los datos"""
    # Convertir la fecha a un formato numérico (por ejemplo, días desde el inicio)
    df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
    df['dias_desde_inicio'] = (df['fecha_venta'] - df['fecha_venta'].min()).dt.days
    
    # Extraer la hora de venta en formato numérico
    df['hora_venta'] = pd.to_datetime(df['hora_venta'], format='%H:%M:%S').dt.hour
    
    # Variables que vamos a usar para predecir (p.ej. cantidad vendida y precio total)
    df = df[['dias_desde_inicio', 'hora_venta', 'cantidad_vendida', 'precio_total', 'descuento_aplicado']]
    
    # Eliminar posibles valores nulos
    df = df.dropna()

    return df

# Preprocesar los datos de ventas
df_ventas_preprocesados = preprocess_data(df_ventas)

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X = df_ventas_preprocesados.drop('cantidad_vendida', axis=1)  # Usamos todas las características menos la variable a predecir
y = df_ventas_preprocesados['cantidad_vendida']  # Variable objetivo (cantidad vendida)

# Dividir en conjunto de entrenamiento y validación (80% y 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Hacer predicciones sobre los datos de validación
y_pred = modelo.predict(X_val)

# Evaluación del modelo
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Error absoluto medio (MAE): {mae}")
print(f"Error cuadrático medio (MSE): {mse}")
print(f"R2 Score: {r2}")

# Exportar el modelo entrenado a un archivo .pkl para su uso futuro
joblib.dump(modelo, 'modelo_predictivo_rf.pkl')

# Función para predecir la cantidad de producto a comprar
def predecir_cantidad_a_comprar(fecha_venta, hora_venta, descuento_aplicado):
    """Predecir la cantidad de producto a comprar usando el modelo entrenado"""
    dias_desde_inicio = (pd.to_datetime(fecha_venta) - pd.to_datetime(df_ventas['fecha_venta'].min())).days
    hora_venta = pd.to_datetime(hora_venta, format='%H:%M:%S').hour

    # Preparar los datos de entrada para la predicción
    datos_entrada = np.array([[dias_desde_inicio, hora_venta, descuento_aplicado]])
    
    # Predecir la cantidad
    cantidad_predicha = modelo.predict(datos_entrada)
    return cantidad_predicha[0]

# Ejemplo de predicción de cantidad a comprar (puedes probar con tus datos)
fecha_venta = '2022-01-01'
hora_venta = '14:30:00'
descuento_aplicado = 0.1

cantidad_comprar = predecir_cantidad_a_comprar(fecha_venta, hora_venta, descuento_aplicado)
print(f"Cantidad recomendada a comprar: {cantidad_comprar}")
