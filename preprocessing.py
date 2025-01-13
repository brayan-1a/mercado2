from supabase import fetch_data_from_supabase

# Obtener los datos de Supabase
ventas = fetch_data_from_supabase('ventas')
inventarios = fetch_data_from_supabase('inventarios')
desperdicio = fetch_data_from_supabase('desperdicio')
promociones = fetch_data_from_supabase('promociones')
clima = fetch_data_from_supabase('condiciones_climaticas')

# Preprocesamiento de los datos
def preprocess_data():
    try:
        # Ejemplo de limpieza: eliminar filas nulas
        ventas.dropna(inplace=True)
        inventarios.dropna(inplace=True)
        desperdicio.dropna(inplace=True)
        
        # Conversiones de formato, eliminación de duplicados, etc.
        ventas['fecha'] = pd.to_datetime(ventas['fecha'])
        return ventas, inventarios, desperdicio, promociones, clima
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")
        return None

# Llamada al preprocesamiento
preprocessed_data = preprocess_data()

