import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path):
    """Carga el modelo desde un archivo .pkl."""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        raise FileNotFoundError(f"El archivo de modelo {model_path} no se encuentra.")

def save_model(model, model_path):
    """Guarda el modelo en un archivo .pkl."""
    joblib.dump(model, model_path)

def plot_feature_importance(model, feature_names, num_features=10):
    """Genera una gráfica de importancia de características del modelo."""
    # Asumiendo que el modelo tiene un atributo 'feature_importances_' como un modelo de árboles
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
        feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(num_features)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title("Importancia de Características")
        plt.show()
    else:
        raise AttributeError("El modelo no tiene el atributo 'feature_importances_'")

def load_data_from_csv(file_path):
    """Carga datos desde un archivo CSV."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"No se encontró el archivo CSV en {file_path}")

def save_data_to_csv(df, file_path):
    """Guarda un DataFrame en un archivo CSV."""
    df.to_csv(file_path, index=False)

def get_file_extension(file_path):
    """Obtiene la extensión del archivo."""
    return os.path.splitext(file_path)[1]

def check_column_exists(df, column_name):
    """Verifica si una columna existe en un DataFrame."""
    if column_name in df.columns:
        return True
    else:
        raise ValueError(f"La columna {column_name} no existe en el DataFrame")

def clean_column_names(df):
    """Limpia los nombres de las columnas, eliminando espacios y convirtiéndolos a minúsculas."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

