from config import Config  # Importar la clase Config
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}


    def preprocess_data(self, sales: pd.DataFrame, inventory: pd.DataFrame,
                       waste: pd.DataFrame, promos: pd.DataFrame,
                       weather: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocesa y combina todos los datasets para el entrenamiento
        """
        # Convertir fechas
        sales['fecha_venta'] = pd.to_datetime(sales['fecha_venta'])
        inventory['fecha_actualizacion'] = pd.to_datetime(inventory['fecha_actualizacion'])
        waste['fecha_registro'] = pd.to_datetime(waste['fecha_registro'])
        weather['fecha'] = pd.to_datetime(weather['fecha'])

        # Agregar características temporales a ventas
        sales['dia_semana'] = sales['fecha_venta'].dt.dayofweek
        sales['mes'] = sales['fecha_venta'].dt.month
        sales['es_fin_semana'] = sales['dia_semana'].isin([5, 6]).astype(int)
        sales['hora'] = pd.to_datetime(sales['hora_venta']).dt.hour

        # Codificar variables categóricas
        categorical_columns = [
            'categoria_producto', 'tipo_producto', 'metodo_pago',
            'canal_venta', 'ubicacion'
        ]
        
        for col in categorical_columns:
            if col in sales.columns:
                self.label_encoders[col] = LabelEncoder()
                sales[f'{col}_encoded'] = self.label_encoders[col].fit_transform(sales[col])

        # Agregar información del clima
        sales = pd.merge_asof(
            sales.sort_values('fecha_venta'),
            weather[['fecha', 'temperatura', 'humedad']].sort_values('fecha'),
            left_on='fecha_venta',
            right_on='fecha',
            direction='nearest'
        )

        # Agregar información de promociones
        sales['en_promocion'] = sales.apply(
            lambda x: promos[
                (promos['producto_id'] == x['producto_id']) &
                (promos['fecha_inicio'] <= x['fecha_venta']) &
                (promos['fecha_fin'] >= x['fecha_venta'])
            ].shape[0] > 0,
            axis=1
        ).astype(int)

        # Calcular métricas de desperdicio
        waste_metrics = waste.groupby(['producto_id', 'fecha_registro'])\
            .agg({'cantidad_perdida': 'sum'})\
            .reset_index()
        
        waste_metrics['desperdicio_7dias'] = waste_metrics.groupby('producto_id')\
            ['cantidad_perdida'].rolling(window=7, min_periods=1).mean()\
            .reset_index(0, drop=True)

        sales = pd.merge_asof(
            sales.sort_values('fecha_venta'),
            waste_metrics.sort_values('fecha_registro'),
            left_on='fecha_venta',
            right_on='fecha_registro',
            by='producto_id',
            direction='backward'
        )

        # Crear características para el modelo
        feature_columns = [
            'dia_semana', 'mes', 'es_fin_semana', 'hora',
            'temperatura', 'humedad', 'en_promocion',
            'desperdicio_7dias', 'precio_unitario',
            'categoria_producto_encoded', 'tipo_producto_encoded',
            'canal_venta_encoded'
        ]

        # Escalar características numéricas
        self.scalers['features'] = StandardScaler()
        X = pd.DataFrame(
            self.scalers['features'].fit_transform(sales[feature_columns]),
            columns=feature_columns
        )

        y = sales['cantidad_vendida']

        return X, y

