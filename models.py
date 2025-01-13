from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from typing import Dict, Any, Tuple
import joblib

class PredictionModel:
    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[int, Any] = {}  # Usar product_id como key
        
    def train(self, X: pd.DataFrame, y: pd.Series, product_id: int) -> Tuple[float, float]:
        """
        Entrena modelos para cada producto y retorna métricas de rendimiento
        """
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Entrenar y evaluar Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        rf_mse = []
        rf_mape = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_mse.append(mean_squared_error(y_test, y_pred))
            rf_mape.append(mean_absolute_percentage_error(y_test, y_pred))

        # Entrenar y evaluar Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        
        gb_mse = []
        gb_mape = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            gb_mse.append(mean_squared_error(y_test, y_pred))
            gb_mape.append(mean_absolute_percentage_error(y_test, y_pred))

        # Usar el modelo con mejor rendimiento
        if np.mean(rf_mse) < np.mean(gb_mse):
            self.models[product_id] = rf.fit(X, y)
            return np.mean(rf_mse), np.mean(rf_mape)
        else:
            self.models[product_id] = gb.fit(X, y)
            return np.mean(gb_mse), np.mean(gb_mape)

    def predict_stock(self, X: pd.DataFrame, product_id: int) -> np.ndarray:
        """Predice la demanda futura para un producto"""
        if product_id not in self.models:
            raise ValueError(f"No hay modelo entrenado para el producto {product_id}")
        
        return self.models[product_id].predict(X)

    def get_feature_importance(self, product_id: int) -> Dict[str, float]:
        """Retorna la importancia de las características para un producto"""
        if product_id not in self.models:
            raise ValueError(f"No hay modelo entrenado para el producto {product_id}")
        
        model = self.models[product_id]
        importance = model.feature_importances_
        features = model.feature_names_in_
        
        return dict(zip(features, importance))


