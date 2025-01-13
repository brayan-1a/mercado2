from supabase import create_client, Client
import pandas as pd
from typing import Optional
from datetime import datetime, timedelta

class SupabaseConnector:
    def __init__(self, config: Config):
        self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.config = config

    def get_products(self) -> pd.DataFrame:
        """Obtiene la lista de productos con sus detalles"""
        query = self.supabase.table('productos').select('*').execute()
        return pd.DataFrame(query.data)

    def get_sales_data(self, start_date: datetime) -> pd.DataFrame:
        """Obtiene datos de ventas con información detallada del producto"""
        query = """
        SELECT v.*, 
               p.nombre_producto,
               p.categoria_producto,
               p.tipo_producto,
               p.precio_unitario,
               c.nombre_cliente,
               c.clientes_frecuentes,
               c.ubicacion_cliente
        FROM ventas v
        JOIN productos p ON v.producto_id = p.producto_id
        JOIN clientes c ON v.cliente_id = c.cliente_id
        WHERE v.fecha_venta >= :start_date
        """
        result = self.supabase.rpc('custom_query', {
            'query': query,
            'start_date': start_date.strftime('%Y-%m-%d')
        }).execute()
        return pd.DataFrame(result.data)

    def get_inventory_data(self, start_date: datetime) -> pd.DataFrame:
        """Obtiene datos de inventario con información del producto"""
        query = """
        SELECT i.*,
               p.nombre_producto,
               p.categoria_producto
        FROM inventarios i
        JOIN productos p ON i.producto_id = p.producto_id
        WHERE i.fecha_actualizacion >= :start_date
        """
        result = self.supabase.rpc('custom_query', {
            'query': query,
            'start_date': start_date.strftime('%Y-%m-%d')
        }).execute()
        return pd.DataFrame(result.data)

    def get_waste_data(self, start_date: datetime) -> pd.DataFrame:
        """Obtiene datos de desperdicios con información del producto"""
        query = """
        SELECT d.*,
               p.nombre_producto,
               p.categoria_producto
        FROM desperdicio d
        JOIN productos p ON d.producto_id = p.producto_id
        WHERE d.fecha_registro >= :start_date
        """
        result = self.supabase.rpc('custom_query', {
            'query': query,
            'start_date': start_date.strftime('%Y-%m-%d')
        }).execute()
        return pd.DataFrame(result.data)

    def get_promotions(self) -> pd.DataFrame:
        """Obtiene datos de promociones activas"""
        query = """
        SELECT pr.*,
               p.nombre_producto,
               p.categoria_producto
        FROM promociones pr
        JOIN productos p ON pr.producto_id = p.producto_id
        WHERE pr.fecha_fin >= CURRENT_DATE
        """
        result = self.supabase.rpc('custom_query', {
            'query': query
        }).execute()
        return pd.DataFrame(result.data)

    def get_weather_data(self, start_date: datetime) -> pd.DataFrame:
        """Obtiene datos climáticos"""
        query = self.supabase.table('condiciones_climaticas')\
            .select('*')\
            .gte('fecha', start_date.strftime('%Y-%m-%d'))\
            .execute()
        return pd.DataFrame(query.data)