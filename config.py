from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Config:
    SUPABASE_URL: str = 'https://odlosqyzqrggrhvkdovj.supabase.co'
    SUPABASE_KEY: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'
    FORECAST_DAYS: int = 7
    MIN_STOCK_THRESHOLD: float = 0.2
    
    # Configuraciones específicas por categoría de producto
    STORAGE_DAYS: Dict[str, int] = None
    
    def __post_init__(self):
        self.STORAGE_DAYS = {
            'VERDURAS': 7,
            'FRUTAS': 5,
            'TUBERCULOS': 14
        }




