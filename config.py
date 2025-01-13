from dataclasses import dataclass
import os
from dotenv import load_dotenv
import streamlit as st

@dataclass
class Config:
    SUPABASE_URL: str = None
    SUPABASE_KEY: str = None
    FORECAST_DAYS: int = 7
    MIN_STOCK_THRESHOLD: float = 0.2
    STORAGE_DAYS: Dict[str, int] = None
    
    def __post_init__(self):
        # Cargar variables de entorno si existe .env
        load_dotenv()
        
        # Intentar obtener credenciales de Streamlit secrets primero
        try:
            self.SUPABASE_URL = st.secrets["SUPABASE_URL"]
            self.SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
        except:
            # Si no hay secrets, intentar variables de entorno
            self.SUPABASE_URL = os.getenv('SUPABASE_URL')
            self.SUPABASE_KEY = os.getenv('SUPABASE_KEY')
            
            # Si aún no hay credenciales, usar valores por defecto (solo para desarrollo)
            if not self.SUPABASE_URL or not self.SUPABASE_KEY:
                self.SUPABASE_URL = 'https://odlosqyzqrggrhvkdovj.supabase.co'  # Cambia esta URL por la correcta
                self.SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'  # Usa la clave correcta aquí
                print("WARNING: Using default credentials. Consider using environment variables or Streamlit secrets.")
        # Aquí puedes imprimir para asegurarte de que las credenciales se cargan bien
        print(f"URL: {self.SUPABASE_URL}, KEY: {self.SUPABASE_KEY}")

        # Configurar días de almacenamiento por categoría
        self.STORAGE_DAYS = {
            'VERDURAS': 7,
            'FRUTAS': 5,
            'TUBERCULOS': 14
        }





