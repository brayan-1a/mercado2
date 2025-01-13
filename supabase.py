from supabase import create_client, Client

# URL y API Key de Supabase
URL = 'https://odlosqyzqrggrhvkdovj.supabase.co'
KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9kbG9zcXl6cXJnZ3Jodmtkb3ZqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzAwNjgyODksImV4cCI6MjA0NTY0NDI4OX0.z5btFX44Eu30kOBJj7eZKAmOUG62IrTcpXUVhMqK9Ck'
supabase: Client = create_client(url, key)

# Función para obtener datos de una tabla en Supabase
def fetch_data_from_supabase(table_name: str):
    data = supabase.table(table_name).select("*").execute()
    return pd.DataFrame(data.data)