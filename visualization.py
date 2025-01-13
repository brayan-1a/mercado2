from supabase import fetch_data_from_supabase
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener los datos
ventas = fetch_data_from_supabase('ventas')

# Visualizaciones
def plot_sales_trends():
    try:
        # Ejemplo: tendencia de ventas
        ventas['fecha'] = pd.to_datetime(ventas['fecha'])
        ventas_grouped = ventas.groupby('fecha')['cantidad_vendida'].sum().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ventas_grouped, x='fecha', y='cantidad_vendida', marker='o')
        plt.title('Tendencia de Ventas')
        plt.xlabel('Fecha')
        plt.ylabel('Cantidad Vendida')
        plt.show()
    except Exception as e:
        print(f"Error al generar la visualización: {e}")

# Generar la visualización
plot_sales_trends()




