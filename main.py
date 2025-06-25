import pandas as pd

# Define la URL de tu archivo CSV en GitHub
# Esta es la URL "raw" o directa al contenido del archivo
csv_url = "https://media.githubusercontent.com/media/JulianTorrest/prueba_real_state/main/data/Real_Estate_Sales_2001-2022_GL.csv"

try:
    # Lee el archivo CSV directamente desde la URL en un DataFrame de pandas
    df = pd.read_csv(csv_url)

    # Muestra las primeras filas del DataFrame para verificar
    print("CSV leído exitosamente. Primeras 5 filas:")
    print(df.head())

    # Muestra el número de filas y columnas
    print(f"\nNúmero de filas y columnas: {df.shape}")

    # Opcional: Obtén información básica sobre el DataFrame
    # print("\nInformación del DataFrame:")
    # df.info()

except Exception as e:
    print(f"Error al intentar leer el CSV desde la URL: {e}")
    print("Asegúrate de que la URL sea correcta y que el archivo sea accesible.")
