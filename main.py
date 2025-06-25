import pandas as pd
import streamlit as st

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(
    page_title="Análisis de Ventas Inmobiliarias",
    page_icon="🏠",
    layout="wide"
)

st.title("🏡 Análisis de Datos de Ventas Inmobiliarias (2001-2022)")
st.markdown("---")

# --- Define la URL de tu archivo CSV en GitHub ---
# Esta es la URL "raw" o directa al contenido del archivo
# Asegúrate de que esta URL sea correcta y apunte a tu archivo en GitHub
CSV_URL = "https://media.githubusercontent.com/media/JulianTorrest/prueba_real_state/main/data/Real_Estate_Sales_2001-2022_GL.csv"

@st.cache_data # Decorador para cachear los datos y evitar recargar en cada interacción
def load_data(url):
    """Carga el archivo CSV desde la URL especificada."""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error al intentar cargar el archivo CSV: {e}")
        st.info("Asegúrate de que la URL sea correcta y que el archivo sea accesible.")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

# --- Cargar los Datos ---
st.header("Cargando Datos...")
df = load_data(CSV_URL)

if not df.empty:
    st.success("¡Datos cargados exitosamente!")

    # --- Muestra las primeras filas del DataFrame ---
    st.subheader("Primeras 5 Filas del Conjunto de Datos")
    st.dataframe(df.head())

    # --- Muestra información básica del DataFrame ---
    st.subheader("Información General del DataFrame")
    st.write(f"**Número de filas:** {df.shape[0]}")
    st.write(f"**Número de columnas:** {df.shape[1]}")

    st.subheader("Estadísticas Descriptivas de Columnas Numéricas")
    st.dataframe(df.describe())

    # --- Ejemplo simple de visualización (puedes expandir esto) ---
    st.subheader("Distribución de 'SalePrice' (Precio de Venta)")
    if 'SalePrice' in df.columns:
        st.hist_chart(df['SalePrice'])
    else:
        st.warning("La columna 'SalePrice' no se encontró en el DataFrame para la visualización.")

    st.subheader("Filtrado Básico")
    # Ejemplo de filtro interactivo
    if 'SalePrice' in df.columns:
        min_price, max_price = float(df['SalePrice'].min()), float(df['SalePrice'].max())
        price_range = st.slider(
            "Selecciona un rango de precios:",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price)
        )
        filtered_df = df[(df['SalePrice'] >= price_range[0]) & (df['SalePrice'] <= price_range[1])]
        st.write(f"Mostrando {filtered_df.shape[0]} registros dentro del rango de precio seleccionado.")
        st.dataframe(filtered_df.head())
    else:
        st.info("No se puede aplicar el filtro de precio sin la columna 'SalePrice'.")

else:
    st.error("No se pudieron cargar los datos. No se puede continuar con el análisis.")

st.markdown("---")
st.caption("Aplicación de ejemplo para análisis de ventas inmobiliarias. Datos de JulianTorrest/prueba_real_state.")
