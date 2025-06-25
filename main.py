import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np # Para manejo de NaN y operaciones numéricas

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(
    page_title="EDA de Ventas Inmobiliarias",
    page_icon="🏠",
    layout="wide" # Usa todo el ancho de la pantalla
)

st.title("🏡 Análisis Exploratorio de Datos (EDA) de Ventas Inmobiliarias")
st.markdown("""
    Esta aplicación interactiva realiza un Análisis Exploratorio de Datos (EDA)
    sobre el conjunto de datos de ventas inmobiliarias de Connecticut (2001-2022).
    Los datos se cargan directamente desde GitHub.
""")
st.markdown("---")

# --- Define la URL de tu archivo CSV en GitHub ---
CSV_URL = "https://media.githubusercontent.com/media/JulianTorrest/prueba_real_state/main/data/Real_Estate_Sales_2001-2022_GL.csv"

@st.cache_data(show_spinner="Cargando y preprocesando datos...")
def load_data(url):
    """
    Carga el archivo CSV y realiza un preprocesamiento inicial.
    """
    try:
        df = pd.read_csv(url)

        # Renombrar columnas para facilitar el uso (opcional, pero buena práctica)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('_-', '_').str.replace('.', '').str.lower()
        # Puedes renombrar manualmente si lo prefieres:
        # df = df.rename(columns={'Serial Number': 'serial_number', 'List Year': 'list_year', ...})

        # Conversión de tipos de datos
        # Date Recorded a datetime
        df['date_recorded'] = pd.to_datetime(df['date_recorded'], errors='coerce')

        # Asegurar que 'assessed_value' y 'sale_amount' sean numéricos
        df['assessed_value'] = pd.to_numeric(df['assessed_value'], errors='coerce')
        df['sale_amount'] = pd.to_numeric(df['sale_amount'], errors='coerce')
        df['sales_ratio'] = pd.to_numeric(df['sales_ratio'], errors='coerce')

        # Extraer año y mes de la fecha
        df['sale_year'] = df['date_recorded'].dt.year
        df['sale_month'] = df['date_recorded'].dt.month_name()

        return df
    except Exception as e:
        st.error(f"Error al intentar cargar o preprocesar el archivo CSV: {e}")
        st.info("Asegúrate de que la URL sea correcta y que el archivo sea un CSV válido.")
        return pd.DataFrame() # Retorna un DataFrame vacío en caso de error

# --- Cargar los Datos ---
st.sidebar.header("Opciones de EDA")
st.sidebar.info("La aplicación se ejecuta de arriba a abajo. Recarga para ver cambios.")

df = load_data(CSV_URL)

if df.empty:
    st.warning("No se pudieron cargar los datos o el DataFrame está vacío. No se puede realizar el EDA.")
    st.stop() # Detiene la ejecución si no hay datos

# --- Resumen de los Datos ---
st.header("1. Resumen General del Dataset")
st.success("¡Datos cargados y preprocesados exitosamente!")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Número de Filas", value=df.shape[0])
with col2:
    st.metric(label="Número de Columnas", value=df.shape[1])
with col3:
    st.metric(label="Rango de Años de Venta", value=f"{int(df['sale_year'].min()) if not df['sale_year'].isnull().all() else 'N/A'} - {int(df['sale_year'].max()) if not df['sale_year'].isnull().all() else 'N/A'}")

st.subheader("Primeras 5 Filas del DataFrame")
st.dataframe(df.head())

st.subheader("Tipos de Datos por Columna")
st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={0: 'Tipo de Dato', 'index': 'Columna'}))

st.subheader("Estadísticas Descriptivas de Columnas Numéricas")
st.dataframe(df.describe().T)

# --- Manejo de Valores Faltantes ---
st.header("2. Manejo de Valores Faltantes (NaNs)")
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_info = pd.DataFrame({
    'Total Faltantes': missing_data,
    'Porcentaje (%)': missing_percentage[missing_data.index].round(2)
})

if not missing_info.empty:
    st.dataframe(missing_info)
    st.info("Columnas con alta proporción de valores faltantes como 'non_use_code', 'assessor_remarks', 'opm_remarks', 'location', y 'residential_type' para comerciales podrían requerir imputación cuidadosa o exclusión.")
else:
    st.info("¡No hay valores faltantes en el dataset!")

# --- Análisis por Columnas Categóricas y Numéricas ---

# Venta por Ciudad (Town)
st.header("3. Análisis de Columnas Categóricas")
st.subheader("Ventas por Ciudad (Top 20)")
top_towns = df['town'].value_counts().nlargest(20).reset_index()
top_towns.columns = ['Town', 'Número de Ventas']
fig_towns = px.bar(top_towns, x='Town', y='Número de Ventas',
                   title='Top 20 Ciudades por Número de Ventas',
                   labels={'Town': 'Ciudad', 'Número de Ventas': 'Recuento de Ventas'})
st.plotly_chart(fig_towns, use_container_width=True)

# Tipo de Propiedad
st.subheader("Ventas por Tipo de Propiedad")
prop_type_counts = df['property_type'].value_counts().reset_index()
prop_type_counts.columns = ['Property Type', 'Count']
fig_prop_type = px.pie(prop_type_counts, values='Count', names='Property Type',
                       title='Distribución por Tipo de Propiedad',
                       hole=0.3)
st.plotly_chart(fig_prop_type, use_container_width=True)

# Tipo Residencial (para Residential Property Type)
st.subheader("Ventas por Tipo Residencial")
# Filtrar solo 'Residential' para este análisis
residential_types = df[df['property_type'] == 'Residential']['residential_type'].value_counts().reset_index()
residential_types.columns = ['Residential Type', 'Count']
fig_res_type = px.bar(residential_types, x='Residential Type', y='Count',
                      title='Distribución por Tipo Residencial (Solo Residencial)',
                      labels={'Residential Type': 'Tipo Residencial', 'Count': 'Recuento'})
st.plotly_chart(fig_res_type, use_container_width=True)


# --- Análisis de Columnas Numéricas ---
st.header("4. Análisis de Columnas Numéricas")

# Histograma de Sale Amount
st.subheader("Distribución de 'Sale Amount' (Precio de Venta)")
fig_sale_amount = px.histogram(df, x='sale_amount', nbins=50,
                               title='Distribución de Precio de Venta',
                               labels={'sale_amount': 'Precio de Venta'},
                               log_y=True) # Escala logarítmica para ver la distribución
st.plotly_chart(fig_sale_amount, use_container_width=True)
st.info("La distribución de precios de venta suele estar sesgada hacia la derecha, por lo que una escala logarítmica ayuda a visualizar.")

# Histograma de Assessed Value
st.subheader("Distribución de 'Assessed Value' (Valor Fiscal)")
fig_assessed_value = px.histogram(df, x='assessed_value', nbins=50,
                                  title='Distribución de Valor Fiscal',
                                  labels={'assessed_value': 'Valor Fiscal'},
                                  log_y=True)
st.plotly_chart(fig_assessed_value, use_container_width=True)


# Relación entre Sale Amount y Assessed Value
st.subheader("Relación entre Precio de Venta y Valor Fiscal")
fig_scatter = px.scatter(df, x='assessed_value', y='sale_amount',
                         title='Precio de Venta vs. Valor Fiscal',
                         labels={'assessed_value': 'Valor Fiscal', 'sale_amount': 'Precio de Venta'},
                         hover_data=['town', 'address', 'property_type'],
                         log_x=True, log_y=True, opacity=0.5)
st.plotly_chart(fig_scatter, use_container_width=True)
st.info("Se espera una correlación positiva. Los puntos dispersos pueden indicar anomalías o tipos de propiedad específicos.")

# Sales Ratio
st.subheader("Distribución de 'Sales Ratio'")
fig_sales_ratio = px.histogram(df, x='sales_ratio', nbins=50,
                               title='Distribución de Sales Ratio',
                               labels={'sales_ratio': 'Sales Ratio'},
                               range_x=[0, 2]) # Limitar el rango para una mejor visualización de ratios comunes
st.plotly_chart(fig_sales_ratio, use_container_width=True)
st.info("Un 'Sales Ratio' cercano a 1 (o 100%) indica que el precio de venta es igual al valor fiscal. Valores muy bajos o muy altos pueden ser atípicos.")

# --- Análisis Temporal (por Año de Venta) ---
st.header("5. Análisis Temporal")
if 'sale_year' in df.columns:
    st.subheader("Ventas Anuales a lo Largo del Tiempo")
    sales_over_time = df.groupby('sale_year').size().reset_index(name='Número de Ventas')
    fig_time_series = px.line(sales_over_time, x='sale_year', y='Número de Ventas',
                              title='Número de Ventas por Año',
                              labels={'sale_year': 'Año de Venta', 'Número de Ventas': 'Número de Transacciones'})
    st.plotly_chart(fig_time_series, use_container_width=True)

    st.subheader("Precio Promedio de Venta por Año")
    avg_price_by_year = df.groupby('sale_year')['sale_amount'].mean().reset_index()
    fig_avg_price = px.line(avg_price_by_year, x='sale_year', y='sale_amount',
                            title='Precio Promedio de Venta por Año',
                            labels={'sale_year': 'Año de Venta', 'sale_amount': 'Precio Promedio de Venta'})
    st.plotly_chart(fig_avg_price, use_container_width=True)
else:
    st.warning("No se pudo realizar el análisis temporal sin la columna 'sale_year'.")


# --- Conclusión del EDA ---
st.header("6. Conclusiones y Próximos Pasos del EDA")
st.write("""
    Este EDA inicial nos ha permitido:
    * **Comprender la estructura** de los datos y los tipos de variables.
    * **Identificar valores faltantes** significativos en columnas como `non_use_code`, `assessor_remarks`, `opm_remarks`, `location`, y `residential_type` (para propiedades no residenciales). Estas columnas podrían requerir imputación o ser descartadas dependiendo del objetivo del análisis.
    * **Visualizar la distribución** de precios de venta y valores fiscales, notando su asimetría.
    * **Explorar la relación** entre valor fiscal y precio de venta a través de scatter plots.
    * **Analizar las ventas por ciudad, tipo de propiedad y tipo residencial.**
    * **Observar tendencias temporales** en el número de ventas y precios promedio a lo largo de los años.

    **Próximos pasos sugeridos:**
    1.  **Limpieza de Datos:** Decidir cómo manejar los valores faltantes.
    2.  **Ingeniería de Características:** Crear nuevas características a partir de las existentes (ej., antigüedad de la propiedad si se pudiera inferir).
    3.  **Análisis de Outliers:** Investigar los valores atípicos en `sale_amount`, `assessed_value`, y `sales_ratio`.
    4.  **Análisis Geográfico:** Si la columna `location` fuera usable (convertida a latitud/longitud), se podría hacer un análisis de mapas.
    5.  **Modelado:** Una vez que los datos estén limpios y preparados, se podría proceder a construir modelos predictivos (ej., predicción de precio de venta).
""")

st.markdown("---")
st.caption("Aplicación creada con Streamlit. Datos de JulianTorrest/prueba_real_state.")
