import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(
    page_title="EDA de Ventas Inmobiliarias",
    page_icon="🏠",
    layout="wide"
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

        # Renombrar columnas para facilitar el uso
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('_-', '_').str.replace('.', '', regex=False).str.lower()

        # Conversión de tipos de datos
        df['date_recorded'] = pd.to_datetime(df['date_recorded'], errors='coerce')
        df['assessed_value'] = pd.to_numeric(df['assessed_value'], errors='coerce')
        df['sale_amount'] = pd.to_numeric(df['sale_amount'], errors='coerce')
        df['sales_ratio'] = pd.to_numeric(df['sales_ratio'], errors='coerce')

        # Extraer año y mes de la fecha
        df['sale_year'] = df['date_recorded'].dt.year.astype('Int64') # Int64 para manejar NaNs en enteros
        df['sale_month'] = df['date_recorded'].dt.month_name()

        # Limpiar town, property_type, residential_type para filtros
        for col in ['town', 'property_type', 'residential_type']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace('nan', np.nan)

        return df
    except Exception as e:
        st.error(f"Error al intentar cargar o preprocesar el archivo CSV: {e}")
        st.info("Asegúrate de que la URL sea correcta y que el archivo sea un CSV válido.")
        return pd.DataFrame()

# --- Cargar los Datos ---
df_original = load_data(CSV_URL)

if df_original.empty:
    st.warning("No se pudieron cargar los datos o el DataFrame está vacío. No se puede realizar el EDA.")
    st.stop() # Detiene la ejecución si no hay datos

# --- Sidebar para Filtros ---
st.sidebar.header("Filtros de Datos")
st.sidebar.info("Usa los filtros a continuación para segmentar los datos del análisis y los gráficos.")

# Crear una copia del DataFrame para aplicar filtros
df_filtered = df_original.copy()

# 1. Filtro por Rango de Años de Venta (sale_year)
if 'sale_year' in df_filtered.columns and not df_filtered['sale_year'].isnull().all():
    min_year, max_year = int(df_filtered['sale_year'].min()), int(df_filtered['sale_year'].max())
    selected_years = st.sidebar.slider(
        "Rango de Años de Venta:",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    df_filtered = df_filtered[
        (df_filtered['sale_year'] >= selected_years[0]) &
        (df_filtered['sale_year'] <= selected_years[1])
    ]
else:
    st.sidebar.warning("Columna 'sale_year' no disponible o vacía para filtrar.")

# 2. Filtro por Mes de Venta (sale_month)
if 'sale_month' in df_filtered.columns and not df_filtered['sale_month'].isnull().all():
    all_months = df_filtered['sale_month'].dropna().unique().tolist()
    # Ordenar meses para mejor visualización
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    all_months_sorted = [m for m in month_order if m in all_months]

    selected_months = st.sidebar.multiselect(
        "Selecciona Meses de Venta:",
        options=all_months_sorted,
        default=all_months_sorted # Por defecto, todos seleccionados
    )
    if selected_months: # Solo filtrar si se seleccionó al menos un mes
        df_filtered = df_filtered[df_filtered['sale_month'].isin(selected_months)]
else:
    st.sidebar.warning("Columna 'sale_month' no disponible o vacía para filtrar.")

# 3. Filtro por Ciudad (town)
if 'town' in df_filtered.columns and not df_filtered['town'].isnull().all():
    all_towns = df_filtered['town'].dropna().unique().tolist()
    all_towns.sort() # Ordenar alfabéticamente
    selected_towns = st.sidebar.multiselect(
        "Selecciona Ciudades:",
        options=all_towns,
        default=all_towns[:min(len(all_towns), 10)] # Seleccionar solo las primeras 10 por defecto para evitar sobrecarga
    )
    if selected_towns:
        df_filtered = df_filtered[df_filtered['town'].isin(selected_towns)]
else:
    st.sidebar.warning("Columna 'town' no disponible o vacía para filtrar.")

# 4. Filtro por Tipo de Propiedad (property_type)
if 'property_type' in df_filtered.columns and not df_filtered['property_type'].isnull().all():
    all_property_types = df_filtered['property_type'].dropna().unique().tolist()
    all_property_types.sort()
    selected_property_types = st.sidebar.multiselect(
        "Selecciona Tipo de Propiedad:",
        options=all_property_types,
        default=all_property_types # Por defecto, todos seleccionados
    )
    if selected_property_types:
        df_filtered = df_filtered[df_filtered['property_type'].isin(selected_property_types)]
else:
    st.sidebar.warning("Columna 'property_type' no disponible o vacía para filtrar.")

# 5. Filtro por Tipo Residencial (residential_type)
# Este filtro solo es relevante si 'Residential' está seleccionado en 'property_type'
if 'residential_type' in df_filtered.columns and not df_filtered['residential_type'].isnull().all():
    all_residential_types = df_filtered['residential_type'].dropna().unique().tolist()
    all_residential_types.sort()
    selected_residential_types = st.sidebar.multiselect(
        "Selecciona Tipo Residencial:",
        options=all_residential_types,
        default=all_residential_types # Por defecto, todos seleccionados
    )
    if selected_residential_types:
        df_filtered = df_filtered[df_filtered['residential_type'].isin(selected_residential_types)]
else:
    st.sidebar.warning("Columna 'residential_type' no disponible o vacía para filtrar.")

st.sidebar.markdown("---")
st.sidebar.write(f"**Registros seleccionados:** {df_filtered.shape[0]} de {df_original.shape[0]}")


# --- Resumen de los Datos (Aplicando Filtros) ---
st.header("1. Resumen General del Dataset (Filtros Aplicados)")
if df_filtered.empty:
    st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Número de Filas (Filtradas)", value=df_filtered.shape[0])
    with col2:
        st.metric(label="Número de Columnas", value=df_filtered.shape[1])
    with col3:
        st.metric(label="Rango de Años de Venta (Filtrado)",
                  value=f"{int(df_filtered['sale_year'].min()) if not df_filtered['sale_year'].isnull().all() else 'N/A'} - {int(df_filtered['sale_year'].max()) if not df_filtered['sale_year'].isnull().all() else 'N/A'}")

    st.subheader("Primeras 5 Filas del DataFrame Filtrado")
    st.dataframe(df_filtered.head())

    st.subheader("Tipos de Datos por Columna")
    st.dataframe(df_filtered.dtypes.astype(str).reset_index().rename(columns={0: 'Tipo de Dato', 'index': 'Columna'}))

    st.subheader("Estadísticas Descriptivas de Columnas Numéricas")
    st.dataframe(df_filtered.describe().T)

    # --- Manejo de Valores Faltantes (Aplicando Filtros) ---
    st.header("2. Manejo de Valores Faltantes (NaNs) (Filtros Aplicados)")
    missing_data_filtered = df_filtered.isnull().sum()
    missing_data_filtered = missing_data_filtered[missing_data_filtered > 0].sort_values(ascending=False)
    missing_percentage_filtered = (df_filtered.isnull().sum() / len(df_filtered)) * 100
    missing_info_filtered = pd.DataFrame({
        'Total Faltantes': missing_data_filtered,
        'Porcentaje (%)': missing_percentage_filtered[missing_data_filtered.index].round(2)
    })

    if not missing_info_filtered.empty:
        st.dataframe(missing_info_filtered)
    else:
        st.info("¡No hay valores faltantes en el dataset filtrado!")

    # --- Análisis por Columnas Categóricas y Numéricas (Aplicando Filtros) ---
    st.header("3. Análisis de Columnas Categóricas (Filtros Aplicados)")

    # Venta por Ciudad (Town)
    st.subheader("Ventas por Ciudad (Top 20 Filtradas)")
    if 'town' in df_filtered.columns and not df_filtered['town'].isnull().all():
        top_towns_filtered = df_filtered['town'].value_counts().nlargest(20).reset_index()
        top_towns_filtered.columns = ['Town', 'Número de Ventas']
        fig_towns_filtered = px.bar(top_towns_filtered, x='Town', y='Número de Ventas',
                                   title='Top 20 Ciudades por Número de Ventas (Filtradas)',
                                   labels={'Town': 'Ciudad', 'Número de Ventas': 'Recuento de Ventas'})
        st.plotly_chart(fig_towns_filtered, use_container_width=True)
    else:
        st.info("Columna 'town' no disponible o insuficiente para este gráfico en los datos filtrados.")

    # Tipo de Propiedad
    st.subheader("Ventas por Tipo de Propiedad (Filtradas)")
    if 'property_type' in df_filtered.columns and not df_filtered['property_type'].isnull().all():
        prop_type_counts_filtered = df_filtered['property_type'].value_counts().reset_index()
        prop_type_counts_filtered.columns = ['Property Type', 'Count']
        fig_prop_type_filtered = px.pie(prop_type_counts_filtered, values='Count', names='Property Type',
                                       title='Distribución por Tipo de Propiedad (Filtradas)',
                                       hole=0.3)
        st.plotly_chart(fig_prop_type_filtered, use_container_width=True)
    else:
        st.info("Columna 'property_type' no disponible o insuficiente para este gráfico en los datos filtrados.")

    # Tipo Residencial (para Residential Property Type)
    st.subheader("Ventas por Tipo Residencial (Filtradas)")
    if 'residential_type' in df_filtered.columns and not df_filtered['residential_type'].isnull().all():
        residential_types_filtered = df_filtered[df_filtered['property_type'] == 'Residential']['residential_type'].value_counts().reset_index()
        residential_types_filtered.columns = ['Residential Type', 'Count']
        if not residential_types_filtered.empty:
            fig_res_type_filtered = px.bar(residential_types_filtered, x='Residential Type', y='Count',
                                          title='Distribución por Tipo Residencial (Filtradas y Solo Residencial)',
                                          labels={'Residential Type': 'Tipo Residencial', 'Count': 'Recuento'})
            st.plotly_chart(fig_res_type_filtered, use_container_width=True)
        else:
            st.info("No hay datos de tipo residencial para propiedades residenciales en los datos filtrados.")
    else:
        st.info("Columna 'residential_type' no disponible o insuficiente para este gráfico en los datos filtrados.")


    # --- Análisis de Columnas Numéricas (Aplicando Filtros) ---
    st.header("4. Análisis de Columnas Numéricas (Filtros Aplicados)")

    # Histograma de Sale Amount
    st.subheader("Distribución de 'Sale Amount' (Precio de Venta) (Filtrada)")
    if 'sale_amount' in df_filtered.columns and not df_filtered['sale_amount'].isnull().all():
        fig_sale_amount_filtered = px.histogram(df_filtered, x='sale_amount', nbins=50,
                                               title='Distribución de Precio de Venta (Filtrada)',
                                               labels={'sale_amount': 'Precio de Venta'},
                                               log_y=True)
        st.plotly_chart(fig_sale_amount_filtered, use_container_width=True)
    else:
        st.info("Columna 'sale_amount' no disponible o insuficiente para este gráfico en los datos filtrados.")

    # Histograma de Assessed Value
    st.subheader("Distribución de 'Assessed Value' (Valor Fiscal) (Filtrada)")
    if 'assessed_value' in df_filtered.columns and not df_filtered['assessed_value'].isnull().all():
        fig_assessed_value_filtered = px.histogram(df_filtered, x='assessed_value', nbins=50,
                                                  title='Distribución de Valor Fiscal (Filtrada)',
                                                  labels={'assessed_value': 'Valor Fiscal'},
                                                  log_y=True)
        st.plotly_chart(fig_assessed_value_filtered, use_container_width=True)
    else:
        st.info("Columna 'assessed_value' no disponible o insuficiente para este gráfico en los datos filtrados.")

    # Relación entre Sale Amount y Assessed Value
    st.subheader("Relación entre Precio de Venta y Valor Fiscal (Filtrada)")
    if 'assessed_value' in df_filtered.columns and 'sale_amount' in df_filtered.columns and \
       not df_filtered['assessed_value'].isnull().all() and not df_filtered['sale_amount'].isnull().all():
        fig_scatter_filtered = px.scatter(df_filtered, x='assessed_value', y='sale_amount',
                                         title='Precio de Venta vs. Valor Fiscal (Filtrada)',
                                         labels={'assessed_value': 'Valor Fiscal', 'sale_amount': 'Precio de Venta'},
                                         hover_data=['town', 'address', 'property_type'],
                                         log_x=True, log_y=True, opacity=0.5)
        st.plotly_chart(fig_scatter_filtered, use_container_width=True)
    else:
        st.info("Columnas 'assessed_value' o 'sale_amount' no disponibles o insuficientes para este gráfico en los datos filtrados.")

    # Sales Ratio
    st.subheader("Distribución de 'Sales Ratio' (Filtrada)")
    if 'sales_ratio' in df_filtered.columns and not df_filtered['sales_ratio'].isnull().all():
        fig_sales_ratio_filtered = px.histogram(df_filtered, x='sales_ratio', nbins=50,
                                               title='Distribución de Sales Ratio (Filtrada)',
                                               labels={'sales_ratio': 'Sales Ratio'},
                                               range_x=[0, 2])
        st.plotly_chart(fig_sales_ratio_filtered, use_container_width=True)
    else:
        st.info("Columna 'sales_ratio' no disponible o insuficiente para este gráfico en los datos filtrados.")

    # --- Análisis Temporal (por Año de Venta) (Aplicando Filtros) ---
    st.header("5. Análisis Temporal (Filtros Aplicados)")
    if 'sale_year' in df_filtered.columns and not df_filtered['sale_year'].isnull().all():
        st.subheader("Ventas Anuales a lo Largo del Tiempo (Filtradas)")
        sales_over_time_filtered = df_filtered.groupby('sale_year').size().reset_index(name='Número de Ventas')
        fig_time_series_filtered = px.line(sales_over_time_filtered, x='sale_year', y='Número de Ventas',
                                          title='Número de Ventas por Año (Filtradas)',
                                          labels={'sale_year': 'Año de Venta', 'Número de Ventas': 'Número de Transacciones'})
        st.plotly_chart(fig_time_series_filtered, use_container_width=True)

        st.subheader("Precio Promedio de Venta por Año (Filtrado)")
        avg_price_by_year_filtered = df_filtered.groupby('sale_year')['sale_amount'].mean().reset_index()
        fig_avg_price_filtered = px.line(avg_price_by_year_filtered, x='sale_year', y='sale_amount',
                                        title='Precio Promedio de Venta por Año (Filtrado)',
                                        labels={'sale_year': 'Año de Venta', 'sale_amount': 'Precio Promedio de Venta'})
        st.plotly_chart(fig_avg_price_filtered, use_container_width=True)
    else:
        st.warning("No se pudo realizar el análisis temporal sin la columna 'sale_year' en los datos filtrados.")

# --- Conclusión del EDA ---
st.header("6. Conclusiones y Próximos Pasos del EDA")

# The 'else' at line 301 is likely related to this 'if' block below
# or one of the previous numerical/temporal analysis blocks.
# Ensure the 'if' that this 'else' corresponds to is correctly structured.
if not df_filtered.empty: # This 'if' block started on line ~200, so this 'else' should correspond to it.
    st.write("""
        Este EDA interactivo permite explorar los datos de ventas inmobiliarias aplicando diversos filtros.
        Las visualizaciones y estadísticas se actualizan automáticamente para reflejar los datos seleccionados.

        **Puntos clave:**
        * **Interactividad:** Los deslizadores y selectores permiten una exploración profunda de subconjuntos de datos.
        * **Robustez:** Se han incluido comprobaciones para columnas ausentes o vacías, y se manejan los `NaNs` para evitar errores.
        * **Preprocesamiento:** Las columnas han sido renombradas y convertidas a tipos de datos adecuados en la función `load_data`.

        **Próximos pasos sugeridos:**
        1.  **Limpieza Avanzada:** Decidir estrategias específicas para imputar o manejar valores faltantes en las columnas clave.
        2.  **Ingeniería de Características:** Por ejemplo, combinar 'town' y 'property_type' o crear categorías de precios.
        3.  **Detección de Outliers:** Utilizar métodos estadísticos para identificar y manejar transacciones atípicas.
        4.  **Análisis Bivariado/Multivariado:** Explorar las relaciones entre más de dos variables a la vez.
        5.  **Modelado Predictivo:** Utilizar estos datos para construir modelos que predigan precios de venta.
    """)

# The 'else' that was throwing the error was likely here,
# and it belongs to the 'if not df_filtered.empty:' block that starts much earlier.
else: # This 'else' correctly belongs to the 'if not df_filtered.empty:' block.
    st.error("El DataFrame filtrado está vacío. No se puede realizar el análisis de datos.")


st.markdown("---")
st.caption("Aplicación creada con Streamlit. Datos de JulianTorrest/prueba_real_state.")
