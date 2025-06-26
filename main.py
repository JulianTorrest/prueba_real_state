print("DEBUG: Script started.") # Added for very early debugging

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- Configuración de la aplicación Streamlit ---
st.set_page_config(
    page_title="Análisis Completo de Ventas Inmobiliarias",
    page_icon="🏠",
    layout="wide" # Usa todo el ancho de la pantalla
)

st.title("🏡 Análisis Completo de Datos de Ventas Inmobiliarias (2001-2022)")
st.markdown("""
    Esta aplicación interactiva realiza un **Análisis Exploratorio de Datos (EDA)**,
    demuestra **limpieza y ingeniería de características**, **detección de outliers**
    y un **esbozo de modelado predictivo** sobre el conjunto de datos de ventas inmobiliarias de Connecticut.
    Los datos se cargan directamente desde GitHub.
""")
st.markdown("---")

# --- Define la URL de tu archivo CSV en GitHub ---
CSV_URL = "https://media.githubusercontent.com/media/JulianTorrest/prueba_real_state/main/data/Real_Estate_Sales_2001-2022_GL.csv"

@st.cache_data(show_spinner="Cargando y preprocesando datos...")
def load_and_preprocess_data(url):
    """
    Carga el archivo CSV y realiza un preprocesamiento inicial.
    Se han especificado los tipos de datos para evitar la advertencia de 'DtypeWarning'
    y asegurar la correcta interpretación de las columnas.
    """
    print("DEBUG: [load_and_preprocess_data] Iniciando carga y preprocesamiento de datos...")
    df = pd.DataFrame() # Initialize df to ensure it exists even if read_csv fails
    try:
        print("DEBUG: [load_and_preprocess_data] Intentando leer CSV...")
        # Especificar los tipos de datos directamente para las columnas problemáticas
        # Las columnas se refieren a los nombres ORIGINALES del CSV.
        df = pd.read_csv(url, dtype={
            'Sales Ratio': float,
            'Property Type': str,
            'Residential Type': str,
            'Non Use Code': str,
            'Assessor Remarks': str,
            'OPM remarks': str,
            'Location': str
        }, low_memory=False)
        print(f"DEBUG: [load_and_preprocess_data] CSV cargado exitosamente. Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

        print("DEBUG: [load_and_preprocess_data] Renombrando columnas...")
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('_-', '_').str.replace('.', '', regex=False).str.lower()
        print("DEBUG: [load_and_preprocess_data] Columnas renombradas a minúsculas y con guiones bajos.")
        
        print("DEBUG: [load_and_preprocess_data] Convirtiendo columnas numéricas...")
        df['assessed_value'] = pd.to_numeric(df['assessed_value'], errors='coerce')
        df['sale_amount'] = pd.to_numeric(df['sale_amount'], errors='coerce')
        df['sales_ratio'] = pd.to_numeric(df['sales_ratio'], errors='coerce')
        print("DEBUG: [load_and_preprocess_data] Columnas numéricas convertidas.")

        print("DEBUG: [load_and_preprocess_data] Procesando 'date_recorded'...")
        df['date_recorded'] = pd.to_datetime(df['date_recorded'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df['date_recorded']):
            df['date_recorded'] = df['date_recorded'].dt.tz_localize(None)
        print("DEBUG: [load_and_preprocess_data] 'date_recorded' procesada.")

        print("DEBUG: [load_and_preprocess_data] Extrayendo 'sale_year' y 'sale_month'...")
        df['sale_year'] = df['date_recorded'].dt.year.astype('Int64')
        df['sale_month'] = df['date_recorded'].dt.month_name()
        print("DEBUG: [load_and_preprocess_data] 'sale_year' y 'sale_month' extraídas.")
        
        print("DEBUG: [load_and_preprocess_data] Limpiando 'sale_month' y formato 'date_recorded'...")
        df['sale_month'] = df['sale_month'].astype(str).replace('NaT', np.nan).fillna('Unknown')
        df['date_recorded'] = df['date_recorded'].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
        print("DEBUG: [load_and_preprocess_data] 'sale_month' y 'date_recorded' limpiadas.")

        print("DEBUG: [load_and_preprocess_data] Limpiando columnas categóricas...")
        for col in ['town', 'property_type', 'residential_type', 'non_use_code', 'assessor_remarks', 'opm_remarks', 'location']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace('nan', np.nan)
                df[col] = df[col].fillna('Unknown')
                print(f"DEBUG: [load_and_preprocess_data] Columna categórica '{col}' limpiada y NaNs rellenados.")

        print("DEBUG: [load_and_preprocess_data] Preprocesamiento de datos completado.")
        return df
    except Exception as e:
        print(f"ERROR: [load_and_preprocess_data] Fallo en el preprocesamiento de datos: {e}")
        st.error(f"Error al intentar cargar o preprocesar el archivo CSV: {e}")
        st.info("Asegúrate de que la URL sea correcta y que el archivo sea un CSV válido.")
        st.exception(e)
        return pd.DataFrame()

# --- Cargar los Datos Originales ---
print("DEBUG: Llamando a load_and_preprocess_data...")
df_original = load_and_preprocess_data(CSV_URL)
print(f"DEBUG: df_original cargado. Vacío: {df_original.empty}")

if df_original.empty:
    st.warning("No se pudieron cargar los datos o el DataFrame está vacío. No se puede continuar con el análisis.")
    st.stop()

print("DEBUG: Iniciando sección de filtros de la barra lateral...")
# --- Sidebar para Filtros (aplicados a todas las pestañas) ---
st.sidebar.header("Filtros de Datos")
st.sidebar.info("Usa los filtros a continuación para segmentar los datos en todas las secciones de la aplicación.")

df_filtered = df_original.copy()

# 1. Filtro por Rango de Años de Venta (sale_year)
try:
    if 'sale_year' in df_filtered.columns and not df_filtered['sale_year'].isnull().all():
        min_year = int(df_filtered['sale_year'].min()) if not df_filtered['sale_year'].isnull().all() else 2001
        max_year = int(df_filtered['sale_year'].max()) if not df_filtered['sale_year'].isnull().all() else 2022
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
    print("DEBUG: Filtro 'Rango de Años de Venta' procesado.")
except Exception as e:
    st.sidebar.error(f"Error en el filtro 'Rango de Años de Venta': {e}")
    st.sidebar.exception(e)
    print(f"ERROR: Filtro 'Rango de Años de Venta' falló: {e}")


# 2. Filtro por Mes de Venta (sale_month)
try:
    if 'sale_month' in df_filtered.columns and not df_filtered['sale_month'].isnull().all():
        all_months = df_filtered['sale_month'].dropna().unique().tolist()
        month_order = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December", "Unknown"]
        all_months_sorted = [m for m in month_order if m in all_months]

        selected_months = st.sidebar.multiselect(
            "Selecciona Meses de Venta:",
            options=all_months_sorted,
            default=all_months_sorted
        )
        if selected_months:
            df_filtered = df_filtered[df_filtered['sale_month'].isin(selected_months)]
    else:
        st.sidebar.warning("Columna 'sale_month' no disponible o vacía para filtrar.")
    print("DEBUG: Filtro 'Mes de Venta' procesado.")
except Exception as e:
    st.sidebar.error(f"Error en el filtro 'Mes de Venta': {e}")
    st.sidebar.exception(e)
    print(f"ERROR: Filtro 'Mes de Venta' falló: {e}")

# 3. Filtro por Ciudad (town)
try:
    if 'town' in df_filtered.columns and not df_filtered['town'].isnull().all():
        all_towns = df_filtered['town'].dropna().unique().tolist()
        all_towns.sort()
        selected_towns = st.sidebar.multiselect(
            "Selecciona Ciudades:",
            options=all_towns,
            default=[]
        )
        if selected_towns:
            df_filtered = df_filtered[df_filtered['town'].isin(selected_towns)]
    else:
        st.sidebar.warning("Columna 'town' no disponible o vacía para filtrar.")
    print("DEBUG: Filtro 'Ciudad' procesado.")
except Exception as e:
    st.sidebar.error(f"Error en el filtro 'Ciudad': {e}")
    st.sidebar.exception(e)
    print(f"ERROR: Filtro 'Ciudad' falló: {e}")

# 4. Filtro por Tipo de Propiedad (property_type)
try:
    if 'property_type' in df_filtered.columns and not df_filtered['property_type'].isnull().all():
        all_property_types = df_filtered['property_type'].dropna().unique().tolist()
        all_property_types.sort()
        selected_property_types = st.sidebar.multiselect(
            "Selecciona Tipo de Propiedad:",
            options=all_property_types,
            default=all_property_types
        )
        if selected_property_types:
            df_filtered = df_filtered[df_filtered['property_type'].isin(selected_property_types)]
    else:
        st.sidebar.warning("Columna 'property_type' no disponible o vacía para filtrar.")
    print("DEBUG: Filtro 'Tipo de Propiedad' procesado.")
except Exception as e:
    st.sidebar.error(f"Error en el filtro 'Tipo de Propiedad': {e}")
    st.sidebar.exception(e)
    print(f"ERROR: Filtro 'Tipo de Propiedad' falló: {e}")

# 5. Filtro por Tipo Residencial (residential_type)
try:
    if 'residential_type' in df_filtered.columns and not df_filtered['residential_type'].isnull().all():
        if 'Residential' in selected_property_types and any(df_filtered['property_type'] == 'Residential'):
            all_residential_types = df_filtered['residential_type'].dropna().unique().tolist()
            if 'Unknown' in all_residential_types and len(all_residential_types) > 1:
                all_residential_types.remove('Unknown')
            all_residential_types.sort()
            selected_residential_types = st.sidebar.multiselect(
                "Selecciona Tipo Residencial:",
                options=all_residential_types,
                default=all_residential_types
            )
            if selected_residential_types:
                df_filtered = df_filtered[df_filtered['residential_type'].isin(selected_residential_types)]
        else:
            st.sidebar.info("Filtro de Tipo Residencial deshabilitado (selecciona 'Residential' en Tipo de Propiedad).")
    else:
        st.sidebar.warning("Columna 'residential_type' no disponible o vacía para filtrar.")
    print("DEBUG: Filtro 'Tipo Residencial' procesado.")
except Exception as e:
    st.sidebar.error(f"Error en el filtro 'Tipo Residencial': {e}")
    st.sidebar.exception(e)
    print(f"ERROR: Filtro 'Tipo Residencial' falló: {e}")

st.sidebar.markdown("---")
st.sidebar.write(f"**Registros seleccionados:** {df_filtered.shape[0]} de {df_original.shape[0]}")
print("DEBUG: Barra lateral de filtros completada.")


# --- Pestañas de la Aplicación ---
print("DEBUG: Creando pestañas de la aplicación...")
tab_eda, tab_clean, tab_feat_eng, tab_outliers, tab_bivariate, tab_modeling = st.tabs([
    "📊 EDA General",
    "🧹 Limpieza Avanzada",
    "💡 Ingeniería de Características",
    "🕵️ Detección de Outliers",
    "📈 Análisis Bivariado",
    "🤖 Modelado Predictivo"
])
print("DEBUG: Pestañas creadas.")

# --- Contenido de la Pestaña: EDA General ---
with tab_eda:
    try:
        print("DEBUG: Iniciando pestaña 'EDA General'...")
        st.header("1. Resumen General del Dataset (Filtros Aplicados)")
        if df_filtered.empty:
            st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
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
            st.dataframe(df_filtered.select_dtypes(include=np.number).describe().T)

            st.header("2. Manejo de Valores Faltantes (NaNs) (Filtros Aplicados)")
            missing_data_filtered = df_filtered.isnull().sum()
            missing_data_filtered = missing_data_filtered[missing_data_filtered > 0].sort_values(ascending=False) # Corrected variable name
            missing_percentage_filtered = (df_filtered.isnull().sum() / len(df_filtered)) * 100
            missing_info_filtered = pd.DataFrame({
                'Total Faltantes': missing_data_filtered,
                'Porcentaje (%)': missing_percentage_filtered[missing_data_filtered.index].round(2)
            })

            if not missing_info_filtered.empty:
                st.dataframe(missing_info_filtered)
                st.info("Columnas con alta proporción de valores faltantes podrían requerir imputación cuidadosa o exclusión. Explora la pestaña 'Limpieza Avanzada'.")
            else:
                st.info("¡No hay valores faltantes en el dataset filtrado! Excelente.")

            st.header("3. Visualizaciones Clave (Filtros Aplicados)")

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

            st.subheader("Distribución de 'Sale Amount' (Precio de Venta) (Filtrada)")
            if 'sale_amount' in df_filtered.columns and not df_filtered['sale_amount'].isnull().all():
                fig_sale_amount_filtered = px.histogram(df_filtered, x='sale_amount', nbins=50,
                                                        title='Distribución de Precio de Venta (Filtrada)',
                                                        labels={'sale_amount': 'Precio de Venta'},
                                                        log_y=True)
                st.plotly_chart(fig_sale_amount_filtered, use_container_width=True)
            else:
                st.info("Columna 'sale_amount' no disponible o insuficiente para este gráfico en los datos filtrados.")

            st.subheader("Ventas Anuales a lo Largo del Tiempo (Filtradas)")
            if 'sale_year' in df_filtered.columns and not df_filtered['sale_year'].isnull().all():
                sales_over_time_filtered = df_filtered.groupby('sale_year').size().reset_index(name='Número de Ventas')
                fig_time_series_filtered = px.line(sales_over_time_filtered, x='sale_year', y='Número de Ventas',
                                                  title='Número de Ventas por Año (Filtradas)',
                                                  labels={'sale_year': 'Año de Venta', 'Número de Ventas': 'Número de Transacciones'})
                st.plotly_chart(fig_time_series_filtered, use_container_width=True)
            else:
                st.warning("No se pudo realizar el análisis temporal sin la columna 'sale_year' en los datos filtrados.")
        print("DEBUG: Pestaña 'EDA General' completada.")
    except Exception as e:
        st.error(f"Error en la pestaña 'EDA General': {e}")
        st.exception(e)
        print(f"ERROR: Pestaña 'EDA General' falló: {e}")


# --- Contenido de la Pestaña: Limpieza Avanzada ---
with tab_clean:
    try:
        print("DEBUG: Iniciando pestaña 'Limpieza Avanzada'...")
        st.header("Limpieza Avanzada de Datos")
        st.markdown("""
            Esta sección muestra estrategias para manejar **valores faltantes (NaNs)**.
            Las operaciones aquí son **demostrativas**; no alteran permanentemente el DataFrame `df_filtered` para las otras pestañas,
            a menos que se indique explícitamente y se haga una copia modificada.
        """)

        if df_filtered.empty:
            st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
        else:
            st.subheader("Columnas con Valores Faltantes en el DataFrame Actual")
            missing_info_current = df_filtered.isnull().sum()
            missing_info_current = missing_info_current[missing_info_current > 0].sort_values(ascending=False)
            missing_percentage_current = (df_filtered.isnull().sum() / len(df_filtered)) * 100
            current_missing_df = pd.DataFrame({
                'Total Faltantes': missing_info_current,
                'Porcentaje (%)': missing_percentage_current[missing_info_current.index].round(2)
            })
            if not missing_info_current.empty:
                st.dataframe(current_missing_df)
            else:
                st.info("¡No hay valores faltantes en el DataFrame actual (filtrado)! Excelente.")

            st.markdown("---")
            st.subheader("Estrategias de Imputación/Eliminación (Demostración)")

            df_cleaned_temp = df_filtered.copy()

            st.write("#### 1. Columnas con alta proporción de NaNs (Ejemplo: `non_use_code`, `assessor_remarks`, `opm_remarks`, `location`)")
            cols_to_drop = ['non_use_code', 'assessor_remarks', 'opm_remarks', 'location']
            cols_to_drop_existing = [col for col in cols_to_drop if col in df_cleaned_temp.columns]
            if cols_to_drop_existing:
                st.write(f"Columnas candidatas a eliminación por muchos NaNs o irrelevancia: `{', '.join(cols_to_drop_existing)}`")
                if st.checkbox("Eliminar estas columnas para la demostración?", key="clean_drop_cols"):
                    df_cleaned_temp = df_cleaned_temp.drop(columns=cols_to_drop_existing)
                    st.success(f"Columnas eliminadas. DataFrame ahora tiene {df_cleaned_temp.shape[1]} columnas.")
            else:
                st.info("Las columnas de ejemplo para eliminación no están presentes o ya fueron manejadas.")

            st.write("#### 2. Imputación para `residential_type`")
            if 'residential_type' in df_cleaned_temp.columns:
                st.write(f"Valores únicos antes de imputación: {df_cleaned_temp['residential_type'].unique()}")
                if df_cleaned_temp['residential_type'].isnull().any():
                    imputation_option = st.radio(
                        "Cómo imputar valores faltantes en 'residential_type'?",
                        ("No imputar", "Imputar con 'Unknown'", "Imputar con la Moda"),
                        key="impute_residential_type"
                    )
                    if imputation_option == "Imputar con 'Unknown'":
                        df_cleaned_temp['residential_type'] = df_cleaned_temp['residential_type'].fillna('Unknown')
                        st.success("Valores faltantes en 'residential_type' imputados con 'Unknown'.")
                    elif imputation_option == "Imputar con la Moda":
                        mode_val = df_cleaned_temp['residential_type'].mode()[0]
                        df_cleaned_temp['residential_type'] = df_cleaned_temp['residential_type'].fillna(mode_val)
                        st.success(f"Valores faltantes en 'residential_type' imputados con la moda: '{mode_val}'.")
                    st.write(f"Valores únicos después de imputación: {df_cleaned_temp['residential_type'].unique()}")
                else:
                    st.info("No hay valores faltantes en 'residential_type' en el DataFrame actual.")
            else:
                st.warning("Columna 'residential_type' no disponible.")

            st.write("#### 3. Imputación para Columnas Numéricas (ej. `sale_amount`, `assessed_value`)")
            numeric_cols_with_nan = df_cleaned_temp.select_dtypes(include=np.number).columns[df_cleaned_temp.select_dtypes(include=np.number).isnull().any()].tolist()
            if numeric_cols_with_nan:
                st.write(f"Columnas numéricas con NaNs: {numeric_cols_with_nan}")
                imputation_method = st.radio(
                    "Método de imputación para numéricas:",
                    ("No imputar", "Mediana", "Media"),
                    key="impute_numeric_method"
                )
                if imputation_method != "No imputar":
                    for col in numeric_cols_nan:
                        if imputation_method == "Mediana":
                            val = df_cleaned_temp[col].median()
                        else:
                            val = df_cleaned_temp[col].mean()
                        df_cleaned_temp[col] = df_cleaned_temp[col].fillna(val)
                        st.success(f"Valores faltantes en '{col}' imputados con la {imputation_method.lower()}: {val:,.2f}.")
            else:
                st.info("No hay columnas numéricas con valores faltantes en el DataFrame actual.")

            st.subheader("Estado del DataFrame después de la Limpieza Demostrativa:")
            st.dataframe(df_cleaned_temp.isnull().sum()[df_cleaned_temp.isnull().sum() > 0].sort_values(ascending=False))
            if df_cleaned_temp.isnull().sum().sum() == 0:
                st.success("¡DataFrame temporal sin valores faltantes!")
        print("DEBUG: Pestaña 'Limpieza Avanzada' completada.")
    except Exception as e:
        st.error(f"Error en la pestaña 'Limpieza Avanzada': {e}")
        st.exception(e)
        print(f"ERROR: Pestaña 'Limpieza Avanzada' falló: {e}")

# --- Contenido de la Pestaña: Ingeniería de Características ---
with tab_feat_eng:
    try:
        print("DEBUG: Iniciando pestaña 'Ingeniería de Características'...")
        st.header("Ingeniería de Características")
        st.markdown("""
            En esta sección, creamos nuevas características a partir de las existentes para potenciar el análisis y el modelado.
            Las características generadas aquí se utilizan en las secciones posteriores.
        """)

        if df_filtered.empty:
            st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
        else:
            df_fe = df_filtered.copy()

            st.subheader("1. Característica Combinada: Ciudad y Tipo de Propiedad")
            if 'town' in df_fe.columns and 'property_type' in df_fe.columns:
                df_fe['town_prop_type'] = df_fe['town'].astype(str) + '_' + df_fe['property_type'].astype(str)
                st.write("Se ha creado la característica `town_prop_type`.")
                st.dataframe(df_fe[['town', 'property_type', 'town_prop_type']].head())
                st.info(f"Número de categorías únicas para `town_prop_type`: {df_fe['town_prop_type'].nunique()}")
            else:
                st.warning("Columnas 'town' o 'property_type' no disponibles para crear `town_prop_type`.")

            st.subheader("2. Categorías de Precio de Venta")
            if 'sale_amount' in df_fe.columns and not df_fe['sale_amount'].isnull().all():
                bins = [0, 100000, 300000, 700000, np.inf]
                labels = ['0-100K', '100K-300K', '300K-700K', '700K+']
                df_fe['sale_amount_category'] = pd.cut(df_fe['sale_amount'], bins=bins, labels=labels, right=False)
                st.write("Se ha creado la característica `sale_amount_category`.")
                st.dataframe(df_fe[['sale_amount', 'sale_amount_category']].head())

                st.subheader("Distribución de `sale_amount_category`")
                price_category_counts = df_fe['sale_amount_category'].value_counts().reset_index()
                price_category_counts.columns = ['Sale_Amount_Category', 'Count']

                fig_price_cat = px.bar(price_category_counts,
                                       x='Sale_Amount_Category', y='Count',
                                       title='Distribución por Categoría de Precio de Venta')

                st.plotly_chart(fig_price_cat, use_container_width=True)
            else:
                st.warning("Columna 'sale_amount' no disponible para crear `sale_amount_category`.")

            st.subheader("3. Indicadores Binarios (Ej. `is_commercial`)")
            if 'property_type' in df_fe.columns:
                df_fe['is_commercial'] = (df_fe['property_type'] == 'Commercial').astype(int)
                st.write("Se ha creado la característica `is_commercial` (1 si es comercial, 0 en otro caso).")
                st.dataframe(df_fe[['property_type', 'is_commercial']].head())
            else:
                st.warning("Columna 'property_type' no disponible para crear `is_commercial`.")

            st.subheader("DataFrame después de la Ingeniería de Características (Primeras Filas)")
            st.dataframe(df_fe.head())
            st.info("Este DataFrame `df_fe` se pasaría al paso de Detección de Outliers y Modelado.")
        print("DEBUG: Pestaña 'Ingeniería de Características' completada.")
    except Exception as e:
        st.error(f"Error en la pestaña 'Ingeniería de Características': {e}")
        st.exception(e)
        print(f"ERROR: Pestaña 'Ingeniería de Características' falló: {e}")

# --- Contenido de la Pestaña: Detección de Outliers ---
with tab_outliers:
    try:
        print("DEBUG: Iniciando pestaña 'Detección de Outliers'...")
        st.header("Detección de Outliers")
        st.markdown("""
            Identificamos valores atípicos en las columnas numéricas clave, que podrían influir en el modelado.
            Aquí usamos el **método del Rango Intercuartílico (IQR)**.
        """)

        if df_filtered.empty:
            st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
        else:
            df_outlier = df_filtered.copy()

            st.subheader("1. Detección de Outliers en 'Sale Amount' (Precio de Venta)")
            if 'sale_amount' in df_outlier.columns and not df_outlier['sale_amount'].isnull().all():
                Q1 = df_outlier['sale_amount'].quantile(0.25)
                Q3 = df_outlier['sale_amount'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_sale_amount = df_outlier[(df_outlier['sale_amount'] < lower_bound) | (df_outlier['sale_amount'] > upper_bound)]
                st.write(f"Número de outliers detectados en 'sale_amount' (usando IQR): **{outliers_sale_amount.shape[0]}**")
                st.write(f"Rango normal (IQR): ${lower_bound:,.2f} - ${upper_bound:,.2f}")

                if not outliers_sale_amount.empty:
                    st.subheader("Ejemplos de Outliers de 'Sale Amount'")
                    st.dataframe(outliers_sale_amount[['town', 'address', 'sale_amount', 'list_year']].head())

                    fig_boxplot_sale = px.box(df_outlier, y='sale_amount',
                                              title='Box Plot de Precio de Venta con Outliers',
                                              labels={'sale_amount': 'Precio de Venta'},
                                              points="outliers")
                    st.plotly_chart(fig_boxplot_sale, use_container_width=True)

                    if st.checkbox("¿Remover outliers de 'Sale Amount' para el análisis subsiguiente?", key="remove_sale_outliers"):
                        df_outlier = df_outlier[(df_outlier['sale_amount'] >= lower_bound) & (df_outlier['sale_amount'] <= upper_bound)]
                        st.success(f"Outliers de 'sale_amount' removidos. Nuevas filas: {df_outlier.shape[0]}. **(Solo para esta pestaña)**")
                else:
                    st.info("No se detectaron outliers significativos en 'sale_amount' con el método IQR en el DataFrame actual.")
            else:
                st.warning("Columna 'sale_amount' no disponible para detección de outliers.")

            st.subheader("2. Detección de Outliers en 'Assessed Value' (Valor Fiscal)")
            if 'assessed_value' in df_outlier.columns and not df_outlier['assessed_value'].isnull().all():
                Q1_assessed = df_outlier['assessed_value'].quantile(0.25)
                Q3_assessed = df_outlier['assessed_value'].quantile(0.75)
                IQR_assessed = Q3_assessed - Q1_assessed
                lower_bound_assessed = Q1_assessed - 1.5 * IQR_assessed
                upper_bound_assessed = Q3_assessed + 1.5 * IQR_assessed

                outliers_assessed = df_outlier[(df_outlier['assessed_value'] < lower_bound_assessed) | (df_outlier['assessed_value'] > upper_bound_assessed)]
                st.write(f"Número de outliers detectados en 'assessed_value' (usando IQR): **{outliers_assessed.shape[0]}**")
                st.write(f"Rango normal (IQR): ${lower_bound_assessed:,.2f} - ${upper_bound_assessed:,.2f}")

                if not outliers_assessed.empty:
                    st.subheader("Ejemplos de Outliers de 'Assessed Value'")
                    st.dataframe(outliers_assessed[['town', 'address', 'assessed_value', 'list_year']].head())

                    fig_boxplot_assessed = px.box(df_outlier, y='assessed_value',
                                              title='Box Plot de Valor Fiscal con Outliers',
                                              labels={'assessed_value': 'Valor Fiscal'},
                                              points="outliers")
                    st.plotly_chart(fig_boxplot_assessed, use_container_width=True)
                else:
                    st.info("No se detectaron outliers significativos en 'assessed_value' con el método IQR en el DataFrame actual.")
            else:
                st.warning("Columna 'assessed_value' no disponible para detección de outliers.")

            st.info("La eliminación de outliers debe hacerse con precaución, ya que puede eliminar información valiosa. A menudo es mejor probar modelos con y sin ellos.")
        print("DEBUG: Pestaña 'Detección de Outliers' completada.")
    except Exception as e:
        st.error(f"Error en la pestaña 'Detección de Outliers': {e}")
        st.exception(e)
        print(f"ERROR: Pestaña 'Detección de Outliers' falló: {e}")

# --- Contenido de la Pestaña: Análisis Bivariado/Multivariado ---
with tab_bivariate:
    try:
        print("DEBUG: Iniciando pestaña 'Análisis Bivariado y Multivariado'...")
        st.header("Análisis Bivariado y Multivariado")
        st.markdown("""
            Explora las relaciones entre dos o más variables usando gráficos avanzados.
            Los gráficos se actualizan según los filtros aplicados en la barra lateral.
        """)

        if df_filtered.empty:
            st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
        else:
            st.subheader("1. Precio de Venta vs. Valor Fiscal por Tipo de Propiedad")
            if 'assessed_value' in df_filtered.columns and 'sale_amount' in df_filtered.columns and 'property_type' in df_filtered.columns:
                fig_scatter_bivar = px.scatter(df_filtered, x='assessed_value', y='sale_amount', color='property_type',
                                               title='Precio de Venta vs. Valor Fiscal por Tipo de Propiedad',
                                               labels={'assessed_value': 'Valor Fiscal', 'sale_amount': 'Precio de Venta'},
                                               hover_data=['town', 'address'],
                                               log_x=True, log_y=True, opacity=0.7)
                st.plotly_chart(fig_scatter_bivar, use_container_width=True)
                st.info("Se observa la correlación positiva. El color muestra cómo se agrupan los tipos de propiedad.")
            else:
                st.warning("Columnas 'assessed_value', 'sale_amount' o 'property_type' no disponibles para este gráfico.")

            st.subheader("2. Distribución de Precio de Venta por Mes")
            if 'sale_month' in df_filtered.columns and not df_filtered['sale_month'].isnull().all():
                month_order = ["January", "February", "March", "April", "May", "June",
                                "July", "August", "September", "October", "November", "December", "Unknown"]
                df_plot = df_filtered.copy()
                current_months = df_plot['sale_month'].dropna().unique().tolist()
                final_month_order = [m for m in month_order if m in current_months]

                df_plot['sale_month'] = pd.Categorical(df_plot['sale_month'], categories=final_month_order, ordered=True)
                df_plot.sort_values('sale_month', inplace=True)

                fig_box_month = px.box(df_plot, x='sale_month', y='sale_amount',
                                       title='Precio de Venta por Mes',
                                       labels={'sale_month': 'Mes de Venta', 'sale_amount': 'Precio de Venta'},
                                       points="outliers", log_y=True)
                st.plotly_chart(fig_box_month, use_container_width=True)
                st.info("Compara la distribución de precios entre diferentes meses.")
            else:
                st.warning("Columna 'sale_month' o 'sale_amount' no disponibles para este gráfico.")

            st.subheader("3. Mapa de Calor de Correlación Numérica")
            numeric_cols_bivar = df_filtered.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols_bivar) >= 2:
                corr_matrix = df_filtered[numeric_cols_bivar].corr()
                fig_corr_heatmap = px.imshow(corr_matrix,
                                              text_auto=True,
                                              aspect="auto",
                                              color_continuous_scale='RdBu_r',
                                              title='Matriz de Correlación Numérica (Filtrada)')
                st.plotly_chart(fig_corr_heatmap, use_container_width=True)
                st.info("Muestra la fuerza y dirección de la relación lineal entre pares de variables numéricas.")
            else:
                st.info("No hay suficientes columnas numéricas para generar un mapa de calor de correlación en los datos filtrados.")
        print("DEBUG: Pestaña 'Análisis Bivariado y Multivariado' completada.")
    except Exception as e:
        st.error(f"Error en la pestaña 'Análisis Bivariado y Multivariado': {e}")
        st.exception(e)
        print(f"ERROR: Pestaña 'Análisis Bivariado y Multivariado' falló: {e}")

# --- Contenido de la Pestaña: Modelado Predictivo ---
with tab_modeling:
    try:
        print("DEBUG: Iniciando pestaña 'Modelado Predictivo'...")
        st.header("Modelado Predictivo (Esbozo)")
        st.markdown("""
            Esta sección demuestra un flujo de trabajo básico para construir un modelo
            predictivo del precio de venta (`sale_amount`) utilizando un **RandomForestRegressor**.
            Se incluyen pasos de preparación de datos, entrenamiento del modelo y evaluación.
            Ten en cuenta que este es un **esbozo simplificado** y no un modelo de producción optimizado.
        """)

        if df_filtered.empty:
            st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
        else:
            st.subheader("Preparación de Datos para el Modelado")
            st.info("Se creará un subconjunto de datos para el modelado, aplicando one-hot encoding y escalado.")

            df_model = df_filtered.copy()

            numerical_features = ['assessed_value', 'list_year']
            if 'list_year' not in df_model.columns:
                st.warning("La columna 'list_year' no está presente y no se incluirá como característica numérica.")
                numerical_features.remove('list_year')
            else:
                df_model['list_year'] = pd.to_numeric(df_model['list_year'], errors='coerce')
                if df_model['list_year'].isnull().any():
                    median_list_year = df_model['list_year'].median()
                    df_model['list_year'].fillna(median_list_year, inplace=True)
                    st.info(f"Valores faltantes en 'list_year' imputados con la mediana: {int(median_list_year)}.")
            
            categorical_features = ['town', 'property_type', 'residential_type', 'sale_month']
            categorical_features = [col for col in categorical_features if col in df_model.columns]

            target = 'sale_amount'

            features_for_model = numerical_features + categorical_features + [target]
            df_model.dropna(subset=features_for_model, inplace=True)

            if df_model.empty:
                st.warning("Después de la preparación, el DataFrame para el modelado está vacío. No se puede entrenar el modelo.")
                st.stop()

            st.write(f"Filas disponibles para modelado después de limpiar NaNs: {df_model.shape[0]}")

            df_model_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True, dtype=int)
            st.write(f"Número de columnas después de One-Hot Encoding: {df_model_encoded.shape[1]}")

            if target not in df_model_encoded.columns:
                st.error(f"La columna objetivo '{target}' no se encontró en el DataFrame después del encoding.")
                st.stop()

            X = df_model_encoded.drop(columns=[target])
            y = df_model_encoded[target]

            final_numerical_features = [col for col in numerical_features if col in X.columns]
            if not final_numerical_features:
                st.warning("No se encontraron características numéricas válidas para el escalado. Esto podría afectar el rendimiento del modelo.")

            if final_numerical_features:
                st.write(f"Escalando características numéricas: {final_numerical_features}")
                scaler = StandardScaler()
                X[final_numerical_features] = scaler.fit_transform(X[final_numerical_features])

            test_size = st.slider("Tamaño del conjunto de prueba (Test Size):", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            random_state = st.slider("Semilla aleatoria (Random State):", min_value=0, max_value=100, value=42, step=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            st.write(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
            st.write(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

            st.markdown("---")
            st.subheader("Entrenamiento del Modelo RandomForestRegressor")
            st.info("El RandomForestRegressor es un modelo de ensemble robusto, ideal para problemas de regresión.")

            n_estimators = st.slider("Número de estimadores (árboles):", min_value=50, max_value=500, value=100, step=50)
            max_depth = st.slider("Profundidad máxima de los árboles (0 para ilimitado):", min_value=0, max_value=20, value=10, step=1)
            actual_max_depth = None if max_depth == 0 else max_depth


            if st.button("Entrenar Modelo"):
                with st.spinner("Entrenando RandomForestRegressor... Esto puede tomar un momento."):
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=actual_max_depth, random_state=random_state, n_jobs=-1)
                    model.fit(X_train, y_train)

                st.success("¡Modelo entrenado exitosamente!")

                st.markdown("---")
                st.subheader("Evaluación del Modelo")
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write(f"**Error Absoluto Medio (MAE):** ${mae:,.2f}")
                st.write(f"**Coeficiente de Determinación (R²):** {r2:.4f}")

                st.info("""
                    **Interpretación:**
                    * **MAE:** Representa la magnitud promedio de los errores en las predicciones. Un valor más bajo es mejor.
                    * **R²:** Indica la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. Un valor más cercano a 1 indica un mejor ajuste del modelo.
                """)

                st.subheader("Visualización de Predicciones vs. Valores Reales")
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                results_df = results_df.sample(min(1000, len(results_df)), random_state=42)

                fig_pred_vs_actual = px.scatter(results_df, x='Actual', y='Predicted',
                                                title='Valores Reales vs. Predicciones',
                                                labels={'Actual': 'Precio de Venta Real', 'Predicted': 'Precio de Venta Predicho'})
                fig_pred_vs_actual.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)],
                                                        mode='lines', name='Línea Ideal (Actual = Predicho)',
                                                        line=dict(color='red', dash='dash')))
                st.plotly_chart(fig_pred_vs_actual, use_container_width=True)

                st.subheader("Importancia de las Características")
                if hasattr(model, 'feature_importances_'):
                    feature_importances = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values(by='Importance', ascending=False)

                    fig_feature_imp = px.bar(feature_importances.head(15), x='Importance', y='Feature', orientation='h',
                                            title='Top 15 Características Más Importantes',
                                            labels={'Importance': 'Importancia', 'Feature': 'Característica'})
                    st.plotly_chart(fig_feature_imp, use_container_width=True)
                else:
                    st.info("El modelo no soporta la extracción de importancia de características.")

                st.info("""
                    **Próximos pasos posibles:**
                    * Ajustar hiperparámetros del modelo (Grid Search, Random Search).
                    * Experimentar con diferentes características o ingeniería de características más compleja.
                    * Probar otros algoritmos de machine learning.
                    * Realizar validación cruzada para una evaluación más robusta.
                """)
            else:
                st.info("Haz clic en 'Entrenar Modelo' para iniciar el proceso de modelado.")
        print("DEBUG: Pestaña 'Modelado Predictivo' completada.")
    except Exception as e:
        st.error(f"Error en la pestaña 'Modelado Predictivo': {e}")
        st.exception(e)
        print(f"ERROR: Pestaña 'Modelado Predictivo' falló: {e}")
