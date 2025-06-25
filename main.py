import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler # Importar StandardScaler

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

        # Limpiar columnas categóricas para filtros y análisis
        for col in ['town', 'property_type', 'residential_type']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace('nan', np.nan)

        return df
    except Exception as e:
        st.error(f"Error al intentar cargar o preprocesar el archivo CSV: {e}")
        st.info("Asegúrate de que la URL sea correcta y que el archivo sea un CSV válido.")
        return pd.DataFrame()

# --- Cargar los Datos Originales ---
df_original = load_and_preprocess_data(CSV_URL)

if df_original.empty:
    st.warning("No se pudieron cargar los datos o el DataFrame está vacío. No se puede continuar con el análisis.")
    st.stop() # Detiene la ejecución si no hay datos

# --- Sidebar para Filtros (aplicados a todas las pestañas) ---
st.sidebar.header("Filtros de Datos")
st.sidebar.info("Usa los filtros a continuación para segmentar los datos en todas las secciones de la aplicación.")

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
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
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

# 3. Filtro por Ciudad (town)
if 'town' in df_filtered.columns and not df_filtered['town'].isnull().all():
    all_towns = df_filtered['town'].dropna().unique().tolist()
    all_towns.sort()
    selected_towns = st.sidebar.multiselect(
        "Selecciona Ciudades:",
        options=all_towns,
        default=[] # Por defecto, ninguna ciudad seleccionada para explorar
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
        default=all_property_types
    )
    if selected_property_types:
        df_filtered = df_filtered[df_filtered['property_type'].isin(selected_property_types)]
else:
    st.sidebar.warning("Columna 'property_type' no disponible o vacía para filtrar.")

# 5. Filtro por Tipo Residencial (residential_type)
if 'residential_type' in df_filtered.columns and not df_filtered['residential_type'].isnull().all():
    # Solo mostrar tipos residenciales si 'Residential' está en los tipos de propiedad seleccionados
    if 'Residential' in selected_property_types:
        all_residential_types = df_filtered['residential_type'].dropna().unique().tolist()
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

st.sidebar.markdown("---")
st.sidebar.write(f"**Registros seleccionados:** {df_filtered.shape[0]} de {df_original.shape[0]}")


# --- Pestañas de la Aplicación ---
tab_eda, tab_clean, tab_feat_eng, tab_outliers, tab_bivariate, tab_modeling = st.tabs([
    "📊 EDA General",
    "🧹 Limpieza Avanzada",
    "💡 Ingeniería de Características",
    "🕵️ Detección de Outliers",
    "📈 Análisis Bivariado",
    "🤖 Modelado Predictivo"
])

# --- Contenido de la Pestaña: EDA General ---
with tab_eda:
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
        st.dataframe(df_filtered.describe().T)

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
            st.info("Columnas con alta proporción de valores faltantes podrían requerir imputación cuidadosa o exclusión. Explora la pestaña 'Limpieza Avanzada'.")
        else:
            st.info("¡No hay valores faltantes en el dataset filtrado!")

        st.header("3. Visualizaciones Clave (Filtros Aplicados)")

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

        # Ventas Anuales a lo Largo del Tiempo
        st.subheader("Ventas Anuales a lo Largo del Tiempo (Filtradas)")
        if 'sale_year' in df_filtered.columns and not df_filtered['sale_year'].isnull().all():
            sales_over_time_filtered = df_filtered.groupby('sale_year').size().reset_index(name='Número de Ventas')
            fig_time_series_filtered = px.line(sales_over_time_filtered, x='sale_year', y='Número de Ventas',
                                              title='Número de Ventas por Año (Filtradas)',
                                              labels={'sale_year': 'Año de Venta', 'Número de Ventas': 'Número de Transacciones'})
            st.plotly_chart(fig_time_series_filtered, use_container_width=True)
        else:
            st.warning("No se pudo realizar el análisis temporal sin la columna 'sale_year' en los datos filtrados.")

# --- Contenido de la Pestaña: Limpieza Avanzada ---
with tab_clean:
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
        if not current_missing_df.empty:
            st.dataframe(current_missing_df)
        else:
            st.info("¡No hay valores faltantes en el DataFrame actual (filtrado)! Excelente.")

        st.markdown("---")
        st.subheader("Estrategias de Imputación/Eliminación")

        df_cleaned_temp = df_filtered.copy() # Copia temporal para demostración de limpieza

        st.write("#### 1. Columnas con alta proporción de NaNs (Ejemplo: `non_use_code`, `assessor_remarks`, `opm_remarks`)")
        cols_to_drop = ['non_use_code', 'assessor_remarks', 'opm_remarks', 'location'] # 'location' si no se va a usar para geomapping
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_cleaned_temp.columns]
        if cols_to_drop_existing:
            st.write(f"Columnas candidatas a eliminación por muchos NaNs o irrelevancia: `{', '.join(cols_to_drop_existing)}`")
            if st.checkbox("Eliminar estas columnas para la demostración?", key="clean_drop_cols"):
                df_cleaned_temp.drop(columns=cols_to_drop_existing, inplace=True)
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
                    df_cleaned_temp['residential_type'].fillna('Unknown', inplace=True)
                    st.success("Valores faltantes en 'residential_type' imputados con 'Unknown'.")
                elif imputation_option == "Imputar con la Moda":
                    mode_val = df_cleaned_temp['residential_type'].mode()[0]
                    df_cleaned_temp['residential_type'].fillna(mode_val, inplace=True)
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
                for col in numeric_cols_with_nan:
                    if imputation_method == "Mediana":
                        val = df_cleaned_temp[col].median()
                    else: # Media
                        val = df_cleaned_temp[col].mean()
                    df_cleaned_temp[col].fillna(val, inplace=True)
                    st.success(f"Valores faltantes en '{col}' imputados con la {imputation_method.lower()}: {val:,.2f}.")
        else:
            st.info("No hay columnas numéricas con valores faltantes en el DataFrame actual.")

        st.subheader("Estado del DataFrame después de la Limpieza Demostrativa:")
        st.dataframe(df_cleaned_temp.isnull().sum()[df_cleaned_temp.isnull().sum() > 0].sort_values(ascending=False))
        if df_cleaned_temp.isnull().sum().sum() == 0:
            st.success("¡DataFrame temporal sin valores faltantes!")

# --- Contenido de la Pestaña: Ingeniería de Características ---
with tab_feat_eng:
    st.header("Ingeniería de Características")
    st.markdown("""
        En esta sección, creamos nuevas características a partir de las existentes para potenciar el análisis y el modelado.
        Las características generadas aquí se utilizan en las secciones posteriores.
    """)

    if df_filtered.empty:
        st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
    else:
        df_fe = df_filtered.copy() # DataFrame para feature engineering

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
            # --- FIX STARTS HERE ---
            # Get value counts and reset index, then explicitly rename columns for clarity
            price_category_counts = df_fe['sale_amount_category'].value_counts().reset_index()
            price_category_counts.columns = ['Category', 'Count'] # Assign clear column names

            fig_price_cat = px.bar(price_category_counts,
                                   x='Category', y='Count', # Use the new explicit column names
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

# --- Contenido de la Pestaña: Detección de Outliers ---
with tab_outliers:
    st.header("Detección de Outliers")
    st.markdown("""
        Identificamos valores atípicos en las columnas numéricas clave, que podrían influir en el modelado.
        Aquí usamos el **método del Rango Intercuartílico (IQR)**.
    """)

    if df_filtered.empty:
        st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
    else:
        df_outlier = df_filtered.copy() # DataFrame para detección de outliers

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

# --- Contenido de la Pestaña: Análisis Bivariado/Multivariado ---
with tab_bivariate:
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
        if 'sale_month' in df_filtered.columns and 'sale_amount' in df_filtered.columns:
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            df_plot = df_filtered.copy()
            df_plot['sale_month'] = pd.Categorical(df_plot['sale_month'], categories=month_order, ordered=True)
            df_plot.sort_values('sale_month', inplace=True)

            fig_box_month = px.box(df_plot, x='sale_month', y='sale_amount',
                                   title='Precio de Venta por Mes',
                                   labels={'sale_month': 'Mes de Venta', 'sale_amount': 'Precio de Venta'},
                                   points="outliers", log_y=True)
            st.plotly_chart(fig_box_month, use_container_width=True)
            st.info("Compara la distribución de precios entre diferentes meses.")
        else:
            st.warning("Columnas 'sale_month' o 'sale_amount' no disponibles para este gráfico.")

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

# --- Contenido de la Pestaña: Modelado Predictivo ---
with tab_modeling:
    st.header("Modelado Predictivo (Esbozo)")
    st.markdown("""
        Esta sección demuestra un flujo de trabajo básico para construir un modelo de regresión
        para predecir el **Precio de Venta (`sale_amount`)**.
        Los pasos incluyen selección de características, codificación y entrenamiento de un **Random Forest Regressor**.
    """)

    if df_filtered.empty:
        st.warning("El DataFrame filtrado está vacío. Ajusta tus filtros para ver datos en esta sección.")
    else:
        df_model = df_filtered.copy() # Copia para el modelado

        st.subheader("1. Preprocesamiento para el Modelo")
        st.write("##### Manejo de NaNs (Estrategia para el Modelo)")
        # Para el modelado, seremos más agresivos con los NaNs en columnas clave
        cols_for_model = ['assessed_value', 'list_year', 'property_type', 'town', 'residential_type', 'sale_amount']
        df_model = df_model[cols_for_model].dropna() # Eliminar filas con NaNs en estas columnas

        if df_model.empty:
            st.warning("El DataFrame para el modelado está vacío después de eliminar NaNs. Ajusta tus filtros o reconsidera las columnas a usar.")
            st.stop()

        st.write("##### Codificación de Variables Categóricas (One-Hot Encoding)")
        categorical_features_for_model = ['property_type', 'town', 'residential_type']
        # Asegurarse de que las columnas existen antes de codificar
        categorical_features_for_model = [col for col in categorical_features_for_model if col in df_model.columns]

        if categorical_features_for_model:
            df_model = pd.get_dummies(df_model, columns=categorical_features_for_model, drop_first=True)
            st.success("Variables categóricas codificadas con One-Hot Encoding.")
        else:
            st.info("No hay columnas categóricas válidas para One-Hot Encoding.")

        st.write("##### Escalado de Características Numéricas")
        numeric_features_for_scaling = ['assessed_value', 'list_year']
        numeric_features_for_scaling = [col for col in numeric_features_for_scaling if col in df_model.columns]

        if numeric_features_for_scaling:
            scaler = StandardScaler()
            df_model[numeric_features_for_scaling] = scaler.fit_transform(df_model[numeric_features_for_scaling])
            st.success("Características numéricas escaladas.")
        else:
            st.info("No hay columnas numéricas válidas para escalado.")


        st.subheader("2. Selección de Características y División de Datos")
        target = 'sale_amount'
        # Quitar la variable objetivo de las características
        X = df_model.drop(columns=[target])
        y = df_model[target]

        st.write(f"**Características (X) para el modelo:**")
        st.write(X.columns.tolist())
        st.write(f"**Variable Objetivo (y):** {target}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        st.write(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        st.write(f"Conjunto de prueba: {X_test.shape[0]} muestras")

        st.subheader("3. Entrenamiento y Evaluación del Modelo (Random Forest Regressor)")
        if st.button("Entrenar Modelo"):
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            st.success("Modelo Random Forest Regressor entrenado exitosamente.")

            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"**Error Absoluto Medio (MAE):** ${mae:,.2f}")
            st.write(f"**Coeficiente de Determinación (R2):** {r2:.4f}")

            st.info("Un R2 cercano a 1 indica que el modelo explica una gran parte de la varianza del precio de venta.")
            st.info("Un MAE indica el error promedio en las predicciones del precio de venta.")

            st.subheader("Importancia de las Características (Top 10)")
            feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig_importance = px.bar(feature_importances.head(10),
                                    title="Importancia de las Características en el Modelo",
                                    labels={'value': 'Importancia', 'index': 'Característica'})
            st.plotly_chart(fig_importance, use_container_width=True)

        else:
            st.info("Haz clic en 'Entrenar Modelo' para ver los resultados.")

    st.markdown("---")
    st.caption("Aplicación creada con Streamlit. Datos de JulianTorrest/prueba_real_state.")
