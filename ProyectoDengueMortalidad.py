import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import plotly.express as px

# --- Configuración de la Base de Datos ---
# IMPORTANTE: Asegúrate de que estos detalles sean correctos para tu configuración de MySQL.
# Si tu MySQL no está en localhost, reemplaza 'localhost' con la IP o el nombre de host.
DB_CONFIG = {
    "host": "localhost",
    "database": "denguebd",  # Tu base de datos se llama 'denguebd'
    "user": "root",          # ¡REEMPLAZA con tu usuario de MySQL!
    "password": "248613"     # ¡REEMPLAZA con tu contraseña de MySQL!
}

# --- Función de Conexión a la Base de Datos ---
# Esta función ahora será usada internamente por la función cacheada.
def create_db_connection():
    """Establece una conexión a la base de datos. Retorna el objeto de conexión o None en caso de error."""
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error al conectar a la base de datos MySQL: {e}")
    return None

# --- Función de Obtención de Datos con Caché ---
# Almacena en caché los datos durante 1 hora (3600 segundos) para evitar llamadas repetidas a la base de datos al volver a ejecutar.
# ¡IMPORTANTE!: Esta función ahora se conecta y desconecta de la base de datos por sí misma.
@st.cache_data(ttl=3600)
def get_dengue_data(): # <--- CAMBIO AQUÍ: Ya no recibe 'conn' como argumento
    """
    Obtiene los datos de casos de dengue de la tabla 'casos_dengue' para los años 2020-2025.
    Establece su propia conexión a la base de datos.
    """
    conn = create_db_connection() # <--- CAMBIO AQUÍ: Crea la conexión internamente
    if conn is None:
        st.warning("No se pudo establecer conexión con la base de datos para cargar los datos.")
        return pd.DataFrame()

    cursor = conn.cursor(dictionary=True)
    # Consulta para seleccionar todos los datos dentro del rango de años especificado
    query = "SELECT * FROM casos_dengue WHERE anio_datos BETWEEN 2020 AND 2025"
    try:
        cursor.execute(query)
        data = cursor.fetchall()  # Obtener todas las filas
        df = pd.DataFrame(data)   # Convertir a DataFrame de pandas
        st.success("Datos de dengue cargados exitosamente.")
        return df
    except Error as e:
        st.error(f"Error al obtener los datos de dengue: {e}")
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected(): # <--- CAMBIO AQUÍ: Cierra la conexión internamente
            conn.close()
            # st.info("Conexión a la base de datos cerrada por get_dengue_data.") # Opcional para depurar

# --- Mapeos de Datos ---
# Estos mapeos se asumen en base a prácticas comunes.
# Por favor, ajústalos si tu base de datos utiliza códigos diferentes.
SEXO_MAP = {
    1: "Hombre",
    2: "Mujer",
    99: "No especificado" # Marcador de posición para valores desconocidos/no especificados
}

TIPO_PACIENTE_MAP = {
    1: "Ambulatorio",
    2: "Hospitalizado"
}

DEFUNCION_MAP = {
    1: "Sí (Fallecido)",
    2: "No (Vivo)",
    99: "No aplica/Desconocido"
}

ESTATUS_CASO_MAP = {
    1: "Confirmado",
    2: "Descartado",
    3: "Probable"
}

# --- Diseño y Lógica de la Aplicación Streamlit ---
# Configurar la página de Streamlit para un diseño amplio
st.set_page_config(layout="wide", page_title="Dashboard de Casos de Dengue")

st.title("📊 Dashboard de Casos de Dengue (2020-2025)")
st.markdown("Explora las métricas y tendencias de los casos de dengue por estado y municipio.")

# Ya no necesitamos una conexión persistente aquí, ya que la función cacheada la maneja.
# Intentar conectar a la base de datos (solo para verificación inicial de la conexión, no se usa para pasar a get_dengue_data)
initial_conn_check = create_db_connection() # Verifica si se puede conectar al inicio
if initial_conn_check:
    if initial_conn_check.is_connected():
        initial_conn_check.close() # Cierra la conexión de verificación inicial
    # Obtener datos usando la función en caché
    # Ahora la función get_dengue_data() no necesita argumentos de conexión
    df_dengue = get_dengue_data() # <--- CAMBIO AQUÍ: Llamada sin argumentos

    # Continuar solo si los datos se cargaron exitosamente
    if not df_dengue.empty:
        # --- Preprocesamiento de Datos ---
        # Aplicar etiquetas legibles por humanos usando los mapeos definidos
        df_dengue['SEXO_LABEL'] = df_dengue['SEXO'].map(SEXO_MAP).fillna("Desconocido")
        df_dengue['TIPO_PACIENTE_LABEL'] = df_dengue['TIPO_PACIENTE'].map(TIPO_PACIENTE_MAP).fillna("Desconocido")
        df_dengue['DEFUNCION_LABEL'] = df_dengue['DEFUNCION'].map(DEFUNCION_MAP).fillna("Desconocido")
        df_dengue['ESTATUS_CASO_LABEL'] = df_dengue['ESTATUS_CASO'].map(ESTATUS_CASO_MAP).fillna("Desconocido")

        # Convertir 'FECHA_SIGN_SINTOMAS' a objetos datetime para posibles análisis de series temporales
        df_dengue['FECHA_SIGN_SINTOMAS'] = pd.to_datetime(df_dengue['FECHA_SIGN_SINTOMAS'], errors='coerce')
        # Extraer el nombre del mes para obtener información de series temporales
        df_dengue['MES_SINTOMAS'] = df_dengue['FECHA_SIGN_SINTOMAS'].dt.month_name(locale='es_ES') # Usar nombres de meses en español

        # Crear grupos de edad para un mejor análisis demográfico
        bins = [0, 5, 12, 18, 30, 50, 65, 120] # Límite superior extendido a 120 para mayor robustez
        labels = ['0-4', '5-11', '12-17', '18-29', '30-49', '50-64', '65+']
        df_dengue['GRUPO_EDAD'] = pd.cut(df_dengue['EDAD_ANOS'], bins=bins, labels=labels, right=False, include_lowest=True)

        # --- Filtros de la Barra Lateral ---
        st.sidebar.header("Filtros de Datos")

        # Filtro de año: permite la selección de múltiples años
        all_years = sorted(df_dengue['anio_datos'].unique().tolist())
        selected_years = st.sidebar.multiselect("Seleccionar Año(s)", all_years, default=all_years)

        # Filtro de estado: permite la selección de múltiples estados (ENTIDAD_RES son IDs enteros)
        # Se recomienda mapear estos IDs a nombres de estados reales si es posible para una mejor legibilidad.
        all_states = sorted(df_dengue['ENTIDAD_RES'].unique().tolist())
        selected_states = st.sidebar.multiselect("Seleccionar Estado (ID ENTIDAD_RES)", all_states, default=all_states)

        # Filtro de municipio: dependiente de los estados seleccionados
        # Primero filtra el DataFrame por los estados seleccionados para obtener los municipios relevantes
        filtered_by_state_df = df_dengue[df_dengue['ENTIDAD_RES'].isin(selected_states)]
        all_municipalities = sorted(filtered_by_state_df['MUNICIPIO_RES'].unique().tolist())
        selected_municipalities = st.sidebar.multiselect("Seleccionar Municipio (ID MUNICIPIO_RES)", all_municipalities, default=all_municipalities)

        # Aplicar todos los filtros seleccionados para crear el DataFrame filtrado final
        filtered_df = df_dengue[
            (df_dengue['anio_datos'].isin(selected_years)) &
            (df_dengue['ENTIDAD_RES'].isin(selected_states)) &
            (df_dengue['MUNICIPIO_RES'].isin(selected_municipalities))
        ]

        # --- Mostrar Contenido del Dashboard ---
        if filtered_df.empty:
            st.warning("No hay datos disponibles para la selección de filtros actual. Por favor, ajusta tus filtros.")
        else:
            st.header("Métricas Clave (KPIs)")

            # Usar columnas para mostrar los KPIs lado a lado
            col1, col2, col3, col4 = st.columns(4)

            total_cases = len(filtered_df)
            # Contar defunciones donde DEFUNCION es 1
            total_deaths = filtered_df[filtered_df['DEFUNCION'] == 1].shape[0]
            # Contar casos confirmados donde ESTATUS_CASO es 1
            confirmed_cases = filtered_df[filtered_df['ESTATUS_CASO'] == 1].shape[0]
            # Contar casos hospitalizados donde TIPO_PACIENTE es 2
            hospitalized_cases = filtered_df[filtered_df['TIPO_PACIENTE'] == 2].shape[0]

            with col1:
                st.metric("Total de Casos", total_cases)
            with col2:
                st.metric("Casos Confirmados", confirmed_cases)
            with col3:
                st.metric("Total de Defunciones", total_deaths)
            with col4:
                st.metric("Casos Hospitalizados", hospitalized_cases)


            st.markdown("---") # Separador visual
            st.header("Visualizaciones de Datos")

            # --- Fila de Gráficos 1: Casos por Año y Casos por Sexo ---
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Casos de Dengue por Año")
                # Agrupar por año y contar casos
                cases_by_year = filtered_df['anio_datos'].value_counts().sort_index().reset_index()
                cases_by_year.columns = ['Año', 'Número de Casos'] # Renombrar columnas para mayor claridad
                fig_year = px.bar(cases_by_year, x='Año', y='Número de Casos',
                                  title='Número de Casos de Dengue por Año',
                                  labels={'Número de Casos': 'Casos'},
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_year, use_container_width=True)

            with chart_col2:
                st.subheader("Distribución de Casos por Sexo")
                # Agrupar por SEXO_LABEL y contar casos
                cases_by_sex = filtered_df['SEXO_LABEL'].value_counts().reset_index()
                cases_by_sex.columns = ['Sexo', 'Número de Casos']
                fig_sex = px.pie(cases_by_sex, values='Número de Casos', names='Sexo',
                                 title='Distribución de Casos por Sexo',
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_sex, use_container_width=True)

            # --- Fila de Gráficos 2: Casos por Estado y Casos por Municipio ---
            chart_col3, chart_col4 = st.columns(2)

            with chart_col3:
                st.subheader("Casos por Estado de Residencia")
                # Agrupar por ENTIDAD_RES y contar casos
                cases_by_state = filtered_df['ENTIDAD_RES'].value_counts().sort_index().reset_index()
                cases_by_state.columns = ['Estado (ID)', 'Número de Casos']
                fig_state = px.bar(cases_by_state, x='Estado (ID)', y='Número de Casos',
                                   title='Casos de Dengue por Estado de Residencia',
                                   labels={'Número de Casos': 'Casos'},
                                   color_discrete_sequence=px.colors.qualitative.Pastel1)
                st.plotly_chart(fig_state, use_container_width=True)

            with chart_col4:
                st.subheader("Casos por Municipio de Residencia")
                # Agrupar por MUNICIPIO_RES y contar casos, mostrar los 10 primeros para mayor legibilidad
                cases_by_municipality = filtered_df['MUNICIPIO_RES'].value_counts().head(10).reset_index()
                cases_by_municipality.columns = ['Municipio (ID)', 'Número de Casos']
                fig_municipality = px.bar(cases_by_municipality, x='Municipio (ID)', y='Número de Casos',
                                         title='Top 10 Municipios con Casos de Dengue',
                                         labels={'Número de Casos': 'Casos'},
                                         color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig_municipality, use_container_width=True)

            # --- Fila de Gráficos 3: Casos por Tipo de Paciente y Casos por Estatus del Caso ---
            chart_col5, chart_col6 = st.columns(2)

            with chart_col5:
                st.subheader("Distribución de Casos por Tipo de Paciente")
                # Agrupar por TIPO_PACIENTE_LABEL y contar casos
                cases_by_patient_type = filtered_df['TIPO_PACIENTE_LABEL'].value_counts().reset_index()
                cases_by_patient_type.columns = ['Tipo de Paciente', 'Número de Casos']
                fig_patient_type = px.pie(cases_by_patient_type, values='Número de Casos', names='Tipo de Paciente',
                                          title='Distribución de Casos por Tipo de Paciente',
                                          color_discrete_sequence=px.colors.qualitative.Pastel2)
                st.plotly_chart(fig_patient_type, use_container_width=True)

            with chart_col6:
                st.subheader("Distribución de Casos por Estatus")
                # Agrupar por ESTATUS_CASO_LABEL y contar casos
                cases_by_status = filtered_df['ESTATUS_CASO_LABEL'].value_counts().reset_index()
                cases_by_status.columns = ['Estatus del Caso', 'Número de Casos']
                fig_status = px.pie(cases_by_status, values='Número de Casos', names='Estatus del Caso',
                                    title='Distribución de Casos por Estatus',
                                    color_discrete_sequence=px.colors.qualitative.D3)
                st.plotly_chart(fig_status, use_container_width=True)

            # --- Fila de Gráficos 4: Casos por Grupo de Edad y Casos por Estado de Defunción ---
            chart_col7, chart_col8 = st.columns(2)

            with chart_col7:
                st.subheader("Casos por Grupo de Edad")
                # Agrupar por GRUPO_EDAD y contar casos, ordenar por el orden del grupo de edad
                cases_by_age_group = filtered_df['GRUPO_EDAD'].value_counts().reindex(labels).reset_index()
                cases_by_age_group.columns = ['Grupo de Edad', 'Número de Casos']
                fig_age = px.bar(cases_by_age_group, x='Grupo de Edad', y='Número de Casos',
                                 title='Número de Casos de Dengue por Grupo de Edad',
                                 labels={'Número de Casos': 'Casos'},
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
                st.plotly_chart(fig_age, use_container_width=True)

            with chart_col8:
                st.subheader("Casos por Resultado de Defunción")
                # Agrupar por DEFUNCION_LABEL y contar casos
                cases_by_defuncion = filtered_df['DEFUNCION_LABEL'].value_counts().reset_index()
                cases_by_defuncion.columns = ['Resultado de Defunción', 'Número de Casos']
                fig_defuncion = px.pie(cases_by_defuncion, values='Número de Casos', names='Resultado de Defunción',
                                       title='Distribución de Casos por Resultado de Defunción',
                                       color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_defuncion, use_container_width=True)

            st.markdown("---")
            st.subheader("Datos Crudos (Primeras 100 Filas)")
            st.dataframe(filtered_df.head(100), use_container_width=True)

    else:
        st.error("No se pudieron cargar los datos de dengue o el DataFrame está vacío.")
else:
    st.error("No se pudo establecer conexión con la base de datos MySQL. Por favor, verifica la configuración.")
