import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- Configuración de la Base de Datos ---
DB_CONFIG = {
    "host": "localhost",
    "database": "pruebita",
    "user": "root",
    "password": "248613"
}

# --- Pool de Conexiones ---
@st.cache_resource
def init_connection():
    """Inicializa el pool de conexiones a la base de datos."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        return None

# --- Funciones de Consulta Optimizadas ---
@st.cache_data(ttl=3600)
def get_years_available():
    """Obtiene los años disponibles en la base de datos."""
    conn = init_connection()
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT anio_datos FROM casos_dengue ORDER BY anio_datos")
        years = [row[0] for row in cursor.fetchall()]
        return years
    except Error as e:
        st.error(f"Error al obtener años: {e}")
        return []
    finally:
        cursor.close()

@st.cache_data(ttl=3600)
def get_estados():
    """Obtiene la lista de estados con sus nombres."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT e.id_estado, e.nombre 
        FROM estados e
        ORDER BY e.nombre
        """
        df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Error al obtener estados: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_municipios(estado_id=None):
    """Obtiene la lista de municipios, opcionalmente filtrado por estado."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        if estado_id:
            query = """
            SELECT m.id, m.clave, m.municipio, m.entidad
            FROM municipios m
            WHERE m.entidad = %s
            ORDER BY m.municipio
            """
            df = pd.read_sql(query, conn, params=[estado_id])
        else:
            query = """
            SELECT m.id, m.clave, m.municipio, m.entidad
            FROM municipios m
            ORDER BY m.municipio
            """
            df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Error al obtener municipios: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_casos_por_anio(years=None):
    """Obtiene casos agrupados por año."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        if years:
            placeholders = ','.join(['%s'] * len(years))
            query = f"""
            SELECT 
                c.anio_datos,
                COUNT(*) as total_casos,
                SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
            FROM casos_dengue c
            WHERE c.anio_datos IN ({placeholders})
            GROUP BY c.anio_datos
            ORDER BY c.anio_datos
            """
            df = pd.read_sql(query, conn, params=years)
        else:
            query = """
            SELECT 
                c.anio_datos,
                COUNT(*) as total_casos,
                SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
            FROM casos_dengue c
            GROUP BY c.anio_datos
            ORDER BY c.anio_datos
            """
            df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Error al obtener casos por año: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_casos_por_estado(years=None):
    """Obtiene casos agrupados por estado."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        if years:
            placeholders = ','.join(['%s'] * len(years))
            query = f"""
            SELECT 
                e.nombre as estado,
                e.id_estado,
                COUNT(c.id) as total_casos,
                SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
            FROM casos_dengue c
            JOIN estados e ON c.ENTIDAD_RES = e.id_estado
            WHERE c.anio_datos IN ({placeholders})
            GROUP BY e.id_estado, e.nombre
            ORDER BY total_casos DESC
            """
            df = pd.read_sql(query, conn, params=years)
        else:
            query = """
            SELECT 
                e.nombre as estado,
                e.id_estado,
                COUNT(c.id) as total_casos,
                SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
            FROM casos_dengue c
            JOIN estados e ON c.ENTIDAD_RES = e.id_estado
            GROUP BY e.id_estado, e.nombre
            ORDER BY total_casos DESC
            """
            df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Error al obtener casos por estado: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_casos_por_municipio(estado_id=None, years=None, limit=20):
    """Obtiene casos agrupados por municipio."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        conditions = []
        params = []
        
        if years:
            placeholders = ','.join(['%s'] * len(years))
            conditions.append(f"c.anio_datos IN ({placeholders})")
            params.extend(years)
        
        if estado_id:
            conditions.append("c.ENTIDAD_RES = %s")
            params.append(estado_id)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT 
            m.municipio,
            e.nombre as estado,
            COUNT(c.id) as total_casos,
            SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
            SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
        FROM casos_dengue c
        JOIN estados e ON c.ENTIDAD_RES = e.id_estado
        JOIN municipios m ON c.MUNICIPIO_RES = m.clave AND c.ENTIDAD_RES = m.entidad
        {where_clause}
        GROUP BY m.municipio, e.nombre
        ORDER BY total_casos DESC
        LIMIT %s
        """
        params.append(limit)
        df = pd.read_sql(query, conn, params=params)
        return df
    except Error as e:
        st.error(f"Error al obtener casos por municipio: {e}")
        return pd.DataFrame()

# --- Configuración de la página ---
st.set_page_config(
    page_title="Dashboard Dengue México",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título y descripción ---
st.title("🦟 Dashboard de Casos de Dengue en México")
st.markdown("### Análisis de casos de dengue por año, estado y municipio")

# --- Sidebar para filtros ---
st.sidebar.header("🔍 Filtros")

# Obtener años disponibles
years_available = get_years_available()
if years_available:
    selected_years = st.sidebar.multiselect(
        "Seleccionar años:",
        options=years_available,
        default=years_available,
        help="Selecciona uno o más años para el análisis"
    )
else:
    st.error("No se pudieron cargar los años disponibles")
    st.stop()

# Obtener estados
df_estados = get_estados()
if not df_estados.empty:
    estado_options = ["Todos"] + df_estados['nombre'].tolist()
    selected_estado_name = st.sidebar.selectbox(
        "Seleccionar estado:",
        options=estado_options,
        help="Selecciona un estado específico o 'Todos' para ver todos"
    )
    
    if selected_estado_name != "Todos":
        selected_estado_id = df_estados[df_estados['nombre'] == selected_estado_name]['id_estado'].iloc[0]
    else:
        selected_estado_id = None
else:
    st.error("No se pudieron cargar los estados")
    st.stop()

# --- Contenido principal ---
if selected_years:
    # Métricas principales
    st.header("📊 Métricas Generales")
    
    # Obtener datos generales
    df_anios = get_casos_por_anio(selected_years)
    
    if not df_anios.empty:
        total_casos = df_anios['total_casos'].sum()
        total_defunciones = df_anios['defunciones'].sum()
        total_confirmados = df_anios['confirmados'].sum()
        tasa_mortalidad = (total_defunciones / total_casos * 100) if total_casos > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Casos", f"{total_casos:,}")
        with col2:
            st.metric("Casos Confirmados", f"{total_confirmados:,}")
        with col3:
            st.metric("Defunciones", f"{total_defunciones:,}")
        with col4:
            st.metric("Tasa de Mortalidad", f"{tasa_mortalidad:.2f}%")
    
    # Gráfica de casos por año
    st.header("📈 Tendencia por Años")
    
    if not df_anios.empty:
        fig_anios = go.Figure()
        
        fig_anios.add_trace(go.Bar(
            x=df_anios['anio_datos'],
            y=df_anios['total_casos'],
            name='Total de Casos',
            marker_color='lightblue',
            text=df_anios['total_casos'],
            textposition='auto'
        ))
        
        fig_anios.add_trace(go.Scatter(
            x=df_anios['anio_datos'],
            y=df_anios['defunciones'],
            mode='lines+markers',
            name='Defunciones',
            line=dict(color='red', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        fig_anios.update_layout(
            title='Casos de Dengue por Año',
            xaxis_title='Año',
            yaxis_title='Número de Casos',
            yaxis2=dict(
                title='Defunciones',
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_anios, use_container_width=True)
    
    # Gráficas por estado
    st.header("🗺️ Análisis por Estados")
    
    df_estados_casos = get_casos_por_estado(selected_years)
    
    if not df_estados_casos.empty:
        # Mostrar top 15 estados
        df_top_estados = df_estados_casos.head(15)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_estados_bar = px.bar(
                df_top_estados,
                x='total_casos',
                y='estado',
                orientation='h',
                title='Top 15 Estados con Más Casos',
                labels={'total_casos': 'Número de Casos', 'estado': 'Estado'},
                color='total_casos',
                color_continuous_scale='Reds'
            )
            fig_estados_bar.update_layout(height=600)
            st.plotly_chart(fig_estados_bar, use_container_width=True)
        
        with col2:
            fig_estados_pie = px.pie(
                df_top_estados,
                values='total_casos',
                names='estado',
                title='Distribución de Casos por Estado (Top 15)'
            )
            fig_estados_pie.update_layout(height=600)
            st.plotly_chart(fig_estados_pie, use_container_width=True)
    
    # Análisis por municipios
    st.header("🏘️ Análisis por Municipios")
    
    # Controles para municipios
    col1, col2 = st.columns([3, 1])
    with col1:
        if selected_estado_name != "Todos":
            st.subheader(f"Municipios en {selected_estado_name}")
        else:
            st.subheader("Municipios con Más Casos (Nacional)")
    
    with col2:
        num_municipios = st.selectbox(
            "Mostrar top:",
            options=[10, 20, 30, 50],
            index=1,
            help="Número de municipios a mostrar"
        )
    
    df_municipios = get_casos_por_municipio(
        estado_id=selected_estado_id,
        years=selected_years,
        limit=num_municipios
    )
    
    if not df_municipios.empty:
        # Crear nombre completo para municipios
        df_municipios['municipio_completo'] = df_municipios['municipio'] + ', ' + df_municipios['estado']
        
        fig_municipios = px.bar(
            df_municipios,
            x='total_casos',
            y='municipio_completo',
            orientation='h',
            title=f'Top {num_municipios} Municipios con Más Casos',
            labels={'total_casos': 'Número de Casos', 'municipio_completo': 'Municipio'},
            color='total_casos',
            color_continuous_scale='Blues',
            text='total_casos'
        )
        
        fig_municipios.update_layout(
            height=max(400, num_municipios * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        fig_municipios.update_traces(textposition='outside')
        
        st.plotly_chart(fig_municipios, use_container_width=True)
        
        # Tabla de municipios
        st.subheader("📋 Detalle por Municipios")
        df_municipios_display = df_municipios[['municipio', 'estado', 'total_casos', 'confirmados', 'defunciones']].copy()
        df_municipios_display.columns = ['Municipio', 'Estado', 'Total Casos', 'Confirmados', 'Defunciones']
        st.dataframe(df_municipios_display, use_container_width=True)
    
    else:
        st.warning("No se encontraron datos de municipios para los filtros seleccionados.")

else:
    st.warning("Por favor, selecciona al menos un año para mostrar los datos.")

# --- Información adicional ---
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Sobre este Dashboard**")
st.sidebar.markdown("Este dashboard muestra datos de casos de dengue en México, permitiendo análisis por año, estado y municipio.")
st.sidebar.markdown("**🔄 Actualización:** Los datos se actualizan automáticamente cada 30 minutos.")