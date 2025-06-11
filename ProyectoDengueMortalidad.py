import streamlit as st
import mysql.connector
from mysql.connector import Error
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

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
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados,
                SUM(CASE WHEN c.HEMORRAGICOS = 1 THEN 1 ELSE 0 END) as hemorragicos,
                SUM(CASE WHEN c.SEXO = 1 THEN 1 ELSE 0 END) as hombres,
                SUM(CASE WHEN c.SEXO = 2 THEN 1 ELSE 0 END) as mujeres
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
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados,
                SUM(CASE WHEN c.HEMORRAGICOS = 1 THEN 1 ELSE 0 END) as hemorragicos,
                SUM(CASE WHEN c.SEXO = 1 THEN 1 ELSE 0 END) as hombres,
                SUM(CASE WHEN c.SEXO = 2 THEN 1 ELSE 0 END) as mujeres
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
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados,
                SUM(CASE WHEN c.HEMORRAGICOS = 1 THEN 1 ELSE 0 END) as hemorragicos
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
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados,
                SUM(CASE WHEN c.HEMORRAGICOS = 1 THEN 1 ELSE 0 END) as hemorragicos
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
        base_query = """
        SELECT 
            m.municipio,
            e.nombre as estado,
            COUNT(c.id) as total_casos,
            SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
            SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
        FROM casos_dengue c
        JOIN estados e ON c.ENTIDAD_RES = e.id_estado
        JOIN municipios m ON c.MUNICIPIO_RES = m.clave AND c.ENTIDAD_RES = m.entidad
        """
        
        conditions = []
        params = []
        
        if years:
            years_list = [int(year) for year in years]
            placeholders = ','.join(['%s'] * len(years_list))
            conditions.append(f"c.anio_datos IN ({placeholders})")
            params.extend(years_list)
        
        if estado_id:
            conditions.append("c.ENTIDAD_RES = %s")
            params.append(int(estado_id))
        
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""
        
        complete_query = f"""
        {base_query}
        {where_clause}
        GROUP BY m.municipio, e.nombre
        ORDER BY total_casos DESC
        LIMIT {int(limit)}
        """
        
        cursor = conn.cursor()
        cursor.execute(complete_query, params)
        
        columns = ['municipio', 'estado', 'total_casos', 'defunciones', 'confirmados']
        results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=columns)
        
        cursor.close()
        return df
        
    except Error as e:
        st.error(f"Error al obtener casos por municipio: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error inesperado en get_casos_por_municipio: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_casos_por_edad_grupo(years=None):
    """Obtiene casos agrupados por grupos de edad."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        if years:
            placeholders = ','.join(['%s'] * len(years))
            query = f"""
            SELECT 
                CASE 
                    WHEN c.EDAD_ANOS < 5 THEN '0-4 años'
                    WHEN c.EDAD_ANOS < 15 THEN '5-14 años'
                    WHEN c.EDAD_ANOS < 25 THEN '15-24 años'
                    WHEN c.EDAD_ANOS < 45 THEN '25-44 años'
                    WHEN c.EDAD_ANOS < 65 THEN '45-64 años'
                    ELSE '65+ años'
                END as grupo_edad,
                COUNT(*) as total_casos,
                SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
            FROM casos_dengue c
            WHERE c.anio_datos IN ({placeholders}) AND c.EDAD_ANOS IS NOT NULL
            GROUP BY grupo_edad
            ORDER BY 
                CASE 
                    WHEN grupo_edad = '0-4 años' THEN 1
                    WHEN grupo_edad = '5-14 años' THEN 2
                    WHEN grupo_edad = '15-24 años' THEN 3
                    WHEN grupo_edad = '25-44 años' THEN 4
                    WHEN grupo_edad = '45-64 años' THEN 5
                    WHEN grupo_edad = '65+ años' THEN 6
                END
            """
            df = pd.read_sql(query, conn, params=years)
        else:
            query = """
            SELECT 
                CASE 
                    WHEN c.EDAD_ANOS < 5 THEN '0-4 años'
                    WHEN c.EDAD_ANOS < 15 THEN '5-14 años'
                    WHEN c.EDAD_ANOS < 25 THEN '15-24 años'
                    WHEN c.EDAD_ANOS < 45 THEN '25-44 años'
                    WHEN c.EDAD_ANOS < 65 THEN '45-64 años'
                    ELSE '65+ años'
                END as grupo_edad,
                COUNT(*) as total_casos,
                SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
                SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados
            FROM casos_dengue c
            WHERE c.EDAD_ANOS IS NOT NULL
            GROUP BY grupo_edad
            ORDER BY 
                CASE 
                    WHEN grupo_edad = '0-4 años' THEN 1
                    WHEN grupo_edad = '5-14 años' THEN 2
                    WHEN grupo_edad = '15-24 años' THEN 3
                    WHEN grupo_edad = '25-44 años' THEN 4
                    WHEN grupo_edad = '45-64 años' THEN 5
                    WHEN grupo_edad = '65+ años' THEN 6
                END
            """
            df = pd.read_sql(query, conn)
        return df
    except Error as e:
        st.error(f"Error al obtener casos por edad: {e}")
        return pd.DataFrame()

# --- NUEVA FUNCIÓN PARA DATOS DETALLADOS DE PREDICCIÓN ---
@st.cache_data(ttl=1800)
def get_datos_prediccion_detallados(years, estado_id=None):
    """Obtiene datos detallados para el modelo de predicción por mes/trimestre."""
    conn = init_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        placeholders = ','.join(['%s'] * len(years))
        
        base_query = f"""
        SELECT 
            c.anio_datos,
            MONTH(c.FECHA_SIGN_SINTOMAS) as mes,
            QUARTER(c.FECHA_SIGN_SINTOMAS) as trimestre,
            COUNT(*) as total_casos,
            SUM(CASE WHEN c.DEFUNCION = 1 THEN 1 ELSE 0 END) as defunciones,
            SUM(CASE WHEN c.ESTATUS_CASO = 1 THEN 1 ELSE 0 END) as confirmados,
            SUM(CASE WHEN c.HEMORRAGICOS = 1 THEN 1 ELSE 0 END) as hemorragicos,
            AVG(c.EDAD_ANOS) as edad_promedio,
            SUM(CASE WHEN c.SEXO = 2 THEN 1 ELSE 0 END) / COUNT(*) * 100 as porcentaje_mujeres
        FROM casos_dengue c
        WHERE c.anio_datos IN ({placeholders})
        AND c.FECHA_SIGN_SINTOMAS IS NOT NULL
        """
        
        params = years
        
        if estado_id:
            base_query += " AND c.ENTIDAD_RES = %s"
            params = years + [estado_id]
        
        base_query += """
        GROUP BY c.anio_datos, mes, trimestre
        ORDER BY c.anio_datos, mes
        """
        
        df = pd.read_sql(base_query, conn, params=params)
        return df
        
    except Error as e:
        st.error(f"Error al obtener datos para predicción: {e}")
        return pd.DataFrame()

# --- FUNCIONES DE PREDICCIÓN ---
def preparar_datos_para_prediccion(df):
    """Prepara los datos para el modelo de predicción."""
    if df.empty:
        return None, None, None
    
    # Crear variables de fecha
    df['fecha'] = pd.to_datetime(df['anio_datos'].astype(str) + '-' + df['mes'].astype(str) + '-01')
    df = df.sort_values('fecha')
    
    # Crear variables estacionales
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # Variables lag (valores anteriores)
    df['casos_lag1'] = df['total_casos'].shift(1)
    df['casos_lag3'] = df['total_casos'].shift(3)
    df['casos_lag12'] = df['total_casos'].shift(12)  # Año anterior mismo mes
    
    # Variables de tendencia
    df['tendencia'] = range(len(df))
    
    # Promedios móviles
    df['ma3'] = df['total_casos'].rolling(window=3).mean()
    df['ma6'] = df['total_casos'].rolling(window=6).mean()
    
    # Eliminar filas con valores NaN generados por los lags
    df_clean = df.dropna()
    
    if len(df_clean) < 6:  # Reducido de 10 a 6
        return None, None, None
    
    # Características y variable objetivo
    features = ['mes', 'trimestre', 'mes_sin', 'mes_cos', 'casos_lag1', 'casos_lag3', 
                'tendencia', 'ma3', 'ma6', 'edad_promedio', 'porcentaje_mujeres']
    
    # Solo usar características que existen
    features = [f for f in features if f in df_clean.columns]
    
    X = df_clean[features]
    y = df_clean['total_casos']
    
    return X, y, df_clean

def entrenar_modelo_prediccion(X, y):
    """Entrena el modelo de predicción usando Random Forest."""
    if len(X) < 6:  # Reducido de 10 a 6
        return None, None, None
    
    # Dividir datos (80% entrenamiento, 20% validación)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Normalizar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Entrenar modelo
    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    modelo.fit(X_train_scaled, y_train)
    
    # Validar modelo
    y_pred = modelo.predict(X_val_scaled)
    
    # Métricas
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    metricas = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': np.mean(np.abs((y_val - y_pred) / y_val)) * 100 if len(y_val) > 0 else 0
    }
    
    return modelo, scaler, metricas

def generar_predicciones_futuras(modelo, scaler, df_historicos, n_meses=12):
    """Genera predicciones para los próximos n meses."""
    if modelo is None or df_historicos.empty:
        return pd.DataFrame()
    
    predicciones = []
    df_work = df_historicos.copy()
    
    # Obtener el último año y mes de los datos
    ultimo_anio = df_work['anio_datos'].max()
    ultimo_mes = df_work[df_work['anio_datos'] == ultimo_anio]['mes'].max()
    
    for i in range(n_meses):
        # Calcular siguiente mes/año
        siguiente_mes = ultimo_mes + i + 1
        siguiente_anio = ultimo_anio
        
        if siguiente_mes > 12:
            siguiente_anio += (siguiente_mes - 1) // 12
            siguiente_mes = ((siguiente_mes - 1) % 12) + 1
        
        siguiente_trimestre = (siguiente_mes - 1) // 3 + 1
        
        # Crear características para predicción
        mes_sin = np.sin(2 * np.pi * siguiente_mes / 12)
        mes_cos = np.cos(2 * np.pi * siguiente_mes / 12)
        
        # Variables lag basadas en datos históricos recientes
        if len(df_work) >= 1:
            casos_lag1 = df_work['total_casos'].iloc[-1]
        else:
            casos_lag1 = df_work['total_casos'].mean()
            
        if len(df_work) >= 3:
            casos_lag3 = df_work['total_casos'].iloc[-3]
        else:
            casos_lag3 = df_work['total_casos'].mean()
        
        # Tendencia continuada
        tendencia = len(df_work) + i + 1
        
        # Promedios móviles
        ma3 = df_work['total_casos'].tail(3).mean()
        ma6 = df_work['total_casos'].tail(6).mean()
        
        # Variables demográficas (usar promedios históricos)
        edad_promedio = df_work['edad_promedio'].mean()
        porcentaje_mujeres = df_work['porcentaje_mujeres'].mean()
        
        # Crear vector de características
        features = np.array([[
            siguiente_mes, siguiente_trimestre, mes_sin, mes_cos,
            casos_lag1, casos_lag3, tendencia, ma3, ma6,
            edad_promedio, porcentaje_mujeres
        ]])
        
        # Normalizar y predecir
        features_scaled = scaler.transform(features)
        prediccion = modelo.predict(features_scaled)[0]
        
        # Asegurar que la predicción sea positiva
        prediccion = max(0, int(prediccion))
        
        predicciones.append({
            'anio': siguiente_anio,
            'mes': siguiente_mes,
            'trimestre': siguiente_trimestre,
            'casos_predichos': prediccion,
            'fecha': f"{siguiente_anio}-{siguiente_mes:02d}"
        })
        
        # Agregar predicción a datos de trabajo para próximas iteraciones
        nuevo_registro = {
            'anio_datos': siguiente_anio,
            'mes': siguiente_mes,
            'trimestre': siguiente_trimestre,
            'total_casos': prediccion,
            'edad_promedio': edad_promedio,
            'porcentaje_mujeres': porcentaje_mujeres
        }
        df_work = pd.concat([df_work, pd.DataFrame([nuevo_registro])], ignore_index=True)
    
    return pd.DataFrame(predicciones)

# --- Funciones para calcular KPIs ---
def calculate_kpis(df_anios, df_estados, df_edad):
    """Calcula los KPIs principales."""
    kpis = {}
    
    if not df_anios.empty:
        # KPI 1: Tasa de Mortalidad General
        total_casos = df_anios['total_casos'].sum()
        total_defunciones = df_anios['defunciones'].sum()
        kpis['tasa_mortalidad'] = (total_defunciones / total_casos * 100) if total_casos > 0 else 0
        
        # KPI 2: Tasa de Confirmación
        total_confirmados = df_anios['confirmados'].sum()
        kpis['tasa_confirmacion'] = (total_confirmados / total_casos * 100) if total_casos > 0 else 0
        
        # KPI 3: Tasa de Casos Hemorrágicos
        total_hemorragicos = df_anios['hemorragicos'].sum()
        kpis['tasa_hemorragicos'] = (total_hemorragicos / total_casos * 100) if total_casos > 0 else 0
        
        # KPI 4: Tendencia Anual (variación porcentual)
        if len(df_anios) >= 2:
            casos_actual = df_anios.iloc[-1]['total_casos']
            casos_anterior = df_anios.iloc[-2]['total_casos']
            kpis['tendencia_anual'] = ((casos_actual - casos_anterior) / casos_anterior * 100) if casos_anterior > 0 else 0
        else:
            kpis['tendencia_anual'] = 0
        
        # KPI 5: Distribución por Sexo
        total_hombres = df_anios['hombres'].sum()
        total_mujeres = df_anios['mujeres'].sum()
        total_sexo = total_hombres + total_mujeres
        kpis['porcentaje_mujeres'] = (total_mujeres / total_sexo * 100) if total_sexo > 0 else 0
    
    # KPI 6: Concentración Geográfica (Índice de Gini simplificado)
    if not df_estados.empty:
        casos_estados = df_estados['total_casos'].values
        casos_estados_sorted = np.sort(casos_estados)
        n = len(casos_estados_sorted)
        cumsum = np.cumsum(casos_estados_sorted)
        kpis['concentracion_geografica'] = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if n > 0 else 0
    
    # KPI 7: Grupos de Mayor Riesgo
    if not df_edad.empty:
        grupo_mayor_riesgo = df_edad.loc[df_edad['defunciones'].idxmax(), 'grupo_edad'] if df_edad['defunciones'].max() > 0 else 'N/A'
        kpis['grupo_mayor_riesgo'] = grupo_mayor_riesgo
        
        # Tasa de mortalidad por grupo más afectado
        max_defunciones_idx = df_edad['defunciones'].idxmax()
        casos_grupo_riesgo = df_edad.loc[max_defunciones_idx, 'total_casos']
        defunciones_grupo_riesgo = df_edad.loc[max_defunciones_idx, 'defunciones']
        kpis['mortalidad_grupo_riesgo'] = (defunciones_grupo_riesgo / casos_grupo_riesgo * 100) if casos_grupo_riesgo > 0 else 0
    
    return kpis

def display_kpi_card(title, value, subtitle="", delta=None, delta_color="normal"):
    """Muestra una tarjeta de KPI personalizada."""
    delta_html = ""
    if delta is not None:
        color = "red" if delta_color == "inverse" and delta > 0 else "green" if delta > 0 else "red"
        arrow = "↗️" if delta > 0 else "↘️" if delta < 0 else "➡️"
        delta_html = f'<small style="color: {color};">{arrow} {delta:+.1f}%</small>'
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin-bottom: 1rem;
    ">
        <h4 style="margin: 0; color: #495057; font-size: 0.9rem;">{title}</h4>
        <h2 style="margin: 0.25rem 0; color: #212529;">{value}</h2>
        <p style="margin: 0; color: #6c757d; font-size: 0.8rem;">{subtitle}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# --- Configuración de la página ---
st.set_page_config(
    page_title="Dashboard Dengue México",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Título y descripción ---
st.title("🦟 Dashboard de Casos de Dengue en México")
st.markdown("### Análisis epidemiológico con KPIs clave y predicción a futuro")

# --- Sidebar para navegación ---
st.sidebar.header("📊 Navegación")
page = st.sidebar.selectbox(
    "Seleccionar vista:",
    ["📈 Análisis Actual", "🔮 Predicción a Futuro"],
    help="Selecciona el tipo de análisis que deseas ver"
)

# --- Sidebar para filtros CORREGIDO ---
st.sidebar.header("🔍 Filtros")

# Obtener años disponibles
years_available = get_years_available()
if years_available:
    if page == "🔮 Predicción a Futuro":
        # NUEVA SECCIÓN: Permitir seleccionar más años para entrenar mejor modelo
        st.sidebar.markdown("### 📚 Datos para Entrenar el Modelo")
        st.sidebar.info("💡 Más años = modelo más robusto. Recomendado: 3-5 años")
        
        selected_years = st.sidebar.multiselect(
            "Seleccionar años para entrenar:",
            options=sorted(years_available, reverse=True),  # Ordenar del más reciente al más antiguo
            default=sorted(years_available, reverse=True)[:3],  # Por defecto los 3 más recientes
            help="Selecciona los años que usarás para entrenar el modelo de predicción"
        )
        
        # NUEVA SECCIÓN: Seleccionar año objetivo para predicción
        st.sidebar.markdown("### 🎯 Año Objetivo de Predicción")
        
        if selected_years:
            ultimo_año_datos = max(selected_years)
            años_disponibles_prediccion = list(range(ultimo_año_datos + 1, ultimo_año_datos + 4))  # 3 años futuros
            
            año_objetivo = st.sidebar.selectbox(
                "¿Qué año quieres predecir?",
                options=años_disponibles_prediccion,
                index=0,
                help="Año para el cual generar las predicciones"
            )
            
            # Calcular automáticamente cuántos meses predecir según el año objetivo
            años_diferencia = año_objetivo - ultimo_año_datos
            meses_prediccion = años_diferencia * 12
            
            st.sidebar.success(f"📅 Prediciendo {meses_prediccion} meses hacia {año_objetivo}")
            
            # Opción para ajustar manualmente el período de predicción
            with st.sidebar.expander("⚙️ Configuración Avanzada"):
                meses_prediccion = st.slider(
                    "Ajustar meses a predecir:",
                    min_value=6,
                    max_value=36,
                    value=meses_prediccion,
                    step=3,
                    help="Número exacto de meses hacia el futuro"
                )
                
                intervalo_confianza = st.selectbox(
                    "Intervalo de confianza:",
                    options=[80, 90, 95],
                    index=1,
                    help="Nivel de confianza para las predicciones"
                )
                
                incluir_estacionalidad = st.checkbox(
                    "Incluir factores estacionales",
                    value=True,
                    help="Incluir patrones estacionales en el modelo"
                )
        else:
            st.sidebar.warning("⚠️ Selecciona al menos un año para continuar")
            año_objetivo = None
            meses_prediccion = 12
            
    else:
        # Para análisis actual, mantener la funcionalidad original
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

# === PÁGINA DE PREDICCIÓN A FUTURO ===
if page == "🔮 Predicción a Futuro":
    st.header("🔮 Predicción de Casos de Dengue a Futuro")
    
    if len(selected_years) < 1:
        st.warning("⚠️ Se necesita al menos 1 año de datos para generar predicciones.")
        st.info("Por favor selecciona al menos un año en el panel lateral.")
    elif selected_years and año_objetivo:
        # Mostrar información del modelo
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Años de Entrenamiento", 
                len(selected_years),
                help="Número de años usados para entrenar el modelo"
            )
        
        with col2:
            st.metric(
                "Año Objetivo", 
                año_objetivo,
                help="Año para el cual se generarán las predicciones"
            )
        
        with col3:
            st.metric(
                "Meses de Predicción", 
                meses_prediccion,
                help="Número de meses hacia el futuro"
            )
        
        # Validación de calidad de datos
        if len(selected_years) >= 3:
            st.success("✅ Excelente: Suficientes años para un modelo robusto")
        elif len(selected_years) == 2:
            st.warning("⚠️ Aceptable: Se recomienda usar más años para mejor precisión")
        else:
            st.info("📊 Básico: El modelo usará los datos disponibles")
        
        # Obtener datos para predicción
        with st.spinner("🔄 Cargando datos y entrenando modelo..."):
            df_prediccion = get_datos_prediccion_detallados(selected_years, selected_estado_id)
            
            if not df_prediccion.empty:
                # Preparar datos
                X, y, df_clean = preparar_datos_para_prediccion(df_prediccion)
                
                if X is not None and len(X) >= 6:  # Reducir requerimiento mínimo
                    # Entrenar modelo
                    modelo, scaler, metricas = entrenar_modelo_prediccion(X, y)
                    
                    if modelo is not None:
                        # Generar predicciones
                        predicciones = generar_predicciones_futuras(
                            modelo, scaler, df_clean, meses_prediccion
                        )
                        
                        # Filtrar predicciones para el año objetivo específico
                        predicciones_año_objetivo = predicciones[
                            predicciones['anio'] == año_objetivo
                        ]
                        
                        # === NUEVA SECCIÓN: RESUMEN PARA EL AÑO OBJETIVO ===
                        if not predicciones_año_objetivo.empty:
                            st.subheader(f"📊 Resumen para el Año {año_objetivo}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            casos_totales_año = predicciones_año_objetivo['casos_predichos'].sum()
                            casos_promedio_mes = predicciones_año_objetivo['casos_predichos'].mean()
                            mes_pico = predicciones_año_objetivo.loc[
                                predicciones_año_objetivo['casos_predichos'].idxmax(), 'mes'
                            ]
                            meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                           'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
                            mes_pico_nombre = meses_nombres[mes_pico - 1]
                            
                            with col1:
                                st.metric(
                                    f"Casos Totales {año_objetivo}",
                                    f"{casos_totales_año:,.0f}",
                                    help=f"Predicción total de casos para {año_objetivo}"
                                )
                            
                            with col2:
                                st.metric(
                                    "Promedio Mensual",
                                    f"{casos_promedio_mes:,.0f}",
                                    help="Promedio de casos por mes"
                                )
                            
                            with col3:
                                st.metric(
                                    "Mes de Pico",
                                    mes_pico_nombre,
                                    help="Mes con mayor predicción de casos"
                                )
                            
                            with col4:
                                # Comparar con año base
                                if not df_clean.empty:
                                    casos_historicos_promedio = df_clean['total_casos'].mean() * 12
                                    variacion = ((casos_totales_año - casos_historicos_promedio) / casos_historicos_promedio * 100)
                                    st.metric(
                                        "Variación vs Histórico",
                                        f"{variacion:+.1f}%",
                                        help="Comparación con promedio histórico anual"
                                    )
                        
                        # === MÉTRICAS DEL MODELO ===
                        st.subheader("📊 Precisión del Modelo")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "R² Score", 
                                f"{metricas['r2']:.3f}",
                                help="Coeficiente de determinación (0-1, mayor es mejor)"
                            )
                        with col2:
                            st.metric(
                                "Error Medio Absoluto", 
                                f"{int(metricas['mae'])} casos",
                                help="Promedio de error en número de casos"
                            )
                        with col3:
                            st.metric(
                                "RMSE", 
                                f"{int(metricas['rmse'])} casos",
                                help="Raíz del error cuadrático medio"
                            )
                        with col4:
                            st.metric(
                                "MAPE", 
                                f"{metricas['mape']:.1f}%",
                                help="Error porcentual absoluto medio"
                            )
                        
                        # Interpretación de la precisión mejorada
                        if metricas['r2'] >= 0.7:
                            precision_text = "🟢 **Excelente** - El modelo tiene alta precisión"
                            precision_color = "success"
                        elif metricas['r2'] >= 0.5:
                            precision_text = "🟡 **Buena** - El modelo tiene precisión aceptable"
                            precision_color = "warning"
                        elif metricas['r2'] >= 0.3:
                            precision_text = "🟠 **Moderada** - Usar predicciones con cautela"
                            precision_color = "warning"
                        else:
                            precision_text = "🔴 **Baja** - Predicciones poco confiables"
                            precision_color = "error"
                        
                        if precision_color == "success":
                            st.success(f"**Precisión del modelo**: {precision_text}")
                        elif precision_color == "warning":
                            st.warning(f"**Precisión del modelo**: {precision_text}")
                        else:
                            st.error(f"**Precisión del modelo**: {precision_text}")
                        
                        # === VISUALIZACIÓN DE PREDICCIONES ===
                        st.subheader("📈 Predicciones vs Datos Históricos")
                        
                        # Preparar datos para visualización
                        df_viz = df_clean.copy()
                        df_viz['tipo'] = 'Histórico'
                        df_viz['fecha_viz'] = pd.to_datetime(
                            df_viz['anio_datos'].astype(str) + '-' + 
                            df_viz['mes'].astype(str) + '-01'
                        )
                        
                        # Agregar predicciones
                        df_pred_viz = predicciones.copy()
                        df_pred_viz['total_casos'] = df_pred_viz['casos_predichos']
                        df_pred_viz['tipo'] = 'Predicción'
                        df_pred_viz['fecha_viz'] = pd.to_datetime(
                            df_pred_viz['anio'].astype(str) + '-' + 
                            df_pred_viz['mes'].astype(str) + '-01'
                        )
                        
                        # Combinar datos
                        df_combined = pd.concat([
                            df_viz[['fecha_viz', 'total_casos', 'tipo']],
                            df_pred_viz[['fecha_viz', 'total_casos', 'tipo']]
                        ])
                        
                        # Gráfico principal
                        fig_prediccion = go.Figure()
                        
                        # Datos históricos
                        df_hist = df_combined[df_combined['tipo'] == 'Histórico']
                        fig_prediccion.add_trace(go.Scatter(
                            x=df_hist['fecha_viz'],
                            y=df_hist['total_casos'],
                            mode='lines+markers',
                            name='Datos Históricos',
                            line=dict(color='blue', width=2),
                            marker=dict(size=6)
                        ))
                        
                        # Predicciones
                        df_pred = df_combined[df_combined['tipo'] == 'Predicción']
                        fig_prediccion.add_trace(go.Scatter(
                            x=df_pred['fecha_viz'],
                            y=df_pred['total_casos'],
                            mode='lines+markers',
                            name='Predicciones',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6, symbol='diamond')
                        ))
                        
                        # Banda de confianza (estimación simple)
                        if not df_pred.empty:
                            error_band = metricas['rmse'] * (1.96 if intervalo_confianza == 95 
                                                           else 1.645 if intervalo_confianza == 90 
                                                           else 1.28)
                            
                            upper_bound = df_pred['total_casos'] + error_band
                            lower_bound = np.maximum(0, df_pred['total_casos'] - error_band)
                            
                            fig_prediccion.add_trace(go.Scatter(
                                x=df_pred['fecha_viz'],
                                y=upper_bound,
                                fill=None,
                                mode='lines',
                                line_color='rgba(255,0,0,0)',
                                showlegend=False
                            ))
                            
                            fig_prediccion.add_trace(go.Scatter(
                                x=df_pred['fecha_viz'],
                                y=lower_bound,
                                fill='tonexty',
                                mode='lines',
                                line_color='rgba(255,0,0,0)',
                                name=f'Intervalo {intervalo_confianza}%',
                                fillcolor='rgba(255,0,0,0.2)'
                            ))
                        
                        fig_prediccion.update_layout(
                            title=f'Predicción de Casos de Dengue - {año_objetivo}',
                            xaxis_title='Fecha',
                            yaxis_title='Número de Casos',
                            hovermode='x unified',
                            height=500,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        
                        st.plotly_chart(fig_prediccion, use_container_width=True)
                        
                        # === TABLA DE PREDICCIONES ===
                        st.subheader("📋 Predicciones Detalladas")
                        
                        # Agregar información adicional a las predicciones
                        df_pred_table = predicciones.copy()
                        df_pred_table['fecha_formatted'] = pd.to_datetime(
                            df_pred_table['anio'].astype(str) + '-' + 
                            df_pred_table['mes'].astype(str) + '-01'
                        ).dt.strftime('%Y-%m')
                        
                        df_pred_table['mes_nombre'] = pd.to_datetime(
                            df_pred_table['anio'].astype(str) + '-' + 
                            df_pred_table['mes'].astype(str) + '-01'
                        ).dt.month_name()
                        
                        # Calcular variación porcentual
                        casos_base = df_clean['total_casos'].mean()
                        df_pred_table['variacion_vs_promedio'] = (
                            (df_pred_table['casos_predichos'] - casos_base) / casos_base * 100
                        ).round(1)
                        
                        # Categorizar riesgo
                        def categorizar_riesgo(casos, percentil_75):
                            if casos >= percentil_75 * 1.5:
                                return "🔴 Alto"
                            elif casos >= percentil_75:
                                return "🟡 Moderado"
                            else:
                                return "🟢 Bajo"
                        
                        percentil_75 = np.percentile(df_clean['total_casos'], 75)
                        df_pred_table['nivel_riesgo'] = df_pred_table['casos_predichos'].apply(
                            lambda x: categorizar_riesgo(x, percentil_75)
                        )
                        
                        # Mostrar tabla
                        df_display = df_pred_table[[
                            'fecha_formatted', 'mes_nombre', 'casos_predichos', 
                            'variacion_vs_promedio', 'nivel_riesgo'
                        ]].copy()
                        
                        df_display.columns = [
                            'Período', 'Mes', 'Casos Predichos', 
                            'Var. vs Promedio (%)', 'Nivel de Riesgo'
                        ]
                        
                        st.dataframe(df_display, use_container_width=True)
                        
                        # === ANÁLISIS ESTACIONAL ===
                        st.subheader("🌡️ Análisis Estacional")
                        
                        # Agregar datos de estacionalidad
                        df_estacional = predicciones.copy()
                        df_estacional['estacion'] = df_estacional['mes'].apply(
                            lambda x: 'Invierno' if x in [12, 1, 2] 
                                     else 'Primavera' if x in [3, 4, 5]
                                     else 'Verano' if x in [6, 7, 8]
                                     else 'Otoño'
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gráfico por estaciones
                            casos_por_estacion = df_estacional.groupby('estacion')['casos_predichos'].mean().reset_index()
                            fig_estaciones = px.bar(
                                casos_por_estacion,
                                x='estacion',
                                y='casos_predichos',
                                title='Casos Predichos Promedio por Estación',
                                color='casos_predichos',
                                color_continuous_scale='Reds'
                            )
                            st.plotly_chart(fig_estaciones, use_container_width=True)
                        
                        with col2:
                            # Gráfico por meses
                            casos_por_mes = df_estacional.groupby('mes')['casos_predichos'].mean().reset_index()
                            casos_por_mes['mes_nombre'] = casos_por_mes['mes'].apply(
                                lambda x: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                          'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][x-1]
                            )
                            fig_meses = px.line(
                                casos_por_mes,
                                x='mes_nombre',
                                y='casos_predichos',
                                title='Patrón Mensual Predicho',
                                markers=True
                            )
                            st.plotly_chart(fig_meses, use_container_width=True)
                        
                        # === RECOMENDACIONES BASADAS EN PREDICCIONES ===
                        st.subheader("💡 Recomendaciones")
                        
                        # Identificar meses de mayor riesgo
                        meses_alto_riesgo = df_pred_table[
                            df_pred_table['nivel_riesgo'] == "🔴 Alto"
                        ]['mes_nombre'].unique()
                        
                        meses_pico = df_pred_table.loc[
                            df_pred_table['casos_predichos'].idxmax(), 'mes_nombre'
                        ]
                        
                        tendencia_general = "creciente" if df_pred_table['casos_predichos'].iloc[-1] > df_pred_table['casos_predichos'].iloc[0] else "decreciente"
                        
                        recomendaciones = []
                        
                        if len(meses_alto_riesgo) > 0:
                            recomendaciones.append(
                                f"🚨 **Alerta temprana**: Se predicen niveles altos de casos en {', '.join(meses_alto_riesgo)}. "
                                "Recomendar intensificar medidas preventivas."
                            )
                        
                        recomendaciones.append(
                            f"📊 **Pico estacional**: El mayor número de casos se predice para {meses_pico}. "
                            "Planificar recursos médicos adicionales."
                        )
                        
                        recomendaciones.append(
                            f"📈 **Tendencia general**: La predicción muestra una tendencia {tendencia_general} "
                            f"para {año_objetivo}."
                        )
                        
                        if metricas['r2'] < 0.6:
                            recomendaciones.append(
                                "⚠️ **Cautela**: La precisión del modelo es moderada. "
                                "Complementar con vigilancia epidemiológica continua."
                            )
                        
                        recomendaciones.append(
                            "🏥 **Preparación hospitalaria**: Ajustar capacidad según picos predichos. "
                            "Considerar campañas de prevención antes de los meses de alto riesgo."
                        )
                        
                        for rec in recomendaciones:
                            st.info(rec)
                        
                        # === FACTORES DEL MODELO ===
                        with st.expander("🔬 Detalles Técnicos del Modelo"):
                            st.markdown("""
                            **Variables utilizadas en el modelo:**
                            - Estacionalidad (mes, trimestre, componentes sinusoidales)
                            - Tendencia temporal
                            - Valores históricos (lags de 1, 3 y 12 meses)
                            - Promedios móviles (3 y 6 meses)
                            - Variables demográficas (edad promedio, distribución por sexo)
                            
                            **Algoritmo:** Random Forest Regressor
                            - Robusto ante outliers
                            - Captura relaciones no lineales
                            - Maneja automáticamente la importancia de variables
                            
                            **Validación:** División temporal 80/20
                            """)
                            
                            # Importancia de características
                            if hasattr(modelo, 'feature_importances_'):
                                feature_names = ['Mes', 'Trimestre', 'Mes_Sin', 'Mes_Cos', 
                                               'Casos_Lag1', 'Casos_Lag3', 'Tendencia', 
                                               'MA3', 'MA6', 'Edad_Prom', 'Porc_Mujeres']
                                
                                importance_df = pd.DataFrame({
                                    'Variable': feature_names[:len(modelo.feature_importances_)],
                                    'Importancia': modelo.feature_importances_
                                }).sort_values('Importancia', ascending=False)
                                
                                fig_importance = px.bar(
                                    importance_df.head(8),
                                    x='Importancia',
                                    y='Variable',
                                    orientation='h',
                                    title='Importancia de Variables en el Modelo'
                                )
                                st.plotly_chart(fig_importance, use_container_width=True)
                    
                    else:
                        st.error("❌ No se pudo entrenar el modelo. Verificar calidad de los datos.")
                
                else:
                    st.error("❌ Datos insuficientes para entrenar el modelo. Se necesitan al menos 6 observaciones.")
                    st.info("💡 Intenta seleccionar un rango de años más amplio o verificar la disponibilidad de datos.")
            
            else:
                st.error("❌ No se encontraron datos suficientes para generar predicciones.")
                st.info("💡 Verifica que los años seleccionados contengan datos de casos de dengue.")

# === PÁGINA DE ANÁLISIS ACTUAL ===
else:  # page == "📈 Análisis Actual"
    if selected_years:
        # Obtener datos para KPIs
        df_anios = get_casos_por_anio(selected_years)
        df_estados_casos = get_casos_por_estado(selected_years)
        df_edad = get_casos_por_edad_grupo(selected_years)
        
        # Calcular KPIs
        kpis = calculate_kpis(df_anios, df_estados_casos, df_edad)
        
        # === SECCIÓN DE KPIs ===
        st.header("📊 Indicadores Clave de Desempeño (KPIs)")
        
        # Fila 1 de KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_kpi_card(
                "Tasa de Mortalidad",
                f"{kpis.get('tasa_mortalidad', 0):.2f}%",
                "Defunciones por cada 100 casos",
                delta=None
            )
        
        with col2:
            tasa_conf = kpis.get('tasa_confirmacion', 0)
            if tasa_conf >= 50:
                status_conf = "Óptimo"
            elif tasa_conf >= 40:
                status_conf = "Aceptable"
            else:
                status_conf = "Bajo"
            
            display_kpi_card(
                "Tasa de Confirmación",
                f"{tasa_conf:.1f}%",
                f"Casos confirmados por laboratorio ({status_conf})",
                delta=None
            )
        
        with col3:
            display_kpi_card(
                "Casos Hemorrágicos",
                f"{kpis.get('tasa_hemorragicos', 0):.1f}%",
                "Porcentaje de dengue hemorrágico",
                delta=None
            )
        
        with col4:
            display_kpi_card(
                "Tendencia Anual",
                f"{kpis.get('tendencia_anual', 0):+.1f}%",
                "Variación respecto año anterior",
                delta=kpis.get('tendencia_anual', 0),
                delta_color="inverse"
            )
        
        # Fila 2 de KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_kpi_card(
                "Distribución por Sexo",
                f"{kpis.get('porcentaje_mujeres', 0):.1f}%",
                "Porcentaje de casos en mujeres",
                delta=None
            )
        
        with col2:
            display_kpi_card(
                "Concentración Geográfica",
                f"{kpis.get('concentracion_geografica', 0):.3f}",
                "Índice de distribución (0=uniforme, 1=concentrado)",
                delta=None
            )
        
        with col3:
            display_kpi_card(
                "Grupo de Mayor Riesgo",
                f"{kpis.get('grupo_mayor_riesgo', 'N/A')}",
                "Grupo etario con más defunciones",
                delta=None
            )
        
        with col4:
            display_kpi_card(
                "Mortalidad Grupo Riesgo",
                f"{kpis.get('mortalidad_grupo_riesgo', 0):.2f}%",
                "Tasa de mortalidad del grupo más afectado",
                delta=None
            )
        
        # === MÉTRICAS GENERALES ===
        st.header("📈 Métricas Generales")
        
        if not df_anios.empty:
            total_casos = df_anios['total_casos'].sum()
            total_defunciones = df_anios['defunciones'].sum()
            total_confirmados = df_anios['confirmados'].sum()
            total_hemorragicos = df_anios['hemorragicos'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Casos", f"{total_casos:,}")
            with col2:
                st.metric("Casos Confirmados", f"{total_confirmados:,}")
            with col3:
                st.metric("Defunciones", f"{total_defunciones:,}")
            with col4:
                st.metric("Casos Hemorrágicos", f"{total_hemorragicos:,}")
        
        # === ANÁLISIS POR GRUPOS DE EDAD ===
        st.header("👥 Análisis por Grupos de Edad")
        
        if not df_edad.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_edad_casos = px.bar(
                    df_edad,
                    x='grupo_edad',
                    y='total_casos',
                    title='Casos por Grupo de Edad',
                    labels={'total_casos': 'Número de Casos', 'grupo_edad': 'Grupo de Edad'},
                    color='total_casos',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_edad_casos, use_container_width=True)
            
            with col2:
                # Calcular tasa de mortalidad por grupo
                df_edad['tasa_mortalidad'] = (df_edad['defunciones'] / df_edad['total_casos'] * 100).fillna(0)
                
                fig_edad_mortalidad = px.bar(
                    df_edad,
                    x='grupo_edad',
                    y='tasa_mortalidad',
                    title='Tasa de Mortalidad por Grupo de Edad',
                    labels={'tasa_mortalidad': 'Tasa de Mortalidad (%)', 'grupo_edad': 'Grupo de Edad'},
                    color='tasa_mortalidad',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_edad_mortalidad, use_container_width=True)
        
        # === TENDENCIA POR AÑOS ===
        st.header("📈 Tendencia Temporal")
        
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
                title='Evolución de Casos de Dengue por Año',
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
        
        # === ANÁLISIS POR ESTADOS ===
        st.header("🗺️ Análisis por Estados")
        
        if not df_estados_casos.empty:
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
        
        # === ANÁLISIS POR MUNICIPIOS ===
        st.header("🏘️ Análisis por Municipios")
        
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

        # === ALERTAS Y RECOMENDACIONES ===
        st.header("🚨 Alertas y Recomendaciones")

        # Crear alertas basadas en KPIs
        alertas = []

        if kpis.get('tasa_mortalidad', 0) > 2:
            alertas.append({
                'tipo': 'error',
                'mensaje': f"⚠️ CRÍTICO: Tasa de mortalidad ({kpis.get('tasa_mortalidad', 0):.2f}%) supera el umbral recomendado (2%)",
                'recomendacion': "Revisar protocolos de atención médica y acceso a servicios de salud"
            })

        if kpis.get('tasa_confirmacion', 0) < 40:
            alertas.append({
                'tipo': 'warning',
                'mensaje': f"⚠️ ATENCIÓN: Tasa de confirmación ({kpis.get('tasa_confirmacion', 0):.1f}%) está por debajo del nivel aceptable (40%)",
                'recomendacion': "Mejorar capacidad diagnóstica y acceso a pruebas de laboratorio"
            })

        if kpis.get('tasa_hemorragicos', 0) > 15:
            alertas.append({
                'tipo': 'warning',
                'mensaje': f"⚠️ ATENCIÓN: Alta proporción de casos hemorrágicos ({kpis.get('tasa_hemorragicos', 0):.1f}%)",
                'recomendacion': "Reforzar vigilancia epidemiológica y manejo clínico temprano"
            })

        if kpis.get('tendencia_anual', 0) > 20:
            alertas.append({
                'tipo': 'warning',
                'mensaje': f"📈 TENDENCIA: Incremento significativo de casos ({kpis.get('tendencia_anual', 0):+.1f}%)",
                'recomendacion': "Intensificar medidas de prevención y control vectorial"
            })

        if kpis.get('concentracion_geografica', 0) > 0.7:
            alertas.append({
                'tipo': 'info',
                'mensaje': f"📍 DISTRIBUCIÓN: Alta concentración geográfica de casos (índice: {kpis.get('concentracion_geografica', 0):.3f})",
                'recomendacion': "Focalizar intervenciones en áreas de mayor transmisión"
            })

        # Mostrar alertas
        if alertas:
            for alerta in alertas:
                if alerta['tipo'] == 'error':
                    st.error(alerta['mensaje'])
                    st.info(f"💡 **Recomendación**: {alerta['recomendacion']}")
                elif alerta['tipo'] == 'warning':
                    st.warning(alerta['mensaje'])
                    st.info(f"💡 **Recomendación**: {alerta['recomendacion']}")
                else:
                    st.info(alerta['mensaje'])
                    st.info(f"💡 **Recomendación**: {alerta['recomendacion']}")
        else:
            st.success("✅ Todos los indicadores están dentro de los parámetros aceptables")

        # === RESUMEN EJECUTIVO ===
        st.header("📄 Resumen Ejecutivo")

        if not df_anios.empty:
            total_casos = df_anios['total_casos'].sum()
            total_defunciones = df_anios['defunciones'].sum()
            
            resumen = f"""
            **Período analizado**: {min(selected_years)} - {max(selected_years)}
            
            **Panorama General**:
            - Se registraron **{total_casos:,}** casos de dengue en total
            - La tasa de mortalidad general es de **{kpis.get('tasa_mortalidad', 0):.2f}%**
            - El **{kpis.get('tasa_confirmacion', 0):.1f}%** de casos fueron confirmados por laboratorio
            
            **Grupos Vulnerables**:
            - El grupo de edad de mayor riesgo es: **{kpis.get('grupo_mayor_riesgo', 'N/A')}**
            - Las mujeres representan el **{kpis.get('porcentaje_mujeres', 0):.1f}%** de los casos
            
            **Distribución Geográfica**:
            - Los casos muestran un patrón de concentración geográfica (índice: **{kpis.get('concentracion_geografica', 0):.3f}**)
            - Los estados más afectados requieren atención prioritaria
            
            **Tendencia**:
            - La variación anual muestra un **{kpis.get('tendencia_anual', 0):+.1f}%** respecto al período anterior
            """
            
            st.markdown(resumen)

    else:
        st.warning("Por favor, selecciona al menos un año para mostrar los datos.")

# --- Información adicional en sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Sobre este Dashboard**")
st.sidebar.markdown("Dashboard de análisis epidemiológico de dengue en México con capacidades de predicción usando Machine Learning.")

if page == "🔮 Predicción a Futuro":
    with st.sidebar.expander("🤖 Sobre el Modelo de Predicción"):
        st.markdown("""
        **Algoritmo**: Random Forest Regressor
        
        **Variables utilizadas**:
        - Estacionalidad (mes, trimestre)
        - Tendencia temporal
        - Valores históricos (lags)
        - Promedios móviles
        - Variables demográficas
        
        **Recomendaciones**:
        - Usar 3-5 años para mejor entrenamiento
        - Verificar métricas de precisión (R² > 0.5)
        - Complementar con vigilancia epidemiológica
        
        **Años objetivo disponibles**:
        - 2026, 2027, 2028
        - Selección automática de período
        """)
else:
    with st.sidebar.expander("📋 Explicación de KPIs"):
        st.markdown("""
        **Tasa de Mortalidad**: Porcentaje de casos que resultan en defunción. Meta: <2%
        
        **Tasa de Confirmación**: Porcentaje de casos confirmados por laboratorio. Meta: >70%
        
        **Casos Hemorrágicos**: Porcentaje de dengue hemorrágico (más grave). Meta: <15%
        
        **Tendencia Anual**: Variación porcentual vs año anterior. Verde=reducción, Rojo=aumento
        
        **Concentración Geográfica**: Medida de dispersión geográfica (0=uniforme, 1=muy concentrado)
        
        **Grupo de Mayor Riesgo**: Grupo etario con mayor número de defunciones
        """)

st.sidebar.markdown("**🔄 Actualización:** Los datos se actualizan automáticamente cada 30 minutos.")

# === FOOTER CON INFORMACIÓN TÉCNICA ===
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    Dashboard desarrollado para análisis epidemiológico de dengue en México<br>
    Incluye capacidades de predicción usando Random Forest y análisis temporal<br>
    Datos actualizados automáticamente desde la base de datos epidemiológica<br>
    Selección flexible de años • Predicción por año específico • Análisis estacional avanzado<br>
</div>
""", unsafe_allow_html=True)