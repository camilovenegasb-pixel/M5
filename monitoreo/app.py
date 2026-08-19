import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from model_monitoring import evaluate_feature_drift

st.set_page_config(page_title="Monitoreo Data Drift - Créditos", layout="wide")

st.title("🛡️ Dashboard de Monitoreo de Data Drift")
st.markdown("Evaluación estadística sobre la base de datos crediticia.")

# --- Carga directa de la Base de Datos Excel ---
@st.cache_data
def load_data():
    df = pd.read_excel("Base_de_datos.xlsx")
    df['fecha_prestamo'] = pd.to_datetime(df['fecha_prestamo'])
    
    # Dividimos temporalmente:
    # Baseline: Hasta Junio 2025
    # Actual: Julio 2025 en adelante
    cutoff_date = pd.Timestamp("2025-07-01")
    ref_df = df[df['fecha_prestamo'] < cutoff_date].copy()
    cur_df = df[df['fecha_prestamo'] >= cutoff_date].copy()
    
    return df, ref_df, cur_df

try:
    df_full, ref_df, cur_df = load_data()

    num_cols = ['capital_prestado', 'salario_cliente', 'cuota_pactada', 'puntaje_datacredito', 'edad_cliente', 'total_otros_prestamos']
    cat_cols = ['tipo_laboral', 'tendencia_ingresos', 'tipo_credito']

    # --- Filtros en Barra Lateral ---
    st.sidebar.header("⚙️ Configuración del Análisis")
    st.sidebar.info(f"**Baseline (Entrenamiento):** {len(ref_df):,} registros")
    st.sidebar.info(f"**Actual (Producción):** {len(cur_df):,} registros")

    # --- Tabs de la Aplicación ---
    tab1, tab2, tab3 = st.tabs(["📊 Visualización de Métricas", "📈 Análisis Temporal", "💡 Recomendaciones"])

    with tab1:
        st.subheader("1. Estado de Métricas y Monitoreo de Variables")
        
        drift_df = evaluate_feature_drift(ref_df, cur_df, num_cols, cat_cols)
        
        total_vars = len(drift_df)
        drifted_vars = sum(drift_df['Estado'] == "⚠️ Drift")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Variables Analizadas", total_vars)
        col2.metric("Variables con Drift", drifted_vars, delta_color="inverse")
        col3.metric("Estado General del Modelo", "🔴 ALERTA" if drifted_vars > 0 else "🟢 ESTABLE")
        
        st.markdown("---")
        st.write("### Tabla de Métricas de Drift por Variable")
        st.dataframe(drift_df, use_container_width=True)
        
        st.markdown("---")
        st.write("### Comparación de Distribución: Histórico vs Actual")
        selected_var = st.selectbox("Seleccione la variable a inspeccionar:", num_cols + cat_cols)
        
        if selected_var in num_cols:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ref_df[selected_var].dropna(), name="Histórico (< Jul 2025)", opacity=0.6, marker_color='#1f77b4'))
            fig.add_trace(go.Histogram(x=cur_df[selected_var].dropna(), name="Actual (≥ Jul 2025)", opacity=0.6, marker_color='#ff7f0e'))
            fig.update_layout(barmode='overlay', title_text=f'Distribución de {selected_var}')
            st.plotly_chart(fig, use_container_width=True)
        else:
            ref_c = ref_df[selected_var].astype(str).value_counts(normalize=True).reset_index()
            ref_c.columns = [selected_var, 'proportion']
            ref_c['Periodo'] = 'Histórico'
            
            cur_c = cur_df[selected_var].astype(str).value_counts(normalize=True).reset_index()
            cur_c.columns = [selected_var, 'proportion']
            cur_c['Periodo'] = 'Actual'
            
            comb = pd.concat([ref_c, cur_c])
            fig = px.bar(comb, x=selected_var, y='proportion', color='Periodo', barmode='group', title=f'Proporción de Categorías: {selected_var}')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("2. Evolución Temporal del Drift (PSI Mensual)")
        
        df_full['año_mes'] = df_full['fecha_prestamo'].dt.to_period('M').astype(str)
        meses = sorted(df_full['año_mes'].unique())
        
        var_temporal = st.selectbox("Seleccione Variable Numérica para Serie Temporal:", num_cols)
        
        psi_list = []
        for mes in meses:
            sub_df = df_full[df_full['año_mes'] == mes]
            if len(sub_df) >= 30: # Asegurar suficientes datos por mes
                res = evaluate_feature_drift(ref_df, sub_df, [var_temporal], [])
                val_str = res['Métrica/P-Value'].iloc[0]
                if "PSI=" in val_str:
                    psi_num = float(val_str.split("PSI=")[1])
                    psi_list.append({"Mes": mes, "PSI": psi_num})
        
        if psi_list:
            time_df = pd.DataFrame(psi_list)
            fig_time = px.line(time_df, x="Mes", y="PSI", markers=True, title=f"Evolución del PSI por Mes - {var_temporal}")
            fig_time.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Alerta Moderada (0.1)")
            fig_time.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Alerta Crítica (0.2)")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No hay suficientes datos por periodo para construir la serie temporal.")

    with tab3:
        st.subheader("3. Recomendaciones y Diagnóstico CI/CD")
        if drifted_vars > 0:
            st.error(f"🚨 **Alerta Activa:** Se han identificado {drifted_vars} variable(s) con alteraciones estadísticas severas.")
            st.warning("**Acciones Correctivas Recomendadas:**")
            st.markdown("""
            * **Re-entrenamiento del Modelo:** Iniciar ejecuciones en el pipeline para re-entrenar con la ventana de datos de 2025-2026.
            * **Ajuste de Variables:** Evaluar si las variables con mayor Drift requieren nuevas transformaciones o recalibración de límites.
            * **Integración CI/CD:** Desplegar una prueba A/B o *Shadow Deployment* del nuevo modelo antes de pasarlo a producción total.
            """)
        else:
            st.success("✅ **Estado Óptimo:** La población actual no muestra desviaciones estadísticamente significativas.")

except Exception as e:
    st.error(f"Ocurrió un error al cargar la aplicación: {e}")