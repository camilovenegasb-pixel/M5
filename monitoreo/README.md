# 📈 Dashboard de Monitoreo de Data Drift & Calidad de Modelos

## 🎯 Caso de Negocio
En entornos de producción crediticia, las variaciones en el perfil financiero de los solicitantes o en las políticas macroeconómicas pueden provocar una degradación progresiva en el desempeño del modelo predictivo (*Data Drift*). Este sistema automatiza la detección temprana de sesgos estadísticos entre la población histórica y la población en producción, permitiendo accionar mecanismos de re-entrenamiento antes de afectar la precisión en los pronósticos de impago.

## 🛠️ Pruebas Estadísticas e Métricas Implementadas
1. **Kolmogorov-Smirnov (KS-Test):** Evalúa la hipótesis nula de que dos distribuciones continuas provienen de la misma función de densidad ($p < 0.05$ indica drift).
2. **Population Stability Index (PSI):** Mide el cambio relativo de la población binned ($PSI > 0.1$ alerta moderada, $PSI > 0.2$ alerta crítica).
3. **Divergencia Jensen-Shannon:** Cuantifica la diferencia de entropía entre la distribución de referencia y la actual.
4. **Prueba Chi-Cuadrado ($\chi^2$):** Analiza la homogeneidad de frecuencias en variables categóricas (`tipo_laboral`, `tendencia_ingresos`, `tipo_credito`).

## 📊 Principales Hallazgos del Dataset
* **Población Evaluada:** 8,378 registros históricos (< Julio 2025) vs 2,385 registros actuales (≥ Julio 2025).
* **Estado Global:** **🔴 ALERTA** (6 de 9 variables presentan un Data Drift estadísticamente significativo).
* **Variables con Desviación Crítica:** `total_otros_prestamos` ($PSI = 0.182$) y `tendencia_ingresos` ($p < 0.01$).

## 🚀 Estructura del Repositorio
* `model_monitoring.py`: Módulo backend con el cálculo de métricas estadísticas (KS, PSI, JS, Chi2).
* `app.py`: Interfaz gráfica interactiva desarrollada en Streamlit.
* `Base_de_datos.xlsx`: Dataset crediticio analizado.

## ⚙️ Instrucciones de Ejecución
1. Instalar dependencias:
   ```bash
   pip install streamlit pandas numpy scipy plotly openpyxl