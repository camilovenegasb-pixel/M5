import numpy as np
import pandas as pd
from scipy import stats

def calculate_psi(reference: pd.Series, current: pd.Series, num_buckets: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(reference, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5
    
    ref_counts, _ = np.histogram(reference, bins=buckets)
    cur_counts, _ = np.histogram(current, bins=buckets)
    
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)
    
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)
    
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

def calculate_jensen_shannon(reference: pd.Series, current: pd.Series, num_buckets: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) == 0 or len(current) == 0:
        return 0.0
        
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0
        
    bins = np.linspace(min_val, max_val, num_buckets + 1)
    p, _ = np.histogram(reference, bins=bins, density=True)
    q, _ = np.histogram(current, bins=bins, density=True)
    
    p = (p + 1e-8) / np.sum(p + 1e-8)
    q = (q + 1e-8) / np.sum(q + 1e-8)
    
    return float(stats.entropy(p, q, base=2))

def evaluate_feature_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> pd.DataFrame:
    results = []
    
    # Pruebas para variables numéricas
    for col in numerical_cols:
        ref_s = ref_df[col].dropna()
        cur_s = cur_df[col].dropna()
        
        ks_stat, ks_pvalue = stats.ks_2samp(ref_s, cur_s)
        psi_val = calculate_psi(ref_s, cur_s)
        js_div = calculate_jensen_shannon(ref_s, cur_s)
        
        drift_detected = (ks_pvalue < 0.05) or (psi_val > 0.2)
        
        results.append({
            "Variable": col,
            "Tipo": "Numérica",
            "Prueba Aplicada": "KS / PSI",
            "Métrica/P-Value": f"KS p={ks_pvalue:.4f} | PSI={psi_val:.3f}",
            "JS Divergence": f"{js_div:.4f}",
            "Estado": "⚠️ Drift" if drift_detected else "✅ OK",
            "Nivel Riesgo": "Alto" if psi_val > 0.2 else ("Medio" if psi_val > 0.1 else "Bajo")
        })

    # Pruebas para variables categóricas
    for col in categorical_cols:
        ref_cat = ref_df[col].astype(str).fillna("Desconocido")
        cur_cat = cur_df[col].astype(str).fillna("Desconocido")
        
        all_cats = list(set(ref_cat.unique()).union(set(cur_cat.unique())))
        ref_counts = ref_cat.value_counts().reindex(all_cats, fill_value=0)
        cur_counts = cur_cat.value_counts().reindex(all_cats, fill_value=0)
        
        obs = np.array([ref_counts.values, cur_counts.values])
        chi2_stat, chi2_pvalue, _, _ = stats.chi2_contingency(obs)
        
        drift_detected = chi2_pvalue < 0.05
        
        results.append({
            "Variable": col,
            "Tipo": "Categórica",
            "Prueba Aplicada": "Chi-Cuadrado",
            "Métrica/P-Value": f"Chi2 p={chi2_pvalue:.4f}",
            "JS Divergence": "N/A",
            "Estado": "⚠️ Drift" if drift_detected else "✅ OK",
            "Nivel Riesgo": "Alto" if drift_detected else "Bajo"
        })
        
    return pd.DataFrame(results)