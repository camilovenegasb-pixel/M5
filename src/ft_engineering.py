import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def get_feature_pipeline(num_cols, cat_cols, ordinal_cols, ordinal_categories):
    """
    Construye el ColumnTransformer según los requerimientos:
    - Numéricas: SimpleImputer(median)
    - Categóricas Nominales: SimpleImputer(most_frequent) + OneHotEncoder
    - Categóricas Ordinales: SimpleImputer(most_frequent) + OrdinalEncoder
    """
    # 1. Pipeline Numérico
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    # 2. Pipeline Categórico Nominal
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Pipeline Categórico Ordinal
    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(categories=ordinal_categories))
    ])
    
    # ColumnTransformer Global
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('ord', ord_pipeline, ordinal_cols)
    ])
    
    return preprocessor

def load_and_prep_data(filepath="../data/raw/base_de_datos.csv", target_col="Pago_atiempo"):
    """
    Carga los datos, separa variables y retorna los conjuntos de train/test y el preprocesador.
    """
    df = pd.read_csv(filepath)
    
    # Unificar categorías inválidas en nulos antes de imputar
    categorias_validas = ['Estable', 'Creciente', 'Decreciente']
    if 'tendencia_ingresos' in df.columns:
        df['tendencia_ingresos'] = df['tendencia_ingresos'].apply(
            lambda x: x if x in categorias_validas else np.nan
        )

    # Definición de columnas
    num_cols = ['capital_prestado', 'edad_cliente', 'puntaje_datacredito']
    cat_cols = ['tipo_credito', 'tipo_laboral']
    ordinal_cols = ['tendencia_ingresos']
    ordinal_cats = [['Decreciente', 'Estable', 'Creciente']]
    
    X = df[num_cols + cat_cols + ordinal_cols]
    y = df[target_col]
    
    # División Train/Test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = get_feature_pipeline(num_cols, cat_cols, ordinal_cols, ordinal_cats)
    
    return X_train, X_test, y_train, y_test, preprocessor