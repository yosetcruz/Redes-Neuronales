import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    
    # ✅ USAR EL NOMBRE CORRECTO EN MINÚSCULAS
    target_col = 'decision'
    
    print(f"✅ Columna objetivo: '{target_col}'")
    print(f"📊 Distribución de clases:\n{df[target_col].value_counts()}")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # Convertir columnas categóricas a numéricas
    cat_cols = X.select_dtypes(include=['object']).columns
    print(f"📝 Columnas categóricas encontradas: {len(cat_cols)}")
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"   - {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Preprocesamiento completado. Features: {X_scaled.shape[1]}")
    
    return X_scaled, y, scaler

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )
    val_relative = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_relative), random_state=random_state, stratify=y_temp
    )
    
    print(f"\n📊 División de datos:")
    print(f"   Entrenamiento: {X_train.shape[0]} ({train_ratio*100:.0f}%)")
    print(f"   Validación: {X_val.shape[0]} ({val_ratio*100:.0f}%)")
    print(f"   Prueba: {X_test.shape[0]} ({test_ratio*100:.0f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
