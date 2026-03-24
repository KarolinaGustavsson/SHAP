import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    columns_of_interest = ["sampleID", "TC", "TG", "fS_Gluk", "S_Krea", "S_Hapt",
                           "S_FAMN", "S_Urat", "S_Alb", "S_Alp", "fS_Jaern","fS_TIBC","S_LD","S_Ca","S_Urea", "S_P", "Fe_maet","S_K", "age", "status"]
    df = df[columns_of_interest].dropna()
    ids = df['sampleID']
    status = df['status']
    ages = df['age'].values
    df_normalized = df.drop(['sampleID', 'status'], axis=1)
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(df_normalized)
    df_normalized = pd.DataFrame(x_normalized, columns=df_normalized.columns)
    return df_normalized, scaler, ids, status, ages

def load_data_with_bioage(path):
    """Load data including biological age measures (kdm_advance, phenoage_advance)"""
    df = pd.read_csv(path)
    columns_of_interest = ["sampleID", "TC", "TG", "fS_Gluk", "S_Krea", "S_Hapt",
                           "S_FAMN", "S_Urat", "S_Alb", "S_Alp", "fS_Jaern","fS_TIBC","S_LD","S_Ca","S_Urea", "S_P", "Fe_maet","S_K", 
                           "age", "status", "kdm_advance", "phenoage_advance"]
    df = df[columns_of_interest].dropna()
    ids = df['sampleID']
    status = df['status']
    ages = df['age'].values
    kdm_advance = df['kdm_advance'].values
    phenoage_advance = df['phenoage_advance'].values
    
    df_normalized = df.drop(['sampleID', 'status', 'kdm_advance', 'phenoage_advance'], axis=1)
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(df_normalized)
    df_normalized = pd.DataFrame(x_normalized, columns=df_normalized.columns)
    
    return df_normalized, scaler, ids, status, ages, kdm_advance, phenoage_advance
