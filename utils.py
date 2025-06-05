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
