import pandas as pd

def load_scoreboard(path="data/final_scoreboard.csv"):
    df = pd.read_csv(path)
    df["domain_norm"] = df["domain"].str.lower().str.strip()
    return df
