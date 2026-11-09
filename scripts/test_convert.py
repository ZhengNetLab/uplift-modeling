import pandas as pd

df = pd.read_parquet("data/processed/criteo_uplift_v2.1.parquet")

print(df.shape)                       # 期望 (13979592, 16)
print(df.dtypes)
print(df[["treatment", "visit", "conversion", "exposure"]].mean())

# Day 5 IV 分析的前提：对照组的 exposure 是否全为 0
print(df.groupby("treatment")["exposure"].agg(["mean", "max"]))