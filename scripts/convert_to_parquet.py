from pathlib import Path
import pandas as pd

SRC = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
DST = Path("data/processed/criteo_uplift_v2.1.parquet")
DST.parent.mkdir(parents=True, exist_ok=True)

FEATURES = [f"f{i}" for i in range(12)]
LABELS = ["treatment", "conversion", "visit", "exposure"]

# 全部先按 float32 读（.gz 会自动解压）；标签列验证后再转 int8
df = pd.read_csv(SRC, dtype={c: "float32" for c in FEATURES + LABELS})

for c in LABELS:
    assert set(df[c].unique()) <= {0, 1}, f"{c} 含有非 0/1 值"
    df[c] = df[c].astype("int8")

df.to_parquet(DST, engine="pyarrow", compression="zstd", index=False)
print(f"完成：{df.shape}，写入 {DST}")