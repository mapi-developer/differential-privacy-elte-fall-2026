import polars as pl

df = pl.read_parquet("data/processed/cohort_raw.parquet")
print("Schema:")
print(df.schema)
print("\nShape:", df.shape)
print("\nTarget Class Distribution (hospital_expire_flag):")
print(df["hospital_expire_flag"].value_counts())
print("\nContinuous Feature Quantiles (prior to clamping):")
print(df.select(["anchor_age", "los_hospital_days", "num_diagnoses", "num_procedures"]).describe())