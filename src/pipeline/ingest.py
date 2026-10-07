from pathlib import Path
import polars as pl
import yaml


class CMSDataIngestion:
    def __init__(self, config_path: str = "config/cms_medicare.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.bounds = self.cfg["continuous_bounds"]

    def clamp(self, input_parquet: Path, output_parquet: Path) -> pl.DataFrame:
        df = pl.read_parquet(input_parquet)
        
        transforms = []
        for col_name, (lower, upper) in self.bounds.items():
            if col_name in df.columns:
                transforms.append(
                    pl.col(col_name)
                    .clip(lower_bound=float(lower), upper_bound=float(upper))
                    .alias(col_name)
                )

        df_clamped = df.with_columns(transforms)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_clamped.write_parquet(output_parquet, compression="zstd")
        print(f"Sensitivity clamping complete: {len(df_clamped):,} records saved to {output_parquet}")
        return df_clamped


if __name__ == "__main__":
    ingestor = CMSDataIngestion()
    ingestor.clamp(
        input_parquet=Path("data/processed/cohort_raw.parquet"),
        output_parquet=Path("data/processed/cohort_clamped.parquet"),
    )