from pathlib import Path
import yaml
from google.cloud import bigquery
import polars as pl


class CMSMedicareExtractor:
    def __init__(self, config_path: str = "config/cms_medicare.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.client = bigquery.Client(project=self.cfg["gcp"]["project_id"])
        self.source_table = self.cfg["gcp"]["source_table"]

    def build_query(self) -> str:
        return f"""
        SELECT
            provider_id,
            provider_state,
            -- Split '039 - EXTRACRANIAL PROCEDURES W/O CC/MCC' into clean code '039'
            SPLIT(drg_definition, ' - ')[OFFSET(0)] AS drg_code,
            CAST(total_discharges AS FLOAT64) AS total_discharges,
            CAST(average_covered_charges AS FLOAT64) AS average_covered_charges,
            CAST(average_medicare_payments AS FLOAT64) AS average_medicare_payments,
            -- Binary Target for TSTR Benchmark: Whether average total payment exceeds $10,000
            CASE WHEN average_total_payments > 10000.0 THEN 1 ELSE 0 END AS high_payment_flag
        FROM `{self.source_table}`
        WHERE total_discharges IS NOT NULL
          AND average_covered_charges IS NOT NULL
        LIMIT 100000
        """

    def run(self) -> Path:
        output_path = Path(self.cfg["cohort"]["output_parquet_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Querying {self.source_table} on BigQuery...")
        arrow_table = self.client.query(self.build_query()).to_arrow()
        df = pl.from_arrow(arrow_table)

        df = df.with_columns(
            pl.col("provider_id").cast(pl.Utf8),
            pl.col("provider_state").cast(pl.Utf8).fill_null("UNKNOWN"),
            pl.col("drg_code").cast(pl.Utf8).fill_null("UNKNOWN"),
            pl.col("total_discharges").cast(pl.Float64),
            pl.col("average_covered_charges").cast(pl.Float64),
            pl.col("average_medicare_payments").cast(pl.Float64),
            pl.col("high_payment_flag").cast(pl.Int32),
        )

        df.write_parquet(output_path, compression="zstd")
        print(f"Extracted {len(df):,} records successfully to {output_path}")
        return output_path


if __name__ == "__main__":
    extractor = CMSMedicareExtractor()
    extractor.run()