from pathlib import Path
import duckdb
import polars as pl


class MIMICCohortExtractor:
    def __init__(self, raw_hosp_dir: Path):
        self.raw_hosp_dir = Path(raw_hosp_dir)
        self._validate_sources()

    def _validate_sources(self) -> None:
        """Verify essential compressed MIMIC-IV CSVs exist."""
        required_files = [
            "admissions.csv.gz",
            "patients.csv.gz",
            "diagnoses_icd.csv.gz",
            "procedures_icd.csv.gz",
        ]
        missing = [f for f in required_files if not (self.raw_hosp_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required MIMIC-IV files in {self.raw_hosp_dir}: {missing}"
            )

    def extract_admission_cohort(self) -> pl.DataFrame:
        """
        Executes out-of-core SQL extraction over compressed archives:
        - Filters adult patients (anchor_age >= 18).
        - Computes hospital length of stay in fractional days.
        - Aggregates clinical event counts (diagnoses, procedures) per stay.
        """
        con = duckdb.connect()

        # DuckDB natively reads and joins .csv.gz via streaming
        query = f"""
        WITH admission_base AS (
            SELECT 
                adm.subject_id,
                adm.hadm_id,
                adm.admission_type,
                adm.admission_location,
                adm.insurance,
                adm.race,
                adm.hospital_expire_flag,
                -- Derive continuous length of stay in days: (dischtime - admittime)
                date_diff('second', CAST(adm.admittime AS TIMESTAMP), CAST(adm.dischtime AS TIMESTAMP)) / 86400.0 AS los_hospital_days,
                p.gender,
                p.anchor_age
            FROM read_csv_auto('{self.raw_hosp_dir}/admissions.csv.gz') adm
            JOIN read_csv_auto('{self.raw_hosp_dir}/patients.csv.gz') p
              ON adm.subject_id = p.subject_id
            WHERE p.anchor_age >= 18
        ),
        diag_counts AS (
            SELECT hadm_id, COUNT(*) AS num_diagnoses
            FROM read_csv_auto('{self.raw_hosp_dir}/diagnoses_icd.csv.gz')
            GROUP BY hadm_id
        ),
        proc_counts AS (
            SELECT hadm_id, COUNT(*) AS num_procedures
            FROM read_csv_auto('{self.raw_hosp_dir}/procedures_icd.csv.gz')
            GROUP BY hadm_id
        )
        SELECT 
            b.subject_id,
            b.hadm_id,
            b.anchor_age,
            b.gender,
            b.admission_type,
            b.admission_location,
            b.insurance,
            b.race,
            b.los_hospital_days,
            COALESCE(d.num_diagnoses, 0) AS num_diagnoses,
            COALESCE(pr.num_procedures, 0) AS num_procedures,
            b.hospital_expire_flag
        FROM admission_base b
        LEFT JOIN diag_counts d ON b.hadm_id = d.hadm_id
        LEFT JOIN proc_counts pr ON b.hadm_id = pr.hadm_id
        """
        # Stream query execution through Arrow into Polars
        arrow_table = con.execute(query).fetch_arrow_table()
        df = pl.from_arrow(arrow_table)

        # Standardize missing strings and types
        df = df.with_columns(
            pl.col("anchor_age").cast(pl.Float64),
            pl.col("los_hospital_days").cast(pl.Float64),
            pl.col("num_diagnoses").cast(pl.Float64),
            pl.col("num_procedures").cast(pl.Float64),
            pl.col("hospital_expire_flag").cast(pl.Int32),
            pl.col(["gender", "admission_type", "admission_location", "insurance", "race"])
            .cast(pl.Utf8)
            .fill_null("UNKNOWN")
        )
        return df

    def run(self, output_parquet_path: Path) -> Path:
        output_parquet_path = Path(output_parquet_path)
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Extracting cohort from {self.raw_hosp_dir}...")
        df = self.extract_admission_cohort()
        
        print(f"Extracted {len(df):,} admissions. Saving to {output_parquet_path}...")
        df.write_parquet(output_parquet_path, compression="zstd")
        print("Extraction complete.")
        return output_parquet_path


if __name__ == "__main__":
    # Point directly to your MIMIC-IV 'hosp' directory
    RAW_DIR = Path("data/raw/hosp")
    OUTPUT_FILE = Path("data/processed/cohort_raw.parquet")

    extractor = MIMICCohortExtractor(raw_hosp_dir=RAW_DIR)
    extractor.run(output_parquet_path=OUTPUT_FILE)