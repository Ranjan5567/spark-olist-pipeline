import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from delta import configure_spark_with_delta_pip
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, date_format

def create_spark_session(app_name="BronzeIngestion") -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    
    spark = configure_spark_with_delta_pip(
        builder, extra_packages=["io.delta:delta-spark_2.12:3.2.0"]
    ).getOrCreate()
    
    return spark


def ingest_csv(spark, input_path, output_path, partition_col=None, target_file_rows=50000):
    """
    Ingest a CSV into Delta format with optional partitioning and target file size (rows per file).

    Args:
        spark: SparkSession
        input_path: Path to CSV
        output_path: Delta output path
        partition_col: Optional partition column name
        target_file_rows: Approximate max rows per file (default 50k)
    """
    df = spark.read.option("header", True).csv(input_path, inferSchema=True)
    total_rows = df.count()
    if total_rows == 0:
        print(f"[WARN] Empty CSV: {input_path}")
        return df

    # Calculate number of partitions (at least 2)
    partitions = max(2, total_rows // target_file_rows)
    df = df.repartition(partitions)
   
    # Handle partition column (for orders)
    if partition_col:
        if partition_col not in df.columns and partition_col == "order_purchase_month":
            df = df.withColumn(
                partition_col,
                date_format(col("order_purchase_timestamp"), "yyyy-MM")
            )
            print(f"[INFO] Added derived partition column '{partition_col}'")
        elif partition_col in df.columns:
            col_type = dict(df.dtypes).get(partition_col, "string")
            if "timestamp" in col_type or "date" in col_type:
                df = df.withColumn(partition_col, date_format(col(partition_col), "yyyy-MM"))
                print(f"[INFO] Converted partition column '{partition_col}' to yyyy-MM format")

    # Write as Delta
    writer = df.write.format("delta").mode("overwrite")
    if partition_col:
        writer = writer.partitionBy(partition_col)
    writer = writer.option("maxRecordsPerFile", str(target_file_rows))
    writer.save(output_path)
    print(f"[INFO] Ingested {input_path} → {output_path}")

    # Show file sizes
    if os.path.exists(output_path):
        print(f"[INFO] Files written under {output_path}:")
        for root, _, files in os.walk(output_path):
            for f in files:
                if f.endswith(".parquet"):
                    size_mb = os.path.getsize(os.path.join(root, f)) / (1024*1024)
                    print(f"   {f} → {size_mb:.2f} MB")

    return df