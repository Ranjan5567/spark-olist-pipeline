from pathlib import Path
import os
import sys
import pytest
from chispa.dataframe_comparer import assert_df_equality

    
# Set PySpark environment
os.environ["PYSPARK_PYTHON"] = r"C:\Users\ranjan\Desktop\spark-olist-pipeline\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\ranjan\Desktop\spark-olist-pipeline\venv\Scripts\python.exe"

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bronze import create_spark_session, ingest_csv

# -------------------------------
# Spark fixture
# -------------------------------
@pytest.fixture(scope="session")
def spark():
    spark = create_spark_session("TestSpark")
    yield spark
    spark.stop()

# -------------------------------
# Test Orders Table
# -------------------------------
from pathlib import Path
import os
import sys
import pytest
from chispa.dataframe_comparer import assert_df_equality

    
# Set PySpark environment
os.environ["PYSPARK_PYTHON"] = r"C:\Users\ranjan\Desktop\spark-olist-pipeline\venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\ranjan\Desktop\spark-olist-pipeline\venv\Scripts\python.exe"

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bronze import create_spark_session, ingest_csv

# -------------------------------
# Spark fixture
# -------------------------------
@pytest.fixture(scope="session")
def spark():
    spark = create_spark_session("TestSpark")
    yield spark
    spark.stop()

# -------------------------------
# Test Orders Table
# -------------------------------
def test_ingest_orders_with_partition_and_file_size(spark, tmp_path):
    input_path = "data/olist_orders_dataset.csv"
    output_path = tmp_path / "orders"

    # Ingest orders with partitioning and target file rows
    ingest_csv(
        spark,
        str(input_path),
        str(output_path),
        partition_col="order_purchase_month",
        target_file_rows=50000
    )

    df = spark.read.format("delta").load(str(output_path))

    # Check partition column exists
    assert "order_purchase_month" in df.columns, "Partition column missing"

    # Check that multiple parquet files exist (depending on partitions)
    parquet_files = list(Path(output_path).rglob("*.parquet"))
    assert len(parquet_files) > 1, "Expected multiple parquet files for orders"

    # Check file sizes
    file_sizes = [f.stat().st_size / (1024*1024) for f in parquet_files]
    assert any(size >= 0.1 for size in file_sizes), "No file >= 0.1 MB; check ingestion logic"
    
    print(f"[DEBUG] Max file size: {max(file_sizes):.2f} MB")
    print(f"[DEBUG] Total files: {len(file_sizes)}")

# -------------------------------
# Test Other Bronze Tables
# -------------------------------
@pytest.mark.parametrize("filename", [
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
])
def test_ingest_other_files(spark, tmp_path, filename):
    input_path = f"data/{filename}"
    output_path = tmp_path / filename.replace(".csv", "")

    # Ingest with target_file_rows for consistency
    ingest_csv(spark, str(input_path), str(output_path), target_file_rows=50000)

    df = spark.read.format("delta").load(str(output_path))

    # Ensure table is not empty
    count = df.count()
    assert count > 0, f"{filename} table is empty"
    print(f"[DEBUG] {filename} ingested rows: {count}")

    # Ensure at least 2 parquet files if dataset is big enough
    parquet_files = list(Path(output_path).rglob("*.parquet"))
    if count > 50000:
        assert len(parquet_files) >= 2, f"{filename} should have multiple files for large dataset"
