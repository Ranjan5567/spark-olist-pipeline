import pytest
from pyspark.sql import SparkSession
from chispa.dataframe_comparer import assert_df_equality
from pyspark.sql import Row

import os 
import sys
# Set PySpark environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\ranjan\Desktop\spark-olist-pipeline\venv\Scripts\python.exe"

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from silver import *

# -------------------------------
# Spark fixture
# -------------------------------
@pytest.fixture(scope="session")
def spark():
    spark = create_spark_session("TestSilver")
    yield spark
    spark.stop()


# ------------------------------------------------
# Unit tests
# ------------------------------------------------

def test_cleanse_text_fields(spark):
    df = spark.createDataFrame([Row(city=" SãO   paulo!! ")])
    result = cleanse_text_fields(df, ["city"])
    expected = spark.createDataFrame([Row(city="são paulo")])
    assert_df_equality(result, expected, ignore_row_order=True, ignore_column_order=True)


def test_deduplicate_customers(spark):
    df = spark.createDataFrame([
        Row(customer_unique_id="c1", name="A"),
        Row(customer_unique_id="c1", name="B"),
        Row(customer_unique_id="c2", name="C"),
    ])
    result = deduplicate_customers(df)
    assert result.count() == 2


def test_deduplicate_sellers(spark):
    df = spark.createDataFrame([
        Row(seller_id="s1", name="X"),
        Row(seller_id="s1", name="Y"),
        Row(seller_id="s2", name="Z"),
    ])
    result = deduplicate_sellers(df)
    assert result.count() == 2


def test_salt_seller_id(spark):
    df = spark.createDataFrame([Row(seller_id="s1"), Row(seller_id="s2")])
    salted = salt_seller_id(df, num_salts=10)
    assert "seller_salt" in salted.columns
    values = [r["seller_salt"] for r in salted.collect()]
    assert all(0 <= v < 10 for v in values)


def test_curate_sales_end_to_end(spark, tmp_path):
    # create mock bronze data
    bronze_base = str(tmp_path / "bronze")
    silver_base = str(tmp_path / "silver")

    # Minimal required mock tables
    orders = spark.createDataFrame([Row(order_id="o1", customer_id="c1")])
    order_items = spark.createDataFrame([Row(order_id="o1", order_item_id=1,
                                             product_id="p1", seller_id="s1",
                                             shipping_limit_date="2025-09-01")])
    payments = spark.createDataFrame([Row(order_id="o1", payment_type="credit_card", payment_value=100.0)])
    customers = spark.createDataFrame([Row(customer_id="c1", customer_unique_id="u1",
                                           customer_city="São Paulo", customer_state="SP")])
    sellers = spark.createDataFrame([Row(seller_id="s1", seller_city="Campinas", seller_state="SP")])
    products = spark.createDataFrame([Row(product_id="p1", product_category_name="beleza_saude")])
    geolocation = spark.createDataFrame([Row(geolocation_city="Campinas", geolocation_state="SP")])
    category_translation = spark.createDataFrame([Row(product_category_name="beleza_saude",
                                                      product_category_name_english="health_beauty")])

    # Save as bronze
    orders.write.format("delta").mode("overwrite").save(f"{bronze_base}/orders")
    order_items.write.format("delta").mode("overwrite").save(f"{bronze_base}/order_items")
    payments.write.format("delta").mode("overwrite").save(f"{bronze_base}/order_payments")
    customers.write.format("delta").mode("overwrite").save(f"{bronze_base}/customers")
    sellers.write.format("delta").mode("overwrite").save(f"{bronze_base}/sellers")
    products.write.format("delta").mode("overwrite").save(f"{bronze_base}/products")
    geolocation.write.format("delta").mode("overwrite").save(f"{bronze_base}/geolocation")
    category_translation.write.format("delta").mode("overwrite").save(f"{bronze_base}/category_translation")

    # Run curation
    curate_sales(spark, bronze_base, silver_base)

    # Validate curated sales exists
    sales = spark.read.format("delta").load(f"{silver_base}/sales")