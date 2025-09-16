from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, date_format

def create_spark_session(app_name="GoldLayer") -> SparkSession:
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

# ---------------------------
# Daily Sales by State (Gold)
# ---------------------------
def create_daily_sales_state(spark, silver_base, gold_base):
    sales_path = f"{silver_base}/sales"
    customers_path = f"{silver_base}/customers"
    sellers_path = f"{silver_base}/sellers"

    gold_path = f"{gold_base}/daily_sales_state"

    # Load Silver data
    sales = spark.read.format("delta").load(sales_path)
    customers = spark.read.format("delta").load(customers_path)
    sellers = spark.read.format("delta").load(sellers_path)

    # Join sales with customers & sellers
    df = (
        sales.alias("s")
        .join(customers.alias("c"), "customer_id")
        .join(sellers.alias("se"), "seller_id")
        .withColumn("order_date", F.to_date("s.order_purchase_timestamp"))
    )

    # Group by state and day
    daily_sales = (
        df.groupBy("order_date", "c.customer_state", "se.seller_state")
        .agg(F.sum(F.col("s.price") + F.col("s.freight_value")).alias("daily_revenue"))
    )

    # Save as Delta (manual optimize with repartition + sort)
    (
        daily_sales.repartition("order_date")
        .sortWithinPartitions("customer_state", "seller_state", "order_date")
        .write.format("delta")
        .mode("overwrite")
        .partitionBy("order_date")
        .option("dataChange", "true")
        .save(gold_path)
    )

    print(f"Gold table created at {gold_path} (customer + seller state revenue)")