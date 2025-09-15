from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, trim, lower, regexp_replace, concat_ws, monotonically_increasing_id, rand, when, broadcast
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip

# ----------------------------------------
# Utility: Cleanse text fields (cities, states)
# ----------------------------------------
def create_spark_session(app_name="SilverLayer") -> SparkSession:
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
    
def cleanse_text_fields(df: DataFrame, columns: list) -> DataFrame:
    for c in columns:
        df = df.withColumn(
            c,
            trim(
                lower(
                    regexp_replace(col(c), r"[^a-zA-ZÀ-ÿ0-9\s]", "")  # remove special chars
                )
            )
        )
        # collapse multiple spaces into one
        df = df.withColumn(c, regexp_replace(col(c), r"\s+", " "))
    return df

# ----------------------------------------
# Deduplicate customers and sellers
# ----------------------------------------
def deduplicate_customers(df: DataFrame) -> DataFrame:
    return df.dropDuplicates(["customer_unique_id"])

def deduplicate_sellers(df: DataFrame) -> DataFrame:
    return df.dropDuplicates(["seller_id"])

# ----------------------------------------
# Handle skew (salting on seller_id)
# ----------------------------------------
def salt_seller_id(df: DataFrame, num_salts: int = 5) -> DataFrame:
    return df.withColumn("seller_salt", (rand() * num_salts).cast("int"))

# ----------------------------------------
# Build curated sales fact table (Silver)
# ----------------------------------------
def curate_sales(spark: SparkSession, bronze_base: str, silver_base: str) -> None:
    # Load bronze tables
    orders = spark.read.format("delta").load(f"{bronze_base}/orders")
    order_items = spark.read.format("delta").load(f"{bronze_base}/order_items")
    payments = spark.read.format("delta").load(f"{bronze_base}/order_payments")
    customers = spark.read.format("delta").load(f"{bronze_base}/customers")
    sellers = spark.read.format("delta").load(f"{bronze_base}/sellers")
    products = spark.read.format("delta").load(f"{bronze_base}/products")
    geolocation = spark.read.format("delta").load(f"{bronze_base}/geolocation")
    category_translation = spark.read.format("delta").load(f"{bronze_base}/category_translation")

    # Deduplicate
    customers = deduplicate_customers(customers)
    sellers = deduplicate_sellers(sellers)

    # Cleanse text
    customers = cleanse_text_fields(customers, ["customer_city", "customer_state"])
    sellers = cleanse_text_fields(sellers, ["seller_city", "seller_state"])
    geolocation = cleanse_text_fields(geolocation, ["geolocation_city", "geolocation_state"])

    # Mitigate skew with salting
    order_items = salt_seller_id(order_items)

    # Broadcast join small table (category_translation)
    products = products.join(
        broadcast(category_translation),
        products["product_category_name"] == category_translation["product_category_name"],
        "left"
    ).drop(category_translation["product_category_name"])

    # Curated sales fact
    sales = (
        orders
        .join(order_items, "order_id", "inner")
        .join(payments, "order_id", "left")
        .join(customers, "customer_id", "left")
        .join(sellers, "seller_id", "left")
        .join(products, "product_id", "left")
    )

    # Cache for reuse
    sales.cache()

    # Write curated fact table
    sales.write.format("delta").mode("overwrite").save(f"{silver_base}/sales")
    customers.write.format("delta").mode("overwrite").save(f"{silver_base}/customers")
    sellers.write.format("delta").mode("overwrite").save(f"{silver_base}/sellers")
    print(" Silver layer sales curated and saved at", f"{silver_base}/sales")

    # Unpersist
    sales.unpersist()