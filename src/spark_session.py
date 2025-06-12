from pyspark.sql import SparkSession

def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("FraudDetectionPipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    return spark
