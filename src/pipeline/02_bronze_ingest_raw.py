import os
import pandas as pd
from pyspark.sql import SparkSession
from src.spark_session import create_spark_session
from delta.tables import DeltaTable

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Caminhos
raw_transactions_dir = os.path.join(BASE_DIR, "data", "raw", "transactions")
raw_customer_csv = os.path.join(BASE_DIR, "data", "raw", "customer_profiles", "customer_profiles_table.csv")
raw_terminal_csv = os.path.join(BASE_DIR, "data", "raw", "terminal_profiles", "terminal_profiles_table.csv")
bronze_dir = os.path.join(BASE_DIR, "data", "bronze")

# Inicializa Spark
spark = create_spark_session()

# 1. Carrega os dados transacionais (pkl)
pkl_files = [os.path.join(raw_transactions_dir, f) for f in os.listdir(raw_transactions_dir) if f.endswith(".pkl")]
df_list = [pd.read_pickle(f) for f in pkl_files]
transactions_pd = pd.concat(df_list)
transactions_spark = spark.createDataFrame(transactions_pd)

# 2. Carrega os dados de cliente e terminal
customer_spark = spark.read.csv(raw_customer_csv, header=True, inferSchema=True)
terminal_spark = spark.read.csv(raw_terminal_csv, header=True, inferSchema=True)

# 3. Salva como Delta na camada Bronze
transactions_spark.write.format("delta").mode("overwrite").save(os.path.join(bronze_dir, "transactions"))
customer_spark.write.format("delta").mode("overwrite").save(os.path.join(bronze_dir, "customers"))
terminal_spark.write.format("delta").mode("overwrite").save(os.path.join(bronze_dir, "terminals"))
