from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('rec').getOrCreate()

recommender_user = spark.read.parquet('user_recs.parquet')