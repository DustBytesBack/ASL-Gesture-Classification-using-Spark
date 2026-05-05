import sys
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import StringIndexer, PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

def initSpark():
    return SparkSession.builder \
        .appName("ASL Gesture Classification Pipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def loadData(spark, dataPath):
    print(f"Loading data from {dataPath}")
    df = spark.read.format("binaryFile").load(dataPath)
    return df

def extractLabels(df):
    return df.withColumn("label_str", regexp_extract(col("path"), r"([^/]+)/[^/]+\.jpe?g$", 1))

@pandas_udf(ArrayType(FloatType()))
def preprocessImageUdf(content_series: pd.Series) -> pd.Series:
    processed_images = []
    for content in content_series:
        try:
            img = Image.open(io.BytesIO(content)).convert('L') # Grayscale
            img = img.resize((64, 64))
            img_arr = np.array(img, dtype=np.float32) / 255.0
            processed_images.append(img_arr.flatten().tolist())
        except Exception as e:
            processed_images.append(np.zeros(64*64, dtype=np.float32).tolist())
    return pd.Series(processed_images)

def preprocessData(df):
    return df.withColumn("image_array", preprocessImageUdf(col("content"))) \
             .drop("content")

def buildFeatures(df, usePca=False, pcaK=100):
    try:
        from pyspark.ml.functions import array_to_vector
        df = df.withColumn("features_raw", array_to_vector(col("image_array")))
    except ImportError:
        list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
        df = df.withColumn("features_raw", list_to_vector_udf(col("image_array")))
        
    df = df.drop("image_array")
    
    indexer = StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="skip")
    df = indexer.fit(df).transform(df)
    
    if usePca:
        print(f"Applying PCA (k={pcaK})")
        pca = PCA(k=pcaK, inputCol="features_raw", outputCol="features")
        pcaModel = pca.fit(df)
        df = pcaModel.transform(df)
        df = df.drop("features_raw")
    else:
        df = df.withColumnRenamed("features_raw", "features")
        
    return df

def dataAnalysis(df):
    print("Running Data Analysis: Class Distribution")
    class_counts = df.groupBy("label_str").count().orderBy("label_str").toPandas()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="label_str", y="count", data=class_counts, palette="viridis", hue="label_str")
    plt.title("Class Distribution in Dataset")
    plt.xlabel("ASL Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    print("Saved class distribution plot to 'class_distribution.png'")
