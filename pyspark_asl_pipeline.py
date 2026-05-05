import sys
import os
import time
import json
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure executors use the same Python executable (the venv) as the driver
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, pandas_udf, split
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.feature import StringIndexer, PCA, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf

def initSpark():
    """Initializes and returns a SparkSession."""
    return SparkSession.builder \
        .appName("ASL Gesture Classification Pipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

def loadData(spark, dataPath):
    """Loads image dataset using Spark's binaryFile data source."""
    print(f"Loading data from {dataPath}")
    df = spark.read.format("binaryFile").load(dataPath)
    return df

def extractLabels(df):
    """Extracts the class label from the file path into a new column."""
    # Matches the immediate parent directory of the .jpg or .jpeg file
    return df.withColumn("label_str", regexp_extract(col("path"), r"([^/]+)/[^/]+\.jpe?g$", 1))

@pandas_udf(ArrayType(FloatType()))
def preprocessImageUdf(content_series: pd.Series) -> pd.Series:
    """
    Pandas UDF for distributed image preprocessing.
    Reads binary content, converts to grayscale, resizes to 64x64,
    normalizes to [0,1], and flattens into a 1D array of 4096 floats.
    """
    from PIL import Image
    processed_images = []
    for content in content_series:
        try:
            # Read image from raw bytes
            img = Image.open(io.BytesIO(content)).convert('L') # Grayscale
            img = img.resize((64, 64))
            # Convert to numpy array, normalize and flatten
            img_arr = np.array(img, dtype=np.float32) / 255.0
            processed_images.append(img_arr.flatten().tolist())
        except Exception as e:
            # In case of corrupt images, return array of zeros
            processed_images.append(np.zeros(64*64, dtype=np.float32).tolist())
    return pd.Series(processed_images)

def preprocessData(df):
    """Applies the distributed image preprocessing Pandas UDF."""
    return df.withColumn("image_array", preprocessImageUdf(col("content"))) \
             .drop("content") # Drop large binary content to save memory

def buildFeatures(df, usePca=False, pcaK=100):
    """
    Creates feature vector compatible with MLlib.
    Uses array_to_vector (if available) or UDF fallback, and StringIndexer for labels.
    """
    # Convert ArrayType to VectorUDT safely
    try:
        from pyspark.ml.functions import array_to_vector
        df = df.withColumn("features_raw", array_to_vector(col("image_array")))
    except ImportError:
        list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
        df = df.withColumn("features_raw", list_to_vector_udf(col("image_array")))
        
    df = df.drop("image_array")
    
    # Label Indexing
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

def trainModels(trainDf):
    """
    Trains Logistic Regression and Random Forest models.
    """
    print("Training Logistic Regression Model...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.1)
    lrModel = lr.fit(trainDf)
    
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, maxDepth=5)
    rfModel = rf.fit(trainDf)
    
    return {"Logistic Regression": lrModel, "Random Forest": rfModel}

def evaluateModel(predictions, modelName="Model"):
    """
    Computes performance metrics.
    """
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    evaluator_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    acc = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    prec = evaluator_prec.evaluate(predictions)
    rec = evaluator_rec.evaluate(predictions)
    
    print(f"--- Evaluation for {modelName} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return acc, f1, prec, rec

def dataAnalysis(df):
    """Analyzes and plots class distribution across the dataset."""
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
    
def plotConfusionMatrix(predictions, indexer_labels, modelName="Model"):
    """Generates and saves a confusion matrix heatmap."""
    conf_matrix_df = predictions.groupBy("label", "prediction").count().toPandas()
    
    num_classes = len(indexer_labels)
    matrix = np.zeros((num_classes, num_classes))
    for _, row in conf_matrix_df.iterrows():
        actual = int(row['label'])
        pred = int(row['prediction'])
        matrix[actual, pred] = row['count']
        
    plt.figure(figsize=(14, 10))
    sns.heatmap(matrix, xticklabels=indexer_labels, yticklabels=indexer_labels, cmap="Blues", annot=False)
    plt.title(f"Confusion Matrix - {modelName}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    fileName = f"confusion_matrix_{modelName.replace(' ', '_')}.png"
    plt.savefig(fileName)
    print(f"Saved confusion matrix plot to '{fileName}'")

def runScalabilityTests(spark, baseDf):
    """
    Trains LR model on different dataset sizes to measure scalability.
    Plots Training Time vs Dataset Size.
    """
    fractions = [0.1, 0.5, 1.0]
    times = []
    accuracies = []
    
    for frac in fractions:
        print(f"\nRunning Scalability Test on {frac*100}% of data...")
        sampleDf = baseDf.sample(withReplacement=False, fraction=frac, seed=42).cache()
        sampleDf.count() # Force action to cache
        
        trainDf, testDf = sampleDf.randomSplit([0.8, 0.2], seed=42)
        
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)
        
        start_time = time.time()
        model = lr.fit(trainDf)
        end_time = time.time()
        
        predictions = model.transform(testDf)
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        acc = evaluator.evaluate(predictions)
        
        duration = end_time - start_time
        times.append(duration)
        accuracies.append(acc)
        print(f"Time taken: {duration:.2f}s, Accuracy: {acc:.4f}")
        sampleDf.unpersist()
        
    # Plot Scalability Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([f * 100 for f in fractions], times, marker='o', linestyle='-', color='b')
    plt.title("Training Time vs Dataset Size")
    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Time (seconds)")
    
    plt.subplot(1, 2, 2)
    plt.plot([f * 100 for f in fractions], accuracies, marker='s', linestyle='-', color='g')
    plt.title("Accuracy vs Dataset Size")
    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    plt.savefig("scalability_test_results.png")
    print("Saved scalability test plot to 'scalability_test_results.png'")

def main():
    spark = initSpark()
    
    # Target dataset path (HDFS or local)
    dataset_path = "hdfs:///asl_dataset/*/*.jpeg"
    
    # If the user runs this locally without HDFS, they can override via args
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    print(f"Using dataset path: {dataset_path}")
    
    # 1. Ingestion
    raw_df = loadData(spark, dataset_path)
    
    # 2. Extract Labels
    labeled_df = extractLabels(raw_df)
    
    # Data Analysis Phase (before heavy preprocessing)
    dataAnalysis(labeled_df)
    
    # 3. Distributed Preprocessing
    preprocessed_df = preprocessData(labeled_df)
    
    # Cache intermediate dataframe
    preprocessed_df.cache()
    print(f"Total images loaded and preprocessed: {preprocessed_df.count()}")
    
    # 4. Feature Engineering (Compare PCA vs No PCA)
    print("\n--- Feature Engineering ---")
    df_pca = buildFeatures(preprocessed_df, usePca=True, pcaK=50)
    df_no_pca = buildFeatures(preprocessed_df, usePca=False)
    
    df_final = df_no_pca.cache()
    
    # Extract string labels from StringIndexer metadata
    indexer_model = StringIndexer(inputCol="label_str", outputCol="label").fit(preprocessed_df)
    labels = indexer_model.labels
    
    # Split Dataset
    trainDf, testDf = df_final.randomSplit([0.8, 0.2], seed=123)
    trainDf.cache()
    testDf.cache()
    
    # 5. Model Training
    print("\n--- Model Training ---")
    models = trainModels(trainDf)
    
    # 6. Evaluation
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        preds = model.transform(testDf)
        acc, f1, prec, rec = evaluateModel(preds, modelName=name)
        plotConfusionMatrix(preds, labels, modelName=name)
        
        # Save Model
        local_dir = os.path.abspath(os.getcwd())
        model_path = f"file://{local_dir}/saved_model_{name.replace(' ', '_')}"
        model.write().overwrite().save(model_path)
        print(f"Saved {name} model locally to {model_path}")
        
        # Save Evaluation Metrics
        metrics = {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}
        metrics_file = f"metrics_{name.replace(' ', '_')}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Saved {name} evaluation metrics to {metrics_file}")
        
    # Evaluate PCA impact briefly on LR
    print("\n--- Evaluating Impact of PCA (Logistic Regression) ---")
    trainDf_pca, testDf_pca = df_pca.randomSplit([0.8, 0.2], seed=123)
    lr_pca = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.1)
    
    start_time_pca = time.time()
    lr_pca_model = lr_pca.fit(trainDf_pca)
    time_pca = time.time() - start_time_pca
    
    preds_pca = lr_pca_model.transform(testDf_pca)
    eval_pca = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    acc_pca = eval_pca.evaluate(preds_pca)
    print(f"PCA LR Training Time: {time_pca:.2f}s, Accuracy: {acc_pca:.4f}")
        
    # 7 & 8. Scalability Testing
    print("\n--- Starting Scalability Experiments ---")
    runScalabilityTests(spark, df_final)
    
    # Save the processed dataframe as parquet
    print("\nSaving processed dataset to parquet...")
    parquet_path = f"file://{local_dir}/processed_asl_dataset.parquet"
    df_final.write.mode("overwrite").parquet(parquet_path)
    print("Done. PySpark Pipeline Execution Completed Successfully.")
    
    spark.stop()

if __name__ == "__main__":
    main()
