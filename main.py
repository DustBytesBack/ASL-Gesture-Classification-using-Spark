import sys
import os
import time
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import from our modular files
from data_pipeline import (
    initSpark, loadData, extractLabels, dataAnalysis, 
    preprocessData, buildFeatures
)
from model_training import trainModels
from evaluation_analysis import (
    evaluateModel, saveEvaluationMetrics, plotConfusionMatrix, runScalabilityTests
)

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
        saveEvaluationMetrics(acc, f1, prec, rec, name)
        
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
    local_dir = os.path.abspath(os.getcwd())
    parquet_path = f"file://{local_dir}/processed_asl_dataset.parquet"
    df_final.write.mode("overwrite").parquet(parquet_path)
    print("Done. PySpark Pipeline Execution Completed Successfully.")
    
    spark.stop()

if __name__ == "__main__":
    main()
