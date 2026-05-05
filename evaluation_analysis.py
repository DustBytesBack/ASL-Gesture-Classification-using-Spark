import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

def evaluateModel(predictions, modelName="Model"):
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

def saveEvaluationMetrics(acc, f1, prec, rec, modelName):
    metrics = {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}
    metrics_file = f"metrics_{modelName.replace(' ', '_')}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved {modelName} evaluation metrics to {metrics_file}")

def plotConfusionMatrix(predictions, indexer_labels, modelName="Model"):
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
    fractions = [0.1, 0.5, 1.0]
    times = []
    accuracies = []
    
    for frac in fractions:
        print(f"\nRunning Scalability Test on {frac*100}% of data...")
        sampleDf = baseDf.sample(withReplacement=False, fraction=frac, seed=42).cache()
        sampleDf.count()
        
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