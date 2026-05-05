# ASL Gesture Classification PySpark Pipeline

This project provides a complete, distributed Apache Spark (PySpark) machine learning pipeline designed to classify American Sign Language (ASL) gestures from images.

The pipeline demonstrates Big Data processing capabilities by using PySpark's DataFrame APIs, distributed Pandas UDFs for image preprocessing, and Spark MLlib for model training and evaluation.

## Features
- **Scalable Image Ingestion**: Uses Spark's `binaryFile` data source to load raw image data across the cluster efficiently.
- **Distributed Preprocessing**: Uses a `Pandas UDF` to distribute the heavy lifting of image processing (grayscale conversion, resizing, normalization, and flattening).
- **Feature Engineering**: Supports dimensionality reduction using Principal Component Analysis (PCA).
- **Model Training**: Trains both **Logistic Regression** and **Random Forest** models using the Spark MLlib Pipeline API.
- **Scalability Experiments**: Automatically tests training performance across varying fractions of the dataset (10%, 50%, 100%) to demonstrate Spark's horizontal scalability.
- **Visual Analytics**: Generates plots for class distributions, confusion matrices, and scalability performance metrics.
- **Modular Architecture**: The codebase is cleanly separated into data pipeline, model training, evaluation, and orchestration modules for easy maintenance.

## Prerequisites
- **Apache Spark 3.0+** (Spark 3.3+ recommended for `array_to_vector` optimization)
- **Python 3.7+**
- **Libraries**: `pyspark`, `pandas`, `numpy`, `pillow`, `matplotlib`, `seaborn`, `pyarrow`

## Setup Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies.

1. Create a virtual environment:
```bash
python3 -m venv venv
```
2. Activate the virtual environment:
```bash
source venv/bin/activate
```
3. Install the required packages (PyArrow is required for efficient Pandas UDF execution):
```bash
pip install pyspark pandas numpy pillow matplotlib seaborn pyarrow
```

## Setup & Data Preparation

The pipeline expects the data to be in HDFS for distributed processing. If your dataset is currently local, you can use the provided bash script to upload it.

1. Ensure your local dataset is located at `/home/navaneeth/Documents/SignLanguageTranslator/asl_dataset`. The folder structure should have subdirectories for each class (e.g., `A/`, `B/`, `space/`).
2. Run the HDFS setup script:
```bash
bash setup_hdfs.sh
```

## Running the Pipeline

### On a Spark Cluster (YARN)
If you are running on a Hadoop/YARN cluster, submit the job using `spark-submit`:

```bash
spark-submit \
  --master yarn \
  --driver-memory 8g \
  --executor-memory 4g \
  main.py
```

### Running Locally (For Testing)
If you don't have HDFS or want to test locally, you can pass the local dataset path as an argument:

```bash
spark-submit \
  --driver-memory 8g \
  --executor-memory 4g \
  main.py "file:///home/navaneeth/Documents/SignLanguageTranslator/asl_dataset/*/*.jpeg"
```

## Project Structure & Pipeline Stages

The pipeline is split into four modular files:

1. **`data_pipeline.py`**: Handles ingestion (`loadData()`), label extraction (`extractLabels()`), distributed image resizing and normalization (`preprocessData()`), and feature engineering (`buildFeatures()`).
2. **`model_training.py`**: Contains the algorithms to train the **Logistic Regression** and **Random Forest** models.
3. **`evaluation_analysis.py`**: Handles metric calculations, confusion matrix plotting, and scalability tests.
4. **`main.py`**: The orchestrator that initializes Spark, sets up paths, and sequentially calls the pipeline stages.

## Outputs
When the pipeline finishes, it generates the following files in your current working directory:
- `class_distribution.png`: Bar chart of class frequencies.
- `confusion_matrix_Logistic_Regression.png`: Heatmap for the LR model.
- `confusion_matrix_Random_Forest.png`: Heatmap for the RF model.
- `scalability_test_results.png`: Graphs showing training time and accuracy vs. dataset size.
- `metrics_Logistic_Regression.json`: JSON file with accuracy, precision, recall, and F1 metrics.
- `metrics_Random_Forest.json`: JSON file with accuracy, precision, recall, and F1 metrics.
- `saved_model_Logistic_Regression/`: Saved LR model weights/metadata.
- `saved_model_Random_Forest/`: Saved RF model weights/metadata.
- `processed_asl_dataset.parquet`: The fully preprocessed feature dataset saved in Parquet format.
