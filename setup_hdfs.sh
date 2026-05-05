#!/bin/bash

hdfs dfs -mkdir -p /asl_dataset

echo "Uploading dataset to hdfs"
hdfs dfs -put /home/navaneeth/Documents/SignLanguageTranslator/asl_dataset/* /asl_dataset/

hdfs dfs -ls /asl_dataset/

echo "HDFS setup complete. You can now run the PySpark pipeline."
