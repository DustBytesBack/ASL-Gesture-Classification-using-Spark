from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

def trainLogisticRegression(trainDf):
    """Trains a Logistic Regression model."""
    print("Training Logistic Regression Model...")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.1)
    return lr.fit(trainDf)

def trainRandomForest(trainDf):
    """Trains a Random Forest Classifier."""
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, maxDepth=5)
    return rf.fit(trainDf)

def trainModels(trainDf):
    """
    Trains Logistic Regression and Random Forest models.
    """
    lrModel = trainLogisticRegression(trainDf)
    rfModel = trainRandomForest(trainDf)
    
    return {"Logistic Regression": lrModel, "Random Forest": rfModel}
