from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

def trainLogisticRegression(trainDf):
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20, regParam=0.1)
    return lr.fit(trainDf)

def trainRandomForest(trainDf):
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, maxDepth=5)
    return rf.fit(trainDf)

def trainModels(trainDf):
    
    lrModel = trainLogisticRegression(trainDf)
    rfModel = trainRandomForest(trainDf)
    
    return {"Logistic Regression": lrModel, "Random Forest": rfModel}
