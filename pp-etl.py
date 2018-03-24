''' Pyspark Template file '''
import sys
from pyspark.sql import SparkSession

# VectorAssembler
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# LinearRegression
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

# Pipeline
from pyspark.ml import Pipeline

# Tuning
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

if len(sys.argv) != 3:
    print ('Usage: spark-submit ' + sys.argv[0] + ' <appName> <master_node>')
    sys.exit(1)

appName = sys.argv[1]
master = sys.argv[2]

# Setup SparkConf and SparkContext

try:
    ss = SparkSession.builder.appName(appName).master(master).getOrCreate()
    print ('Started app ' + appName + ' with master ' + master)
except:
    print ('Error creating SparkSession')
    sys.exit(1)

''' Do what you need here '''

powerplantDF = ss.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('data.csv')

# Create feature vectors by combining the columns AT, V, AP, RH using VectorAssembler
vec_assembler = VectorAssembler(inputCols=['AT', 'V', 'AP', 'RH'], outputCol='features')

# Create an estimator object
lr = LinearRegression().setLabelCol('PE')

# Build the pipeline
pipeline = Pipeline(stages=[vec_assembler, lr])

# Model data - split into train and test
(testDF, trainDF) = powerplantDF.randomSplit([0.2, 0.8])

# a) Arbitrary hyperparameter setting

# Train the model
model = pipeline.fit(trainDF)

# Make predictions
predictionDF = model.transform(testDF)

# Choose (observation, prediction) pairs. Need this for calculating metrics
metricDF = predictionDF.select('PE', 'prediction')

# Calculate metrics
metrics = RegressionMetrics(metricDF.rdd)
print ("Default LR metrics: RMSE = %s, R2 = %s, MSE = %s" %(metrics.rootMeanSquaredError, metrics.r2, metrics.meanSquaredError))

# b) Parameter Tuning using CV

paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [10, 25, 50]).addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4]).build()

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator().setLabelCol('PE').setMetricName('rmse'), numFolds=3)

cvModel = cv.fit(trainDF)

# Make predictions
predictionDF = cvModel.transform(testDF)

# Choose (observation, prediction) pairs. Need this for calculating metrics
metricDF = predictionDF.select('PE', 'prediction')

# Calculate metrics
metrics = RegressionMetrics(metricDF.rdd)
print ("CV LR metrics: RMSE = %s, R2 = %s, MSE = %s" %(metrics.rootMeanSquaredError, metrics.r2, metrics.meanSquaredError))

# print ('Best Param (regParam, maxIter): (%s, %s)' %(cvModel.bestModel.stages[1]._java_obj.parent().getRegParam(), cvModel.bestModel.stages[-1]._java_obj.parent().getMaxIter()))

''' Stop the SparkContext'''
ss.stop()

print ("Spark session closed.")
