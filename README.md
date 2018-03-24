# An Apache Spark pipeline for predicting power output

I have implemented an Apache Spark pipeline for predicting the power output of a power plant, given the following factors.

- Ambient Temperature (in Celsius)
- Ambient Pressure
- Relative Humidity
- Exhaust speed

These factors affect the power output of the plant and by making predictions, the plant will be able to allocate resources to meet the demand. The training data is available in the UCI repository [here](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant#)

The pipeline consists of two stages: 
- A *VectorAssembler* transformer that constructs the feature vectors 
- A *LinearRegression* estimator. 

I compare the root mean squared errors of the linear regression model with default hyper parameters against the best model obtained using cross validation evaluated on the root mean squared error.
