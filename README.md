# Machine-Learning-Algorithms
This repository includes basic ML algorithms coded in Python from scratch

## 1. Linear Regression algorithm
Linear Regression is a supervised ML algorithm that aims to find the line (in case of 2D) that best fits a set 2D scattered points. The model can be extended to a higher dimension, in this case we try to find the hyperplane that best approximate the scattered points. If the dimension of data (point) is N then the hyperplane dimension is (N-1).
The code presented in linear_regression2D.py uses the Gradient Descent optimization method to find the values of the coefficients b and m that define the line that best approximate the data included in the 'data.csv' file. The equation of that line is defined as follows:
  ![eq1](http://latex.codecogs.com/gif.latex?Y%20%3D%20m%5Ccdot%20X%20&plus;%20b)

## 2. K-means Clustering  algorithm
k-means in a hard clustering algorithm that consists of grouping data into the most homogeneous groups. It is based on the classical theory of sets to creates homogeneous groups. Actually, during the clustering process each datum is assigned to one and only one cluster (the cluster of the nearest center).

It takes as input a set of data ![eq2](http://latex.codecogs.com/gif.latex?D%20%3D%20%5Cleft%20%5C%7B%20X_%7Bj%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20%5Cright%20%5C%7D_%7Bj%20%3D%201%2C%20...%2C%20N%7D) (n in the number of features and N in number of observations), the number of clusters k and creates iteratively k clusters.

|Algorithm|
| --- |
|1.	Input the dataset D and initialize the clusters number k.|
|2.	Initialize the clusters centers ![eq3](http://latex.codecogs.com/gif.latex?C%20%3D%20%5Cleft%20%5C%7B%20C_%7Bi%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20%5Cright%20%5C%7D_%7Bi%20%3D%201%2C%20...%2C%20k%7D)  randomly from the input dataset.|

# Dependencies
- numpy
- matplotlib
