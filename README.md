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
|3.	Assign each datum ![eq4](http://latex.codecogs.com/gif.latex?X_%7Bj%7D) to the cluster of the nearest center|
|4.	Update the clusters centers, each center ![eq5](http://latex.codecogs.com/gif.latex?C_%7Bi%7D) is computed as the mean of the points belonging to the ![eq6](http://latex.codecogs.com/gif.latex?i%5E%7Bth%7D) cluter.|
|5.	Compute  ![eq7](http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20C%5E%7BI%7D%20-%20C%5E%7BI-1%7D%5Cright%20%5C%7C), if it is smaller than a fixed threshold stop iterating, otherwise, repeat the process from the third step.|

The python code of the k-means algorithm is included in the file k-means.py,  it is tested on synthetic data.

## 3. Logistic Regression
Logistic regression is a supervised learning method for binary classification. It inherits the name ***Logistic*** from the logistic function (called also sigmoid function) that is used at core of the algorithm.

   ![sigmoid](http://latex.codecogs.com/gif.latex?sigmoid%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D)
   
In this algorithm the features of the input observation  ![eq8](http://latex.codecogs.com/gif.latex?X_%7Bi%7D%3D%5Cleft%20%5B%20x_%7Bi%2C1%7D%2C...%2C%20x_%7Bi%2Cm%7D%20%5Cright%20%5D)  (m is the number of features) are combined linearly using some weights  ![eq9](http://latex.codecogs.com/gif.latex?B%3D%5Cleft%20%5B%20B_%7B0%7D%2C...%2C%20B_%7Bm%7D%20%5Cright%20%5D) to predict an output value ![eq10](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_%7Bi%7D):

   ![prediction](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_%7Bi%7D%20%3D%20P%28X_%7Bi%7D%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-%28B_%7B0%7D&plus;%5Csum_%7B1%7D%5E%7Bm%7DB_%7Bj%7Dx_%7Bi%2C%20j%7D%29%7D%7D)  ; ![b0](http://latex.codecogs.com/gif.latex?B_%7B0%7D)  is the bias or the intercept.

The main goal of this algorithm is to find the best values of ![bj](http://latex.codecogs.com/gif.latex?B_%7Bj%7D) that minimize the error ![error](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2N%7D%5Csum_%7B1%7D%5E%7BN%7D%28y_%7Bi%7D-%5Cwidehat%7By%7D_%7Bi%7D%29)  using the gradient descent optimization method.

# Dependencies
- numpy
- matplotlib
