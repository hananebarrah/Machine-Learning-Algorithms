# Machine-Learning-Algorithms
This repository includes basic ML algorithms coded in Python from scratch

## 1. Linear Regression algorithm
Linear Regression is a supervised ML algorithm that aims to find the line (in case of 2D) that best fits a set of 2D scattered points. The model can be extended to a higher dimension, in this case we try to find the hyperplane that best approximate the scattered points. If the dimension of the input data (point) is D > 1 then the hyperplane dimension is (D-1), in this case the algorithm is called ***Multiple Linear Regression***.

In linear_regression2D.py is implemented the ***Simple Linear Regression*** algorithm that uses the Gradient Descent optimization method to find the values of the coefficients b (bias or intercept) and m that define the line that best approximate the data included in the 'data.csv' file. The equation of that line is defined as follows:
  ![eq1](http://latex.codecogs.com/gif.latex?Y%20%3D%20m%5Ccdot%20X%20&plus;%20b)

## 2. K-means Clustering  algorithm
k-means is (unsupervised algorithm) a hard clustering algorithm that consists of grouping data into the most homogeneous groups. It is based on the classical theory of sets to creates homogeneous groups. Actually, during the clustering process each datum is assigned to one and only one cluster (the cluster of the nearest center).

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

   ![prediction](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_%7Bi%7D%20%3D%20P%28X_%7Bi%7D%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-%28B_%7B0%7D&plus;%5Csum_%7B1%7D%5E%7Bm%7DB_%7Bj%7Dx_%7Bi%2C%20j%7D%29%7D%7D)
   
   ![b0](http://latex.codecogs.com/gif.latex?B_%7B0%7D)  is the bias or the intercept.

The main goal of this algorithm is to find the best values of ![bj](http://latex.codecogs.com/gif.latex?B_%7Bj%7D) that minimize the following error using the gradient descent optimization method:

   ![error](http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B2N%7D%5Csum_%7B1%7D%5E%7BN%7D%28y_%7Bi%7D%20-%20%5Cwidehat%7By%7D_%7Bi%7D%29%5E%7B2%7D)
 
 The weights are modified iteratively as follows:
 
   ![b0](http://latex.codecogs.com/gif.latex?B_%7B0%7D%20%3D%20B_%7B0%7D%20&plus;%20%5Calpha%20%5Ctimes%20%28y_%7Bi%7D-%5Cwidehat%7By%7D_%7Bi%7D%29%5Ctimes%20%5Cwidehat%7By%7D_%7Bi%7D%5Ctimes%20%281-%5Cwidehat%7By%7D_%7Bi%7D%29)
 
   ![bj](http://latex.codecogs.com/gif.latex?B_%7Bj%7D%20%3D%20B_%7Bj%7D%20&plus;%20%5Calpha%20%5Ctimes%20%28y_%7Bi%7D-%5Cwidehat%7By%7D_%7Bi%7D%29%5Ctimes%20%5Cwidehat%7By%7D_%7Bi%7D%5Ctimes%20%281-%5Cwidehat%7By%7D_%7Bi%7D%29%5Ctimes%20x_%7Bi%2Cj%7D)
   
   ![alpha](http://latex.codecogs.com/gif.latex?%5Calpha)  is the learning rate its good values are between 0.1 and 0.3, ![yi](http://latex.codecogs.com/gif.latex?y_%7Bi%7D)  is the actual label, ![yi^](http://latex.codecogs.com/gif.latex?%5Cwidehat%7By%7D_%7Bi%7D) is the predicted value, and ![xij](http://latex.codecogs.com/gif.latex?x_%7Bi%2Cj%7D)  is the ![jTh](http://latex.codecogs.com/gif.latex?j%5E%7Bth%7D) feature of the ![iTh](http://latex.codecogs.com/gif.latex?i%5E%7Bth%7D) observation.
   
 
 In the file LogisticRegression.py the logistic regression algorithm is implemented and tested on synthetic 2D data.
 
 ## 4. Linear Discriminant Analysis
 Linear Discriminant Analysis is a supervised ML algorithm used for multi-class classification problems. The model is defined by the mean of each class (![muk](http://latex.codecogs.com/gif.latex?%5Cmu_%7Bk%7D)) and the variance (![sigma](http://latex.codecogs.com/gif.latex?%5Csigma%20%5E%7B2%7D)):
 
  ![mean](http://latex.codecogs.com/gif.latex?%5Cmu%20_%7Bk%7D%20%3D%20%5Cfrac%7B1%7D%7Bn_%7Bk%7D%7D%5Ctimes%20%5Csum_%7Bi%5Cin%20C_%7Bk%7D%7D%5E%7B%20%7Dx_%7Bi%7D)
  
  ![variance](http://latex.codecogs.com/gif.latex?%5Csigma%20%5E%7B2%7D%20%3D%20%5Cfrac%7B1%7D%7Bn-K%7D%5Ctimes%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_%7Bi%7D%20-%20%5Cmu%20_%7Bk%7D%29%5E%7B2%7D)
  
![n](http://latex.codecogs.com/gif.latex?n) is the total number of observations, ![K](http://latex.codecogs.com/gif.latex?K) is the number of classes, and ![nk](http://latex.codecogs.com/gif.latex?n_%7Bk%7D) is the number of observations in the ![kTh](http://latex.codecogs.com/gif.latex?k%5E%7Bth%7D) class. ![muk](http://latex.codecogs.com/gif.latex?%5Cmu%20_%7Bk%7D) is the mean of the ![kTh](http://latex.codecogs.com/gif.latex?k%5E%7Bth%7D) class. ![muk](http://latex.codecogs.com/gif.latex?%5Cmu%20_%7Bk%7D) class to which ![xi](http://latex.codecogs.com/gif.latex?x%20_%7Bk%7D) belongs.

The predictions are made by estimating the probability that a new input (x) belongs to each class, which are calculated using the Bayes Theorem:

  ![proba](http://latex.codecogs.com/gif.latex?P%28Y%20%3D%20k%7CX%20%3D%20x%29%3D%5Cfrac%7BP%28k%29%5Ctimes%20P%28x%7Ck%29%7D%7B%5Csum_%7Bl%3D1%7D%5E%7BK%7DP%28l%29%5Ctimes%20P%28x%7Cl%29%7D)
 
## Dependencies
- numpy
- matplotlib

## References
Brownlee, J. Master Machine Learning Algorithms: Discover How They Work and Implement Them From Scratch. Jason Brownlee, 2016. https://books.google.co.ma/books?id=PdZBnQAACAAJ.
