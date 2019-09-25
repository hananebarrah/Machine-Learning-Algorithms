# Machine-Learning-Algorithms
This repository includes basic ML algorithms coded in Python from scratch

## 1. Linear Regression algorithm
Linear Regression is a supervised ML algorithm that aims to find the line (in case of 2D) that best fits a set of 2D scattered points. The model can be extended to a higher dimension, in this case we try to find the hyperplane that best approximate the scattered points. If the dimension of the input data (point) is D > 1 then the hyperplane dimension is (D-1), in this case the algorithm is called ***Multiple Linear Regression***.

In *linear_regression2D.py* is implemented the ***Simple Linear Regression*** algorithm that uses the Gradient Descent optimization method to find the values of the coefficients b (bias or intercept) and m that define the line that best approximate the data included in the 'data.csv' file. The equation of that line is defined as follows:
  ![eq1](http://latex.codecogs.com/gif.latex?Y%20%3D%20m%5Ccdot%20X%20&plus;%20b)

## 2. K-means Clustering  algorithm
k-means is (unsupervised algorithm) a hard clustering algorithm that consists of grouping data into the most homogeneous groups. It is based on the classical theory of sets to creates homogeneous groups. Actually, during the clustering process each datum is assigned to one and only one cluster (the cluster of the nearest center).

It takes as input a set of data ![eq2](http://latex.codecogs.com/gif.latex?D%20%3D%20%5Cleft%20%5C%7B%20X_%7Bj%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20%5Cright%20%5C%7D_%7Bj%20%3D%201%2C%20...%2C%20N%7D) (n in the number of features and N in number of observations), the number of clusters k and creates iteratively k clusters.

|**k-means Algorithm**|
| --- |
|1.	Input the dataset D and initialize the clusters number k.|
|2.	Initialize the clusters centers ![eq3](http://latex.codecogs.com/gif.latex?C%20%3D%20%5Cleft%20%5C%7B%20C_%7Bi%7D%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D%20%5Cright%20%5C%7D_%7Bi%20%3D%201%2C%20...%2C%20k%7D)  randomly from the input dataset.|
|3.	Assign each datum ![eq4](http://latex.codecogs.com/gif.latex?X_%7Bj%7D) to the cluster of the nearest center.|
|4.	Update the clusters centers, each center ![eq5](http://latex.codecogs.com/gif.latex?C_%7Bi%7D) is computed as the mean of the points belonging to the ![eq6](http://latex.codecogs.com/gif.latex?i%5E%7Bth%7D) cluter.|
|5.	Compute  ![eq7](http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20C%5E%7BI%7D%20-%20C%5E%7BI-1%7D%5Cright%20%5C%7C), if it is smaller than a fixed threshold stop iterating, otherwise, repeat the process from the third step.

For each ![xi](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) in the training dataset:|

The python code of the k-means algorithm is included in the file *k-means.py*,  it is tested on synthetic data.

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
   
 
 In the file *LogisticRegression.py* the logistic regression algorithm is implemented and tested on synthetic 2D data.
 
 ## 4. Linear Discriminant Analysis
 Linear Discriminant Analysis is a supervised ML algorithm that is used for multi-class classification problems. The model is defined by the mean of each class (![muk](http://latex.codecogs.com/gif.latex?%5Cmu_%7Bk%7D)) and the variance (![sigma](http://latex.codecogs.com/gif.latex?%5Csigma%20%5E%7B2%7D)):
 
  ![mean](http://latex.codecogs.com/gif.latex?%5Cmu%20_%7Bk%7D%20%3D%20%5Cfrac%7B1%7D%7Bn_%7Bk%7D%7D%5Ctimes%20%5Csum_%7Bi%5Cin%20C_%7Bk%7D%7D%5E%7B%20%7Dx_%7Bi%7D)
  
  ![variance](http://latex.codecogs.com/gif.latex?%5Csigma%20%5E%7B2%7D%20%3D%20%5Cfrac%7B1%7D%7Bn-K%7D%5Ctimes%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28x_%7Bi%7D%20-%20%5Cmu%20_%7Bk%7D%29%5E%7B2%7D)
  
![n](http://latex.codecogs.com/gif.latex?n) is the total number of observations, ![K](http://latex.codecogs.com/gif.latex?K) is the number of classes, and ![nk](http://latex.codecogs.com/gif.latex?n_%7Bk%7D) is the number of observations in the ![kTh](http://latex.codecogs.com/gif.latex?k%5E%7Bth%7D) class. ![muk](http://latex.codecogs.com/gif.latex?%5Cmu%20_%7Bk%7D) is the mean of the ![kTh](http://latex.codecogs.com/gif.latex?k%5E%7Bth%7D) class to which ![xi](http://latex.codecogs.com/gif.latex?x_%7Bi%7D) belongs.

The predictions are made by estimating the probability that a new input ![x](http://latex.codecogs.com/gif.latex?x) belongs to each class, which are calculated as follows (The Bayes Theorem):

  ![proba](http://latex.codecogs.com/gif.latex?P%28Y%20%3D%20k%7CX%20%3D%20x%29%3D%5Cfrac%7BP%28k%29%5Ctimes%20P%28x%7Ck%29%7D%7B%5Csum_%7Bl%3D1%7D%5E%7BK%7DP%28l%29%5Ctimes%20P%28x%7Cl%29%7D)
 
By using a Gaussian Distribution Function to estimate the probability ( ![probaPxk](http://latex.codecogs.com/gif.latex?P%28x%7Ck%29) ) of belongingness of a new observation ![x](http://latex.codecogs.com/gif.latex?x) to the ![kTh](http://latex.codecogs.com/gif.latex?k%5E%7Bth%7D) class, we find the following discriminate funtion:

   ![Dxk](http://latex.codecogs.com/gif.latex?D_%7Bk%7D%28x%29%20%3D%20x%5Ctimes%20%5Cfrac%7B%5Cmu%20_%7Bk%7D%7D%7B%5Csigma%20%5E%7B2%7D%7D-%5Cfrac%7B%5Cmu%20_%7Bk%7D%5E%7B2%7D%7D%7B2%5Ctimes%20%5Csigma%20%5E%7B2%7D%7D&plus;ln%28P%28k%29%29)
   
   where:  
   ![pk](http://latex.codecogs.com/gif.latex?P%28k%29%20%3D%20%5Cfrac%7Bn_%7Bk%7D%7D%7Bn%7D)
  
The class that provides the highest value of  ![Dxk](http://latex.codecogs.com/gif.latex?D_%7Bk%7D%28x%29)  is the output class. 

The LDA algorithm is coded in the Python file *liear_discriminant_analysis.py* and tested of synthetic data generated using Gaussian Distribution function.

## 5. Naive Bayes Algorithm
Naive bayes is a supervised ML algorithm for binary and multi-class classification problems, it has many applications such as text classification, spam filtration, sentiment analysis and recommendation systems. It is based on the Bayes' Theorem.

|**Bayes' Theorem**|
| --- |
|The probability of the event A given B is the equal to the probability of the event B given A multiplied by the probability of A upon the probability of B.
![BayesTh](https://latex.codecogs.com/gif.latex?P%28A%7CB%29%3D%5Cfrac%7BP%28B%7CA%29*P%28A%29%7D%7BP%28B%29%7D)|
|![posterior](https://latex.codecogs.com/gif.latex?P%28A%7CB%29%3A)  is the probability of occurrence of event A given the event B is true. It is called the ***Posterior Probability***.|
|![ab](https://latex.codecogs.com/gif.latex?P%28A%29%20and%20P%28B%29%3A)  these are the probabilities of accurrence of event A and B respectively. The first quantity is called the ***Prior Probability of Proposition***, while the last one is called the ***Prior Probability of Evidence***.|
|![ba](https://latex.codecogs.com/gif.latex?P%28B%7CA%29%3A)  is the probability of occurrence of event B given the event A is true. It is called the ***Likelihood***.|


In a classification problem where multi-dimensional data are assigned to ![k](https://latex.codecogs.com/gif.latex?K) classes. The goal is to calculate the conditional probability of a new observation ![x](https://latex.codecogs.com/gif.latex?X) belongs to a class ![ck](https://latex.codecogs.com/gif.latex?C_%7Bk%5Cin%20%7B1%2C%20...%2C%20K%7D%7D)  :

![CiX](https://latex.codecogs.com/gif.latex?P%28C_%7Bk%7D%7CX%29%20%3D%20P%28C_%7Bk%7D%7Cx_%7B1%7D%2C%20...%2C%20x_%7Bd%7D%29%20%3D%20%5Cfrac%7BP%28x_%7B1%7D%2C%20...%2C%20x_%7Bd%7D%7CC_%7Bk%7D%29*P%28C_%7Bk%7D%29%7D%7BP%28x_%7B1%7D%2C%20...%2C%20x_%7Bd%7D%29%7D)

Where ![d](https://latex.codecogs.com/gif.latex?d) is the dimension of the input data (number of attributes). The new observation ![x](https://latex.codecogs.com/gif.latex?X) belongs to the class that provides the ***highest probability***.

The Naive Bayes algorithm for multi-dimensional **categorical** data is implemented in the python file *naive_bayes_categorical_data.py* and tested on the data described in [1] and stored in the csv file *naive_bayes_data.csv*.

In the case of **non-categorical** data; data has real-valued attributes; the ***Gaussian Naive Bayes Algorithm*** is preferred. This later assumes that the distribution of features is Gaussian and calculates the conditional probabilities  ![xick](https://latex.codecogs.com/gif.latex?P%28x_%7Bi%7D%7CC_%7Bk%7D%29) using the statistical properties of the data as described bellow:

|Conditional Probabilities|Means|Standard Deviations²|
| :---: |:---: |:---: |
|![gnb](https://latex.codecogs.com/gif.latex?P%28x_%7Bi%7D%7CC_%7Bk%7D%29%20%3D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%20%5Csigma%20_%7Bi%2C%20k%7D%5E%7B2%7D%7D%7D%20%5Ccdot%20e%5E%7B-%5Cfrac%7B%28x_%7Bi%7D-%5Cmu%20_%7Bi%2C%20k%7D%29%5E%7B2%7D%7D%7B2%5Csigma%20_%7Bi%2C%20k%7D%5E%7B2%7D%7D%7D)| ![muk](https://latex.codecogs.com/gif.latex?%5Cmu%20_%7Bi%2C%20k%7D%3D%5Cfrac%7B1%7D%7Bn_%7Bk%7D%7D%5Ccdot%20%5Csum_%7Bj%5Cin%20C_%7Bk%7D%7Dx_%7Bi%2C%20j%7D)| ![sigma2](https://latex.codecogs.com/gif.latex?%5Csigma%20_%7Bi%2C%20k%7D%5E%7B2%7D%3D%5Cfrac%7B1%7D%7Bn_%7Bk%7D%7D%5Ccdot%20%5Csum_%7Bj%5Cin%20C_%7Bk%7D%7D%28x_%7Bi%2C%20j%7D-%5Cmu%20_%7Bi%2C%20k%7D%29%5E%7B2%7D)|

![nk](https://latex.codecogs.com/gif.latex?n_%7Bk%7D)  is number of observations in the class ![k](https://latex.codecogs.com/gif.latex?k).  ![xij](https://latex.codecogs.com/gif.latex?x_%7Bi%2C%20j%7D)  is the ![iTh](https://latex.codecogs.com/gif.latex?i%5E%7Bth%7D)  feature of the ![jTh](https://latex.codecogs.com/gif.latex?j%5E%7Bth%7D) observation.

The python code of the Gaussian Naive Bayes aldorithm is described in *gaussian_naive_bayes.py* and tested on synthetic data generated randomly using gaussian destributions. This code works on muli-dimensional data.

## 6. K-Nearest Neighbor Algorithm
**KNN** is a supervised ML amgorithm that does not require any learning step. Actually, it uses each time the entire training dataset to make a new prediction which makes it a little bit slower than the other ML algorithms. The prediction for a new data point is made my comparing it with all the observations in order to choose the ![k](https://latex.codecogs.com/gif.latex?k) most similar ones and then the result  is the mean/ median value in case of a **regression** problem, or the most common class value in case of a **classification** problem. To do so, a distance measure is used (Euclidean, Manhatten, Hamming, ...) based on the data properties. When the input variables are real-valued and have the same scale, the Euclidean distance is a good choice. It is calculated as  follows:

![distance](https://latex.codecogs.com/gif.latex?EuclideanDistance%28X%2C%20Y%29%3D%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7Bd%7D%28x_%7Bi%7D-y_%7Bi%7D%29%5E%7B2%7D%7D)

![x](https://latex.codecogs.com/gif.latex?X%3D%5Bx_%7B1%7D%2C%20...%2C%20x_%7Bd%7D%5D)

![y](https://latex.codecogs.com/gif.latex?Y%3D%5By_%7B1%7D%2C%20...%2C%20y_%7Bd%7D%5D)

The parameter ![k](https://latex.codecogs.com/gif.latex?k) has to be chosen using a **Cross-Validation** method. In fact, the training data has to be splitted, several times, into training and testing sets. Each time we test the algorithm with different values of ![k](https://latex.codecogs.com/gif.latex?k) in order to choose the ![k](https://latex.codecogs.com/gif.latex?k) that provides the best result. In addition, it is important to pay attention to the number of classes when choosing ![k](https://latex.codecogs.com/gif.latex?k). Actually, if the number of classes is odd then ![k](https://latex.codecogs.com/gif.latex?k) must be an even number, and the inverse is correct.

In *k_nearest_neighbor.py* the **KNN** algorithm is implemented and tested on synthetic data generated randomly using gaussian destributions. This code works on muli-dimensional data.

## 7. Learning Vector Quantization
**LVQ** is an artificial neural network algorithm for binary and multiclass classification that has been adopted for regression too. It uses the principle of the nearest neighbor used by the **KNN** algorithm. In contrast to KNN, the **LVQ** requires a training stage where it learns from the entire dataset a smaller set of vectors (called the *codebook*) that best characterize the separation of the classes. Instead of using the entire training data to search for the nearest neighbor, we search in the *codebook*.

|**Learning Algorithm**|
| --- |
|1- Initialize randomly the *codebook* with vectors generated randomly or chosen from the training dataset.|
|2- Update the *codebook* for each observation of the training dataset:|
|______ For each ![xi](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) in the training dataset:|
|____________ Choose its nearest vector ![Vnear](https://latex.codecogs.com/gif.latex?v_%7Bnearest%7D) from the *codebook*.|
|____________ If ![xi](https://latex.codecogs.com/gif.latex?x_%7Bi%7D)  and  ![Vnear](https://latex.codecogs.com/gif.latex?v_%7Bnearest%7D)  belong to the same class, then move ![Vnear](https://latex.codecogs.com/gif.latex?v_%7Bnearest%7D) closer to  ![xi](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) using the following expression:|
|__________________![close](https://latex.codecogs.com/gif.latex?v_%7Bnearest%7D%20%3D%20v_%7Bnearest%7D%20&plus;%20LearningRate%5Ctimes%20%28x_%7Bi%7D-v_%7Bnearest%7D%29)|
|____________ Else, move ![Vnear](https://latex.codecogs.com/gif.latex?v_%7Bnearest%7D) away from  ![xi](https://latex.codecogs.com/gif.latex?x_%7Bi%7D) using the following expression:|
|__________________![away](https://latex.codecogs.com/gif.latex?v_%7Bnearest%7D%20%3D%20v_%7Bnearest%7D%20-%20LearningRate%5Ctimes%20%28x_%7Bi%7D-v_%7Bnearest%7D%29)|

Updating the *codebook* has to be repeated several times, the number of iterations (called also **epochs**) is set by the user to 50, 100, or more.

![LR](https://latex.codecogs.com/gif.latex?LearningRate) is a parameter that controls the amount of moving a codebook vector to or away from the training data. This parameter has to be decreased in each iteration (epoch) in order to make minor changes when the processing is close to the end (to the maximum number of iterations).

The **LVQ** algorithm is coded in *learning_vector_quantization_regression.py* and tested on synthetic data generated randomly using Gaussian distributions.

# 8. Support Vector Machines: SVM
**SVM** is a supervised ML algorithm for binary classification problems. It aim at finding the optimal hyperplane that maximizes the margin width between two data classes using an optimization method such as the gradient descent. It assumes that all the data features are numerical.

The hyperplane is a linear surface that splits the surface into two parts. If the data points are in ![R^n](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5E%7Bn%7D), then the hyperplane is an ![n-1](https://latex.codecogs.com/gif.latex?%28n-1%29) dimensional subspace. The figure bellow shows the boundary decision (red line) that separates positive and negative samples.

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Separatrice_lineaire_avec_marges.svg/400px-Separatrice_lineaire_avec_marges.svg.png)


![w](https://latex.codecogs.com/gif.latex?w) is a vector that is perpendecular to the decision boundary (red line). positive samples are labeled as (+1) while the negative samples are labeled as (-1). Circled samples are called the **Support Vectors**. The decision rule is defined as follows:

|**Decision rule**|**Class**|**Label ![yi](https://latex.codecogs.com/gif.latex?y_%7Bi%7D)**|
| --- | --- | --- |
|![rule](https://latex.codecogs.com/gif.latex?w.x&plus;b%20%5Cgeqslant%201)|positive samples|1|
|![rule](https://latex.codecogs.com/gif.latex?w.x&plus;b%20%5Cleqslant%20-1)|megative samples|-1|


If we set ![fx](https://latex.codecogs.com/gif.latex?f%28x_%7Bi%7D%29%3Dw.x_%7Bi%7D&plus;b), then, according to the decision rule we have ![yifx](https://latex.codecogs.com/gif.latex?y_%7Bi%7D*f%28x_%7Bi%7D%29%5Cgeq%201) for all the well classified points. For samples inside the gutter we have ![yifx1](https://latex.codecogs.com/gif.latex?y_%7Bi%7D*f%28x_%7Bi%7D%29%3D%201).

The loss function is defined as follows:

![loss](https://latex.codecogs.com/gif.latex?LF%28x_%7Bi%7D%2C%20y_%7Bi%7D%2C%20f%28x_%7Bi%7D%29%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%2C%20if%20y_%7Bi%7D%5Ccdot%20f%28x_%7Bi%7D%29%5Cgeq%201%5C%5C%201-y_%7Bi%7D%5Ccdot%20f%28x_%7Bi%7D%29%2C%20else%5Cend%7Bmatrix%7D%5Cright.)

The main aim of the **SVM** algorithm is to find the parameters ![w](https://latex.codecogs.com/gif.latex?w) and ![b](https://latex.codecogs.com/gif.latex?b) that defines the decision boundary. In other words, the **SVM** aims at minimizing the loss function for all the training data and maximizing the margin between the two classes.

Let ![x+](https://latex.codecogs.com/gif.latex?x%5E%7B&plus;%7D) and ![x-](https://latex.codecogs.com/gif.latex?x%5E%7B-%7D) be a positive and a negative support vector respectively, then the width of the gutter if defined as:

![width](https://latex.codecogs.com/gif.latex?width%3D%28x_%7B&plus;%7D-x_%7B-%7D%29%5Ccdot%20%5Cfrac%7Bw%7D%7B%5Cleft%20%5C%7C%20w%20%5Cright%20%5C%7C%7D)

After simplifying the previous expression, we find that:

![width](https://latex.codecogs.com/gif.latex?width%20%3D%20%28x%5E%7B&plus;%7D-x%5E%7B-%7D%29%5Ccdot%20%5Cfrac%7Bw%7D%7B%5Cleft%20%5C%7C%20w%20%5Cright%20%5C%7C%7D)

The **SVM** optimization problem can be formulated as follows:

![svm](https://latex.codecogs.com/gif.latex?min_%7Bw%2C%20b%7D%20%5C%2C%20%5C%3B%20%5Clambda%20%5Ccdot%20%5Cleft%20%5C%7C%20w%20%5Cright%20%5C%7C%5E%7B2%7D%20&plus;%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20LF%28x_%7Bi%7D%2C%20y_%7Bi%7D%2C%20f%28x_%7Bi%7D%29%29)

![lambda](https://latex.codecogs.com/gif.latex?%5Clambda) is a regularizer parameter and ![N](https://latex.codecogs.com/gif.latex?N) is the total number of observations.


## Dependencies
- numpy
- matplotlib

## References
[1] Brownlee, J. Master Machine Learning Algorithms: Discover How They Work and Implement Them From Scratch. Jason Brownlee, 2016. https://books.google.co.ma/books?id=PdZBnQAACAAJ.
