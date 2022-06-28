# Statical Data Analysis
 MNIST dataset using SVM with Kernel and without Kernel and ﻿Comparing with Logistic Regression(Python &amp; ML)

# Introduction
This work is demonstrated by predicting the accuracy of models. The MNIST dataset is used in this project using the Support Vector Machine model without kernel 
and with Kernel which should correctly classify the handwritten digits from 0-9 based on the pixel values given as features. Thus, this is a 10- class 
classification problem. The same Dataset is used in Logistic regression which is another model which could classify the handwritten digits from 0-9.  
Then compare the results of the SVM model and Logistic Regression model to find out more accuracy. The more accurate model is taken to overcome the error
(digits Mismatched) on the less accurate model. And also it explains how it works and the results obtained. The Tools we are used in this project are Python
Programming language and libraries.

This work is demonstrated by predicting the accuracy of models. The MNIST dataset is used in this project using the Support Vector Machine and
logistic regression....... In svm we are using without kernel that is linear SVM and with Kernel that are Polynomial and RBF and both the values
of SVM and logistic are taken and compared to know which is the more accuracy. here we have used python programming language.

#MNIST DATA
In this project, a handwritten digits recognition system was implemented with the famous MNIST data set. This is not a new topic what we are using 
right now..... the MNIST data set is very popular and important for the evaluation and validation of new algorithms. Handwritten digit recognition is
the ability of computers to recognize human handwritten digits.
It is a hard task for the machine because handwritten digits are not perfect. The MNIST Dataset contains 70,000 images of
handwritten digits. The MNIST dataset has different classes i.e. (0, 1, 2, 3, 4, 5, 6, 7, 8, and 9).Each image is 28 pixels by 28 pixels matrix
where each cell contains a grayscale pixel valueand we can think of it as a vector of 28x28 = 784 numbers.

The main goal or the process of the project follows as....... Step by step process....... the first step is to
import the required library function like numpy, Matplot library function , Sklearn all this library
should by imported ....... as numpy is used for matrix representation like multi dimensional array
and matplot is used for ploting the graph and Sklearn is used for importing the algorithms which
are already inbuilt in it.... later we are importing the MNIST data sets..... each MNIST data point
has 2 parts 1. Image of hand written digits and corresponding labels . we have to train the data
sets where it will be done in random permutation

Why we have to random permutation?
The image should not repeat, So to avoid the repeated labels we do random permutations 

This code clearly represents for plotting the digits and printing the corresponding labels Where x_matrix is the data and label and index
says where the data is present Cmap is the function which is used for visualization of image and........matrix of values
that define the colors for graphics objects such as surface, image, and patch objects Interpolation = “ nearest “ simply displays an
image without trying to interpolate between pixels if the display resolution is not the same as the
image resolution
The above code is just functions
for example
we are taking 100th image x train and 40000th image as x test when we run each time the image
will be changing because of random permutation here in example x train is seen as 9 and y train
is label mention it as 9 but in x test the image is not so clear to identity when compared to x train
but the y test label identifies as 5

#Support Vector Machine
First half all we have understand what type of problem will SVM solves with respect to supervised data. it is useful in solving both
classification and regression problems The main objective of Support vector machine. Is to identify an optimal separating hyperplane which
maximizes the margin between different classes of the training data. Before going to the objective we have to know what is support vector
hyper plane, Marginal distance.
Hyper plane : are decision boundaries that help classify the data points
Support vectors : the data sets which are above the marginal plane
Marginal distance : is the distance from the hyperplane (solid line) to the closest points in either class
SVM is been divided into further multiple parts
that is Linear, Polynomial , RBF.

#Linear SVM 
is the algorithm for solving multiclass classification problems from large data sets that implements an original proprietary version of a
cutting plane algorithm for designing a linear support vector machine.
Linear Kernel is used when the data is Linearly separable, that is, it can be separated using a single Line. It is one of the most common kernels to be
used. It is mostly used when there are a Large number of Features in a particular Data Set.

Why cross validation is required?
Cross Validation is a very useful technique for assessing the effectiveness of your model, particularly in cases where you need to
makeoverfitting. It is also of use in determining the hyper parameters of your model, in the sense that which parameters will result in lowest test error.

In the figure, the data set is split into 5 folds. In the first iteration, the first fold is used to test the model and the rest are used to train the model. In
the second iteration, 2nd fold is used as the testing set while the rest serve as the training set. This process is repeated until each fold of the 5 folds
have been used as the testing set.

C Parameter
C- It is a hypermeter in SVM to control error.
What does that mean to control error or margin?
Let’s understand with visualization in figure and I will show c values I given in code the C parameter tells the SVM optimization how
much you want to avoid misclassifying each training example. For large values of C, the optimization will choose a smaller-margin
hyperplane if that hyperplane does a better job of getting all the training points classified correctly.

Like ex : 0.1, 1, 10, 100
You can see in figure if we have low C means low error and if we have large C means large error.In low C we have only one error but in case of large
C, we have four errors.

GridSearchCV implements a “fit” and a “score” method. It also implements “score_samples”, “predict”, “predict_proba”, “decision_function”,
“transform” and “inverse_transform” if they are implemented in the estimator used.

polynomial kernel is a kernel function it is used with support vector machines “Kernel” is used due to set of mathematical
functions used in Support Vector Machine
Examples like degree = 2, 3, 4

What is the difference between 2, 3
If the degree is more then the predicted means collection of data is more than the accuracy will increases if we use more values for degree in polynomial kernel

Gamma is used when we use the Gaussian RBF kernel if you use linear then you do not need gamma only you need C hypermeter
Gamma is a hyper parameter which we have to set before training model. Gamma decides that how much curvature we want in a decision boundary means coves
the data points in plane Gamma high means more curvature. If the more curvature is occur in plane then the collection of data points is more.
For this reason will get more accuracy in sum model Gamma low means less curvature.

So the question is when we should use high or low gamma?
The answer is it totally depends upon data to data.

Accuracy in rbf is multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match
the corresponding set of labels in prediction of model is true.

#Results
Here we compare the linear, polynomial and rbf kernals to find the best accuracy in svm model By comparing all these 3
Linear is having 83%
Polynomial is 85.59%
RBf is 86.6%
Of accuracy so RBF is the best among all three
