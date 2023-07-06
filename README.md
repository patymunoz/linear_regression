# Linear regression

[![Binder]()

## Introduction

Welcome! This repository is dedicated to exploring linear regression - both simple and multiple. It is a tool for analyzing a particular set of problems. 

My example data comes from the [scikit-learn library's](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  "7.13. Diabetes dataset". This dataset is well-documented in the paper "Least Angle Regression" by Bradley Efron, Trevor Hastie, Iain Johnstone, and Robert Tibshirani (2004) [available here.](https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf) 

I'll be using two Python libraries for our analysis - scikit-learn for implementing the regression, and [statsmodels library](https://www.statsmodels.org/stable/regression.html) for a detailed statistical analysis of the results.

Really hope that this exploration will be useful to understand the basics of linear regression and how to implement it in Python.

## Dependencies

To replicate the analysis, you'll need the following libraries:

* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
* [scipy](https://www.scipy.org/)
* [seaborn](https://seaborn.pydata.org/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)
* [scikit-learn](https://scikit-learn.org/stable/index.html)

## Running the code

It's recommended to run the code in a virtual environment. Follow these steps:

Set up a virtual environment and install necessary libraries with:

```
pip install -r requirements.txt
```

or first install pip-tools and create a virtual environment with:

```
pip-sync
```

Make sure you're using Python 3.9 and the library versions specified in the requirements.txt file.

Run the Jupyter notebooks available in this repository.


## 1. Understanding linear regression

### 1.1. What is linear regression?

*Linear regression* is an approach for predicting a quantitative response $Y$ on the basis of a single predictor variable $X$. 

Some important points to consider in this approach are: 

* that it assumes that there is a relationship between $X$ and $Y$.
* that it assumes that this relationship is linear.

Mathematically, we can write this linear relationship as

$$
\begin{equation} Y \approx \beta_0 + \beta_1 X \end{equation}
$$

where $\beta_0$ and $\beta_1$ are two unknown constants that represent the *intercept* and *slope* terms in the linear model.

* $\beta_0$ is the *intercept term* (the expected value of $Y$ when $X$ = 0)

* $\beta_1$ is the *slope term* (the average increase in $Y$ associated with a one-unit increase in $X$).

Together, $\beta_0$ and $\beta_1$ are known as the *model coefficients* or *parameters*. Usually in some literature, $\beta_0$ and $\beta_1$ are instead denoted as $\theta$. 


## 2. Estimating the coefficients

In practice, $\beta_0$ and $\beta_1$ are unknown and must be estimated from the data. We find these estimates by minimizing the sum of the squared residuals (differences between observed and predicted values), a method known as *ordinary least squares (OLS)*.

There are a number of ways of measuring *closeness*. Howerver, by far the most common approach involves *minimizing the least squares criterion* which is OLS (James, et al., 2013).

Let $\hat{y_i} = \hat{\beta_0} + \hat{\beta_1} x_i$ be the prediction for $Y$ based on the $i$th value of $X$. Then $e_i = y_i - \hat{y_i}$ represents the $i$th *residual*, this is the difference between the $i$th observed response value and the $i$th response value that is predicted by our linear model. We define the *residual sum of squares (RSS)* as

$$
\begin{equation} RSS = e_1^2 + e_2^2 + ... + e_n^2 \end{equation}
$$

or equivalently

$$
\begin{equation} RSS = (\overbrace{y_1}^{observed} \overbrace{- \hat{\beta_0} - \hat{\beta_1} x_1}^{predicted})^2 + (y_2 - \hat{\beta_0} - \hat{\beta_1} x_2)^2 + ... + (y_n - \hat{\beta_0} - \hat{\beta_1} x_n)^2 \end{equation}
$$

The least squares approach chooses $\beta_0$ and $\beta_1$ to minimize the RSS. Using some calculus, one can show that the minimizers are

$$
\begin{equation} \hat{\beta_1} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2} \end{equation}
$$

$$
\begin{equation} \hat{\beta_0} = \bar{y} - \hat{\beta_1} \bar{x} \end{equation}
$$

where $\bar{y} = \frac{1}{n} \sum_{i=1}^n y_i$ and $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ are the sample means (James et al,. 2013).


This is the same method used by the LinearRegression function in the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) in Python to fit a linear model to the data.

Here's how it relates to scikit-learn:

* **Creating the Model:** When you create a linear regression model in scikit-learn using LinearRegression(), you're setting up a model that will find the best fit line through your data using the least squares method, i.e., it will find the line that minimizes the residual sum of squares (RSS).

* **Fitting the Model:** When you call the .fit(X, y) method on a LinearRegression object, scikit-learn will use your input features (X) and target variable (y) to compute the optimal parameters (β0 and β1) that minimize the RSS, just as described in your content.

* **Model Coefficients:** After fitting, the estimated coefficients can be accessed using the .coef_ and .intercept_ attributes of the LinearRegression object. These correspond to β1 (the slope) and β0 (the intercept) respectively.



# References

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning with Applications in R. Springer.

Deisenroth, M. P., Faisal, A. A.,, Ong, C. S. (2020). Mathematics for Machine Learning. Cambridge University Press.
