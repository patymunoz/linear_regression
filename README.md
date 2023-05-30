# Linear regression

[![Binder]()

## Introduction

This is an application of lineal simple and multiple regression to analize a problem. 

The data used in this example is from scikit-learn library "7.13. Diabetes dataset" Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499. (https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)

We will see the basics of regression. We will see the difference between simple and multiple regression, how to interpret the results, how to evaluate the model and how to use it to predict new values.

In addition, we will see how to use the [scikit-learn library](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to perform the regression and how to use the [statsmodels library](https://www.statsmodels.org/stable/regression.html) to perform the regression. Even more, we can see that statsmodels OLS provides more information for statistical inference and probabilistic interpretation of its results.

## Dependencies

* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
* [scipy](https://www.scipy.org/)
* [seaborn](https://seaborn.pydata.org/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)
* [scikit-learn](https://scikit-learn.org/stable/index.html)

## How to run the code

It is recommended to run the code in a virtual environment.

1. To create a virtual environment, I recommend:

```
python install -r requirements.txt
```

But you can install the libraries manually if you want. Only be sure to install the libraries in the same version as the ones in the requirements.txt file and use **python 3.9**.

2. Run the code

There are *n* notebooks that you can run:







## 1. Introduction to Regression

*Linear regression* is an approach for predicting a quantitative response $Y$ on the basis of a single predictor variable $X$. 

Some important point to consider in this approach are: 

* that it assumes that there is a relationship between $X$ and $Y$.
* that it assumes that this relationship is linear.

Mathematically, we can write this linear relationship as

$$
\begin{equation} Y \approx \beta_0 + \beta_1 X + \varepsilon \end{equation}
$$

where $\beta_0$ and $\beta_1$ are two unknown constants that represent the *intercept* and *slope* terms in the linear model.

* $\beta_0$ is the *intercept term* (the expected value of $Y$ when $X$ = 0)

* $\beta_1$ is the *slope term* (the average increase in $Y$ associated with a one-unit increase in $X$).

Together, $\beta_0$ and $\beta_1$ are known as the *model coefficients* or *parameters*. Usually in some literature, $\beta_0$ and $\beta_1$ are instead denoted as $\theta$. 


## 2. For estimating coefficients / parameteres

In practice, $\beta_0$ and $\beta_1$ are unknown. So before we can use (1) to make predictions, we must **use data** to estimate the coefficients. 

We want to find an intercept $\beta_0$ and a slope $\beta_1$ such that the resulting line is as close as possible to the *n* data points. There are a number of ways of measuring *closeness*. Howerver, by far the most common approach involves *minimizing the least squares criterion* which is called *least squares regression* or *ordinary least squares (OLS)* regression.

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



## 3. For modeling the error ($\varepsilon$ - noise term)

Because of the presence of *observation noise*, we will adopt a **probabilistic approach** and model the noise using likelihood function. We consider a regression problem with the likelihood function:

$$
\begin{equation} p(y|x) = \mathcal{N}(y; f(x), \sigma^2) \end{equation}
$$

where $x \in R$ are inputs and $y \in R$ are noisy function values (targets).

With (6), the functional relationship between $x$ and $y$ is given as

$$
\begin{equation} y = f(x) + \varepsilon \end{equation}
$$

where $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ is independent, identically distributed (i.i.d.) Gaussian measuremente noise with mean 0 and variance $\sigma^2$ (Deisenroth et al,. 2020).

Assumes:

* that $\varepsilon$ is a mean-zero random error term that is independent of $X$. This means that $\varepsilon$ -error term, usually called *noise* - is assumed to be generated by an independent and identically distributed (i.i.d.) random variable, meaning each noise sample is independent of the others and they all come from the same distribution.

## 4. Problem formulation - Linear regression model in a probabilistic framework

Linear regression is given by:

$$
\begin{equation} p(y| x, \theta) = \mathcal{N}(y; x^T \theta, \sigma^2) \leftrightarrow y=x^T\theta+\varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)\end{equation}
$$




# References

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). An Introduction to Statistical Learning with Applications in R. Springer.

Deisenroth, M. P., Faisal, A. A.,, Ong, C. S. (2020). Mathematics for Machine Learning. Cambridge University Press.
