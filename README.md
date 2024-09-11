# Logistic Regression From Scratch with Cost Computation, Gradient Descent, and Visualization

## Overview

This project implements logistic regression from scratch, including the computation of the cost function and the application of gradient descent to optimize model parameters. The dataset used is the Pima Indians Diabetes dataset, which contains information on various health metrics and diabetes outcomes. This repository demonstrates core concepts of logistic regression, cost functions, gradient descent, and how to effectively visualize these aspects.

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data](#data)
- [Computation](#computation)
  - [Cost Function](#cost-function)
  - [Gradient Descent](#gradient-descent)
- [Visualization](#visualization)

## Project Structure

- `logistic_regression.ipynb`: Jupyter notebook containing the code for logistic regression, cost function computation, gradient descent, and visualizations.
- `Diabetes.csv`: Dataset used for demonstration.

## Dependencies

Ensure the following Python libraries are installed:

- `pandas`
- `matplotlib`
- `numpy`
- `scienceplots`
- `seaborn`
- `plotly`

You can install the dependencies using:

```bash
pip install pandas matplotlib numpy scienceplots seaborn plotly
```

## Data

The dataset used is the Pima Indians Diabetes dataset, which contains information on health metrics and diabetes outcomes. The columns of interest are:

- `BloodPressure`: Blood pressure measurement (mm Hg)
- `Outcome`: Diabetes outcome (0 = No Diabetes, 1 = Diabetes)

## Computation 

### Cost Function

The cost function for logistic regression is computed using a vectorized approach:

```python
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def vectorised_cost_function(X, y, w, b):
    # Forward propagation
    z = np.dot(X, w) + b
    a = sigmoid_function(z)
    
    # Compute the cost function
    cost = -(1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    
    return cost, a
```

The cost function is defined as:

$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[y_i \log(a_i) + (1 - y_i) \log(1 - a_i)\right]$

where:
- \( w \) is the weight
- \( b \) is the bias
- \( a_i \) is the predicted probability for the \(i\)-th example
- \( m \) is the number of examples

### Gradient Descent

Gradient descent is used to minimize the cost function by iteratively updating the weight and bias:

- **Update Equations**:
  - \( w := w - \text{learning\_rate} \cdot \frac{\partial J}{\partial w} \)
  - \( b := b - \text{learning\_rate} \cdot \frac{\partial J}{\partial b} \)
    
```python
def gradient_descent(X, y, w, b, learning_rate, num_iterations):
    m = X.shape[0]
    costs = []
    w_history = []
    b_history = []
    
    for _ in range(num_iterations):
        # Forward propagation
        z = np.dot(X, w) + b
        a = sigmoid_function(z)
        
        # Compute cost
        cost, _ = vectorised_cost_function(X, y, w, b)
        costs.append(cost)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (a - y))
        db = (1/m) * np.sum(a - y)
        
        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record history
        w_history.append(w[0, 0])
        b_history.append(b)
    
    return w, b, w_history, b_history, costs
```

## Visualization

### Cost vs Weight

The cost function is plotted against a range of weight values while keeping the bias constant.

### Cost vs Bias

The cost function is plotted against a range of bias values while keeping the weight constant.

### 3D Cost Surface

A 3D surface plot shows the cost function in relation to both weight and bias, providing a comprehensive view of the optimization landscape.

### Logistic Regression Line

The final logistic regression model is visualized along with the scatter plot of the data to show the fit of the model.

### Cost vs Iterations

The cost function is plotted against the number of iterations to show how the cost decreases over time during gradient descent.

