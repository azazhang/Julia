# ECON 9011 Assignment 1
## Author: Ang Zhang

## 1. Give an equation for the likelihood function of an AR(1) model.

An AR(1) model can be specified as: $X_(t+1) = \phi*X_t + \epsilon_t$, where $\epsilon \sim N(0,\sigma^2)$. Then: 
$$
L = P(x|\phi, \sigma) = \frac{1}{\sqrt(2\pi\sigma^2)}e
$$

```Julia
print("Hello, World")
