# ECON 9011 Assignment 1
## Author: Ang Zhang

### 1. Give an equation for the likelihood function of an AR(1) model.

An AR(1) model can be specified as: $X_{t+1} = \alpha + \phi*X_t + \epsilon_t$, where $\epsilon \sim N(0,\sigma^2)$. Assue no drift, i.e. $\alpha = 0$. Assuming IID of x, Then $ X \sim N(\phi x_{t-1}, \sigma^2)$. Then the likelihood function:
$$
L = P(x|\phi, \sigma) = \frac{1}{(\sqrt{2\pi\sigma^2})^n}exp(-\frac{\sum{(x_i - \phi x_{i-1})^2}}{2\sigma^2})
$$
### 2. Write down suitable (relatively) uninformative priors for an AR(1) model (give equations). 
$\phi$ could be anywhere on the real line. $\sigma^2$ would be greater than 0. Suppose N is a relative large number, So:
$$
p(\phi) \propto 1, -N<\phi<N
$$
$$
p(\sigma^2) \propto 1, 0<\sigma^2<N
$$

### 3. Using the above likelihood and priors, give an equation for each of the conditional posterior densities of each parameter of an AR(1) model up to a constant of proportionality. 

From our prior, both parameters are propostional to 1 in their respective range, so the posterior density would be:
$$
p(\phi, \sigma^2 | x) \propto Lp(\phi)p(\sigma^2)\propto L
$$
So, for conditional posterior density of $\phi$ we treat $\sigma^2$ as constant:
$$
p(\phi|\sigma^2,x) \propto \frac{1}{(\sqrt{2\pi\sigma^2})^n}exp(-\frac{\sum{(x_i - \phi x_{i-1})^2}}{2\sigma^2}) \propto \frac{1}{(\sqrt{2\pi\sigma^2})^n}exp(-\frac{\sum{(\phi - something)^2}}{2\sigma^2})
$$
this is also a Normal distribution.
For conditional posterior density of $\sigma^2$ we treat $\phi$ as constant:
$$
p(\sigma^2|\phi,x) \propto \frac{1}{(\sqrt{2\pi\sigma^2})^n}exp(-\frac{\sum{(x_i - \phi x_{i-1})^2}}{2\sigma^2}) \propto (\sigma^2)^{-2/n}exp(-\frac{something}{\sigma^2})
$$
which looks like a Inverted Gamma distribution.
### 4. Based on the above, write down, in detail, an MCMC algorithm for conducting posterior inference for an AR(1) model. 

Step 1: Start with uniformative starting value of $\phi^{(0)}$  
Step 2: Plug in the $\phi$ value into the conditional distribution of $\sigma^2$, then draw a value for $\sigma^2$ from that distribution.  
Step 3: Plug in the $\sigma^2$ value into the conditional distribution of $\phi$, then draw a value for $\phi$ from that distribution.  
Step4: Repeat above steps forever (seriously).  
Then we get a distribution of the parameters $\phi$ and $\sigma^2$.

### 5. What is a suitable prior to enforce a stationarity assumption in an AR(1) model, i.e., impose the constraint(s) necessary ensure the variable is stationary? 
For AR(1) model to be stable we have $-1<\phi<1$, then the prior can be:
$$
p(\phi) \propto 
\begin{cases}
    1, -1<\phi<1 \\
    0, all\ else
\end{cases}
$$
### 6. Using Julia, generate pseudo-data for two variables, 100 observations, each from an AR(1) DGP; one stationary, one nonstationary (you can experiment with different parameter values), and provide a time plot for each one. 


```Julia
phi = 0.5
alpha = 0
z = zeros(n+20)
for t = 2:n+20
    z[t] = alpha + phi*z[t-1] + randn(1)[1]
end
y = z[21:end]
plot(y)
phi = 1.5
alpha = 0
z = zeros(n+20)
for t = 2:n+20
    z[t] = alpha + phi*z[t-1] + randn(1)[1]
end
y = z[21:end]
plot(y)

```

### 7. Using Julia and you data generated in qu. 6, estimate an AR(1) model (i) using Bayesian HMC/MCMC methods in Turing.jl, and also (ii) using Gibbs sampling (not via Turing.jl). Provide estimation summary statistics and plots of the posterior densities along with plot and/or information about the prior distributions used. 

Since AR(1) model taken on the same form as a linear model if we set $Y=x$ and $X=x_{t-1}$, we can utilize the *gsreg()* function provided in *gsreg.jl* to do the magic. Since we have to use the first observation as the value for $x_0$, the actual size of the data will be 99. Code as below: 

```Julia
yt = y[2:end]
yt1 = y[1:end-1]
X = [ones(n-1) yt1]
bdraws,s2draws = gsreg(yt,X)

plot(bdraws[:,2], st=:density, fill=true, alpha=0.5, title = "phi posterior", label= "uninformative" )
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])
```
