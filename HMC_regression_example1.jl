
# regression using HMC in Turing.jl example code

# set path to working directory

# Windows
#cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2024")

# Mac
#cd("/Users/millsja/Library/CloudStorage/OneDrive-UniversityofCincinnati/Class/9011 2024")

# load packages
using Distributions, Plots, StatsPlots, Random, Turing


# Read in data here, or generate pseudo-data as below:
## AR(1) DGP
n = 100
phi = 0.8
Random.seed!(42)

z = zeros(n+20)
for t = 2:(n+20)
    z[t] = 0.0 + phi*z[t-1] + randn(1)[1]
end
y = z[21:(n+20)]
plot(y, label="y")  # timeplot

############################################### 

include("useful_functions_mcmc.jl")  # some useful functions and example Turing model statements


# Turing.jl model statement:
@model simple_regression(y,X, ::Type{TV}=Vector{Float64}) where {TV} =begin
    n, D = size(X)
    #alpha ~ Normal(0,1)
    sig ~ Uniform(0.01,10)
    #m ~ Truncated(Normal(-2,3),-999.9,0.999)  # truncating a distribution
    beta = TV(undef,(D))
    # sd<10 too restrictive for beta coeffs 
    for k in 1:(D)
        beta[k] ~ Normal(0, 20.0)
    end
    #delta ~ Normal(0,3.0)
 #   mu = logistic.(Matrix(X) * beta) # logit model specification
    for i in 1:n
        y[i] ~ Normal(X[i,:]'*beta, sig)
    end
end

yt = y[2:n]
yt1 = y[1:(n-1)]

cor(yt, yt1)
acf = autocor(y)
plot(acf, st=:bar, title = "ACF of y", label = false)
X = [ones(n-1) yt1]

model = simple_regression(yt, X)
Turing.setprogress!(true)
iter = 6000
@time cc = sample(model, NUTS(0.65),iter)

cc

plot(cc)

# one parameter (on MCMC chain) 
sigma = get(cc,:sig)
sigma_est = Array(sigma.sig)
mean(sigma_est)
plot(sigma_est, st=:density, fill=true, alpha=0.5, label = "mean = $(round(mean(sigma_est),digits=3))")
sigmav = sigma_est[:,1]
h1 = hpdi(sigmav)
h1
c1 = quantile(sigmav,[0.025,0.975])
vline(h1, label = "hpdi")
std(sigma_est)
plot(sigma_est, st=:density)

# several (k) parameters (MCMC chains) in a vector
k = 2
b = get(cc,:beta)
bs = zeros(iter,k)
for i in 1:k
    bs[:,i] = Array(b.beta[i])
end

bs

coefnames = ["alpha" "phi"]
j = 2 # column of bs to plot

summary_mcmc_array(bs,coefnames)
mnb = round.(mean(bs[:,j]), digits = 3)
plot(bs[:,j], st=:density, fill=true, alpha=0.5, label = "mean = $mnb", title = "beta $j")

median(bs[:,j])
prob_int_99 = hpdi(bs[:,j], alpha = 0.01)
vline!(prob_i, label=false, linecolor = :black)

# experiment with narrowest interval that gives a sensible result and/or
# increase the number of iterations to allow a narrower interval
mode_j = hpdi(bs[:,j], alpha = 0.998)

# hypothesis testing
null = 0.6
bj = bs[:,j]
#tol_int = 0.0001  # 1/2 (one side) interval to evaluate area under density

function post_density_ratio(bj, null; tol_int = 0.001)
    # intervals to evaluate area under curve:
    mode_j = hpdi(bj, alpha = 0.998)
    null_u = null + tol_int
    null_l = null - tol_int
    mode_u = mean(mode_j) + tol_int
    mode_l = mean(mode_j) - tol_int
    #check
    #mode_u - mode_l
    # compute areas under the curve
    p_gt_null = length(bj[bj .>= null_u])/iter
    p_le_null = length(bj[bj .>= null_l])/iter
    prob_null = abs(p_le_null - p_gt_null)
    p_gt_mode = length(bj[bj .>= mode_u])/iter
    p_le_mode = length(bj[bj .>= mode_l])/iter
    prob_mode = abs(p_le_mode - p_gt_mode)
    # compute odds against the null
    post_odds = round(prob_mode / prob_null, digits = 4)
    # compute prob < null and prob > null
    prob_le_null = p_le_null = length(bj[bj .<= null])/iter
    prob_gt_null = 1 - prob_le_null
    return post_odds, prob_le_null, prob_gt_null
end


post_odds, prob_le_null, prob_gt_null = post_density_ratio(bj, null)
post_odds, prob_le_null, prob_gt_null = post_density_ratio(bj, null, tol_int = 0.02)

null = 0.8
post_odds, prob_le_null, prob_gt_null = post_density_ratio(bj, null)
post_odds, prob_le_null, prob_gt_null = post_density_ratio(bj, null, tol_int = 0.0001)

null = 1.0
post_odds, prob_le_null, prob_gt_null = post_density_ratio(bj, null)
post_odds, prob_le_null, prob_gt_null = post_density_ratio(bj, null, tol_int = 0.01)

mcodds(bj, h0 = null)
n, k = size(X)
todds(mean(bj), std(bj), (n-k), h0 = null)

1+1  # if this doesn't equal 2, you could be in an alternative universe.

z = rand(Gamma(2, 5), 10^6)
plot(z)