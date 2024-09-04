using Distributions, StatsBase, Random, Plots, StatsPlots, PrettyTables

Random.seed!(1249)

x = [-0.51, 1.86, 0.91, 1.42, 0.87, 1.23, 0.02, 0.28, 0.14, 2.24, 2.63, 
-0.02, 0.83, -0.4, 2.33, -0.56, 1.27, 0.16, 0.6, 2.41]
n = length(x)
mean(x)
std(x)
sample_mean = std(x)/sqrt(n)
# with uniform priors for mu and sigma, and a normal likelihood for the 
# sample, we get a Student t distribution.
mu = -5.0:0.1:5.0
v = n-1
s_sq = var(x)
p_mu_x = (1.0.+(n*(mu.-mean(x)).^2)/(v*s_sq)).^(-n/2)
mus = collect(mu)
plot(mus, p_mu_x)
mu_draws = rand(TDist(n-1), 10^6).*(sqrt(s_sq)/sqrt(n)) .+ mean(x)
plot(mu_draws, st=:density)

y = [0.37, 1.52, 2.46, 2.89, 2.17, 1.86, 1.96, 0.06, 2.07, 7.06, 12.74, 
4.04, 3.09, 4.13, 5.16, 0.24, 4.9, 2.07, 0.8, 5.39]


function post_t(x)
    n_x = length(x)
    mn_x = mean(x)
    sd_x = std(x)
    x_draws = rand(TDist(n_x-1), 10^6).*(sd_x/sqrt(n_x)) .+ mn_x
    return x_draws, mn_x, sd_x, n_x
end

y_draws, mn_y, sd_y, n_y = post_t(y)
x_draws, mn_x, sd_x, n_x = post_t(x)

plot(y_draws, st=:density, label="posterior for mean of y")
plot!(x_draws, st=:density, label="posterior for mean of x")

# We can compute the posterior density for the DIFFERENCE of the 
# means by looking at the difference of the MC draws!
diff_yx = y_draws .- x_draws
plot!(diff_yx, st=:density, label="posterior for difference of means")
vline!([0.0], label=false, linecolor=:black, linewidth=2)

# Compute a "Bayesian p-value"
pval = length(diff_yx[diff_yx .< 0.0])/length(diff_yx)

# Ratio of means?
ratio_yx = x_draws ./ y_draws
plot(ratio_yx, st=:density, xlims=(-10, 10), label="posterior for ratio of means")

using LinearAlgebra
function bayesreg(y,x)
  n = length(y)
  k = size(x,2)
  X = [ones(n) x]
  # uninformative analytical
  iXX = inv(X'*X)
  bhat = iXX*X'*y  # bhat = X \ y
  s2hat = (y - X*bhat)'*(y - X*bhat)/n
  Vb = s2hat.*iXX
  seb = sqrt.(diag(Vb))

  println("linear model results:")
  println("coeff estimates ", bhat)
  println("standard errors ",seb)
  println("s^2 (eqn. variance) = ",s2hat)

  # compute R^2
  tss = sum((y .- mean(y)).^2)
  R2 = 1 - s2hat*n/tss
  println("Rsquared = ",R2)
  yhat = X*bhat

  return bhat, seb, s2hat, Vb, R2, yhat
end

bhat, seb, s2hat, Vb, R2, yhat = bayesreg(y,x)
bhat[2]
seb[2]

that = bhat[2]/seb[2]

function marg_post_mu(b,seb,v; M = 10^6)
    vs2 = v*seb^2
    ts = b .+ seb.*rand(TDist(v),M)
    return ts
  end

v_b = length(y) - 2
b_draws = marg_post_mu(bhat[2], seb[2], v_b)
plot(b_draws, st=:density)