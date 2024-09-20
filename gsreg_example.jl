# gsres function example

## data here:
using Distributions, Plots, StatsPlots
n = 100
x = randn(n)
y = 1.0 .+ 1.0.*x .+ 0.2.*randn(n)
plot(x,y,st=:scatter)

phi = 0.5
alpha = 0
z = zeros(n+20)
for t = 2:n+20
    z[t] = alpha + phi*z[t-1] + randn(1)[1]
end
y = z[21:end]
plot(y)

phi = 1.2
alpha = 0
z = zeros(n+20)
for t = 2:n+20
    z[t] = alpha + phi*z[t-1] + randn(1)[1]
end
y = z[21:end]
plot(y)

phi = 1.0
alpha = 0
z = zeros(n+20)
for t = 2:n+20
    z[t] = alpha + phi*z[t-1] + randn(1)[1]
end
y = z[21:end]
plot(y)

include("gsreg.jl")

yt = y[2:end]
yt1 = y[1:end-1]



## gsreg(y::Array{Float64},X::Array{Float64}; M=10000::Int64, burnin = Int(floor(M/10.0))::Int64,tau=[1.0]::Array{Float64},iB0=[0.0001]::Array{Float64},b0=[0.0]::Array{Float64},d0=0.0001::Float64, a0=3.0::Float64)
# uninformative prior
X = [ones(99) yt1]
bdraws,s2draws = gsreg(yt,X)

plot(bdraws[:,2],st=:density)
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])

plot(s2draws[:,1],st=:density)
mean(s2draws[:,1])
std(s2draws[:,1])
quantile(s2draws[:,1],[0.025,0.975])


# Informative prior (still uninformative for variance parameter)
b = [0.0; 1.0]    # prior coeff. means
iB = inv([1000.0 0.0; 0.0 0.001])  # prior cov matrix
bdraws,s2draws = gsreg(y,X, b0=b, iB0=iB)

plot!(bdraws[:,2],st=:density,linecolor=:red,label="informative")
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])

z = rand(100, 2)
plot(z[:,1],z[:,2],st=:scatter, xlim = (-0.2, 1.2), ylim = (-0.2, 1.2), label = false)
vline([0, 1.0], label = false)
hline([0, 1.0], label = false)

# using Turing and MCMC

using Turing

@model simple_regression(y,X, ::Type{TV}=Vector{Float64}) where {TV} =begin
    n, D = size(X)
    #alpha ~ Normal(0,1)
    sig ~ Uniform(0.01,10)
    #m ~ Truncated(Normal(-2,3),-999.9,0.999)
    beta = TV(undef,(D))
    # sd<10 too restrictive for beta coeffs 
    for k in 1:(D)
        beta[k] ~ Normal(0, 20.0)
    end
    #delta ~ Normal(0,3.0)
 #   mu = logistic.(Matrix(X) * beta)
    for i in 1:n
        y[i] ~ Normal(X[i,:]'*beta, sig)
    end
end

model = simple_regression(y, X)
Turing.setprogress!(true)
@time chain = sample(model, NUTS(0.65), 1000)
b_draws = get(chain, :beta)
s2_draws = get(chain, :sig)
sigma_est = Array(s2_draws.sig)
plot(sigma_est, st=:density, fill=true, alpha=0.5, title = "σ squared posterior", label= false )

b = get(chain, :beta)
s = length(s2_draws.sig)
k = 2
bs = zeros(s, k)
for i in 1:X
    bs[i, :] = b[i].beta
end
