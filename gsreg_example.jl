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

include("gsreg.jl")

yt = y[2:end]
yt1 = y[1:end-1]



## gsreg(y::Array{Float64},X::Array{Float64}; M=10000::Int64, burnin = Int(floor(M/10.0))::Int64,tau=[1.0]::Array{Float64},iB0=[0.0001]::Array{Float64},b0=[0.0]::Array{Float64},d0=0.0001::Float64, a0=3.0::Float64)
# uninformative prior
X = [ones(n) x]
bdraws,s2draws = gsreg(y,X)

plot(bdraws[:,2],st=:density)
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])


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