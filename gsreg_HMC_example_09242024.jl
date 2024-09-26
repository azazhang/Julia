# gsres function example
cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2024")

## data here:
using Distributions, Plots, StatsPlots, StatsBase, CSV, DataFrames

using CSV, DataFrames, Plots, StatsPlots, Distributions, LinearAlgebra
using Turing, StatsFuns, Statistics #, PrettyTables

n = 100
x = randn(n)
y = 1.0 .+ 0.5.*x .+ 0.4.*randn(n)
plot(x,y,st=:scatter)

## AR(1) DGP
n = 100
ϕ = 1.0
α = 0.01
z = zeros(n+20)
for t = 2:(n+20)
    z[t] = α + ϕ*z[t-1] + randn(1)[1]
end
y = z[21:(n+20)]
plot(y)



z = rand(Gamma(2,5), 1000)
plot(z)
pz95 = quantile(z,[0.025,0.975])
hline!(pz95, linecolor = "green", label=false)
hline!([mean(z)], label=false, linecolor="black")
mean(z)
std(z)

plot(z, st=:density)

pz95 = quantile(z,[0.025,0.975])
vline!(pz95, linecolor = "green", label=false)


z1 = cumsum(z)

plot(z1)

mean(z1[1:500])
mean(z1[501:end])

std(z1[1:500])
std(z1[501:end])

plot(z1, st=:density)
plot(z1, st=:histogram)
[z z1]


za = autocor(z)
plot(za, st=:bar, label="white noise")

z1a = autocor(z1)
plot(z1a, st=:bar, label="I(1)")

ya = autocor(y)
plot(ya, st=:bar, label="AR(1) with ϕ = $phi")





z2 = cumsum(z1)
plot(z2)

dz2 = diff(z2)
plot(dz2)
plot!(z1, linestyle = :dash)

d2z2 = diff(dz2)
plot(d2z2)
plot!(z, linestyle = :dash)


# generate a trend stationary variable
T = 200
t = 1:T
a = 0.5
b = 0.2
zz = a .+ b.*t + randn(T)
plot(zz)

# zz is nonstationary
azz = autocor(zz)
plot(azz, st=:bar)


# deterministic trend, different functional form:
zzl = exp.(a .+ b.*t + randn(T))
plot(zzl)
lzz1 = log.(zzl)
plot(lzz1)

# not working?!
pya = StatsBase.pacf(z1, 20)
plot(pya, st=:bar, label="AR(1) with ϕ = $phi", title = "PACF")




############################################### 

include("gsreg.jl")

include("useful_functions_mcmc.jl")

## gsreg(y::Array{Float64},X::Array{Float64}; M=10000::Int64, burnin = Int(floor(M/10.0))::Int64,tau=[1.0]::Array{Float64},iB0=[0.0001]::Array{Float64},b0=[0.0]::Array{Float64},d0=0.0001::Float64, a0=3.0::Float64)
# uninformative prior
b = [0.0; 0.0]    # prior coeff. means
iB = inv([0.0001 0.0; 0.0 0.0001]) 

X = [ones(n) x]
bdraws,s2draws = gsreg(y,X)

plot(bdraws[:,2], st=:density, fill=true, alpha=0.5, title = "β posterior", label= "uninformative" )
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])

plot(bdraws[:,1], st=:density, fill=true, alpha=0.5, title = "α posterior", label= "uninformative" )
mean(bdraws[:,1])
std(bdraws[:,1])
quantile(bdraws[:,1],[0.025,0.975])

# Informative prior (still uninformative for variance parameter)
b = [0.0; 1.0]    # prior coeff. means
iB = inv([1000.0 0.0; 0.0 0.001])  # prior cov matrix
bdraws,s2draws = gsreg(y,X, b0=b, iB0=iB)

plot!(bdraws[:,2],st=:density,linecolor=:red,label="informative")
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])


plot(s2draws, st=:density, fill=true, alpha=0.5, title =  "σ-squared posterior", label= false )



z = randn(1000,2).*0.3 .+ 0.5
plot(z[:,1], z[:,2], st = :scatter) #, xlims = (-0.2,1.2), ylims = (-0.2,1.2), label=false)
#vline!([0.0 1.0], label=false)
#hline!([0.0 1.0], label=false)
# correlated variables
z3 = 0.5*z[:,1] .+ randn(1000).*0.2
cor(z[:,1], z3)
plot(z[:,1], z3, st = :scatter, xlabel = "z1", ylabel="z3") 


# AR(1) model estimation

# uninformative prior
b = [0.0; 0.0]    # prior coeff. means
iB = inv([0.0001 0.0; 0.0 0.0001]) 


yt = y[2:n]
yt1 = y[1:(n-1)]

X = [ones(n-1) yt1]
bdraws,s2draws = gsreg(yt,X, M = 30000)

plot(bdraws[:,2], st=:density, fill=true, alpha=0.5, title = "phi posterior", label= "uninformative" )
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])
coefnames = ["α" "ϕ"]
summary_mcmc_array(bdraws,coefnames)


# try using HMC in Turing.jl to do this
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


@model informative_prior_model(y,X, ::Type{TV}=Vector{Float64}) where {TV} =begin
    n, D = size(X)
    #alpha ~ Normal(0,1)
    sig ~ Uniform(0.01,10)
    beta = TV(undef,(D))
    # sd<10 too restrictive for beta coeffs 
    #for k in 1:(D)
    beta[1] ~ Normal(0, 20.0)
    beta[2] ~ Truncated(Normal(0.0,0.001),0.0,1.1)
    #end
    #delta ~ Normal(0,3.0)
 #   mu = logistic.(Matrix(X) * beta)
    for i in 1:n
        y[i] ~ Normal(X[i,:]'*beta, sig)
    end
end


model = simple_regression(yt, X)
model = informative_prior_model(yt, X)
Turing.setprogress!(true)
iter = 6000
@time cc = sample(model, NUTS(0.65),iter)

cc

plot(cc)


# one parameter (on MCMC chain) 
sigma = get(cc,:sig)
sigma_est = Array(sigma.sig)
mean(sigma_est)
std(sigma_est)
plot(sigma_est, st=:density)

# several (k) parameters (MCMC chains) in a vector

k = 2
b = get(cc,:beta)
bs = zeros(iter,k)
for i in 1:k
    bs[:,i] = Array(b.beta[i])
end

summary_mcmc_array(bs,coefnames)
# bs2 = bs[:,j]
j = 2
plot(bs[:,j], st=:density, fill=true, alpha=0.5, label="HMC posterior") #, xlims=(-0,1.2), grid=false)
plot!(bdraws[:,j], st=:density, fill=true, alpha=0.5, label="Gibbs posterior")

prior_HMC = rand(Normal(0, 20.0),100000)
plot!(prior_HMC, st=:density, fill=true, alpha=0.5, label="HMC prior")

prior_Gibbs = rand(Normal(0, sqrt(1000.0)),100000)
plot!(prior_Gibbs, st=:density, fill=true, alpha=0.5, color=:red, label="Gibbs prior")
#savefig("posteriors_priors_qu6")


# informative prior for phi?
ϕ_draws = rand(Truncated(Normal(0.0,0.001),0.0,1.1), 100000)
plot!(ϕ_draws, st=:density, fill=true, alpha=0.5, color=:green, label="Informative prior?")
plot!(bs[:,j], st=:density, fill=true, alpha=0.5, color=:blue, label="HMC posterior", xlims=(-0,1.2), grid=false)


# BAD informative prior
ϕ_draws = rand(Truncated(Normal(0.0,0.1),0.0,1.1), 100000)
plot(ϕ_draws, st=:density, fill=true, alpha=0.5, color=:green, label="Informative prior?")

##### Estimated y_0 in an AR(1) model:
@model ar1_with_y0(y, ::Type{TV}=Vector{Float64}) where {TV} =begin
    n = length(y)
    D = 2
    # priors
    y0 ~ Normal(y[1],std(y[1:5])*1.5)  # reasonable prior for y0?
    sig ~ Uniform(0.01,10)
    beta = TV(undef,(D))
    # for k in 1:(D)
    #     beta[k] ~ Normal(0, 20.0)
    # end
    beta[1] ~ Normal(0, 20.0)
    beta[2] ~ Truncated(Normal(0, 10), -0.9999, 0.9999)
    # likelihood
    for i in 1:n
        if i == 1
            y[i] ~ Normal((beta[1] + beta[2]*y0), sig)
        else
        y[i] ~ Normal(beta[1] + beta[2]*y[i-1], sig)
        end
    end
end


model = ar1_with_y0(y)
Turing.setprogress!(true)
@time cc = sample(model, NUTS(0.65),3000)

cc

plot(cc)

### need to get draws from cc to do:
coefnames = ["y0" "alpha" "phi" "sig"]
summary_mcmc_array(bs,coefnames)

#################################

th = get(cc,:θ)
th_est = Array(th.θ)[burn+1:end]
mean(th_est)

# theta HMC draws
plot(th_est)

# theta posterior
plot(th_est, st=:density, label = "theta posterior")
vline!([theta], label="true theta")


ts = zeros(s-burn,n_ind)
for i in 1:size(ts,2)
    ts[:,i] = Array(t.t[i])[burn+1:end]
end

summary_mcmc_array(bs,coefnames)

