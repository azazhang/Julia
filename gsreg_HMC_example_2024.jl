# gsres function example
# cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2022")

## data here:
using Distributions, Plots, StatsPlots
using StatsBase
n = 1000
x = randn(n)
y = 1.0 .+ 0.5.*x .+ 0.4.*randn(n)
plot(x,y,st=:scatter)

## AR(1) DGP
phi = 0.5
z = zeros(n+20)
for t = 2:(n+20)
    z[t] = 0.0 + phi*z[t-1] + randn(1)[1]
end
y = z[21:(n+20)]
plot(y)

z = rand(Gamma(2, 5), 1000)
plot(z)
plot(z, st = :density)
pz95 = quantile(z,(0.025,0.975))
vline!(pz95, linecolor = "green", label = false)

ya = autocor(y)
pya = pacf(y, 1:20)
plot(pya, st = :bar, label = "AR(1) with phi = $phi")



############################################### 


include("gsreg.jl")

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
bdraws,s2draws = gsreg(yt,X)

plot(bdraws[:,2], st=:density, fill=true, alpha=0.5, title = "phi posterior", label= "uninformative" )
mean(bdraws[:,2])
std(bdraws[:,2])
quantile(bdraws[:,2],[0.025,0.975])

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

model = simple_regression(yt, X)
Turing.setprogress!(true)
@time cc = sample(model, NUTS(0.65),3000)

cc

plot(cc)

summary_mcmc_array(bs,coefnames)

# one parameter (on MCMC chain) 
sigma = get(cc,:)
sigma_est = Array(sigma.sig)
mean(sigma_est)
std(sigma_est)
plot(sigma_est, st=:density)
sigma = get(cc,:sig)
sigma_est = Array(sigma.sig)
mean(sigma_est)
std(sigma_est)
plot(sigma_est, st=:density)

# several (k) parameters (MCMC chains) in a vector
s = length(sigma_est)
k = 2
b = get(cc,:beta)
bs = zeros(s,k)
for i in 1:k
    bs[:,i] = Array(b.beta[i])
end

bs

plot(bs[:,2], st=:density, fill=true, alpha=0.5)
mean(bs[:,2])

##### Estimated y_0 in an AR(1) model:
@model ar1_with_y0(y, ::Type{TV}=Vector{Float64}) where {TV} =begin
    n = length(y)
    D = 2
    # priors
    y0 ~ Normal(y[1],std(y[1:5])*1.5)  # reasonable prior for y0?
    sig ~ Uniform(0.01,10)
    beta = TV(undef,(D))
    for k in 1:(D)
        beta[k] ~ Normal(0, 20.0)
    end
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

