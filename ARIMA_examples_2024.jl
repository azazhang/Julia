# gsres function example
cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2024")

## data here:
using Distributions, Plots, StatsPlots, StatsBase, CSV, DataFrames, LinearAlgebra

using Turing, StatsFuns, Statistics #, PrettyTables

# use for cointegration and spurious regression?
n = 100
x = randn(n)
y = 1.0 .+ 0.5.*x .+ 0.4.*randn(n)
plot(x,y,st=:scatter)

## AR(1) DGP
n = 500
phi = 2
z = zeros(n)
z[1] = 0.1
for t = 2:(n)
    z[t] = phi*z[t-1] + randn(1)[1]
end
#y = z[21:(n+20)]
y = z
plot(y)
mean(y)
plot(y[1:499])

n = 500
phi1 = 0.6
phi2 = 1.05
#z = zeros(n+20)
z = zeros(n)
z[1] = 0.1
z[2] = z[1] + randn(1)[1]
for t = 3:(n)
    z[t] = phi1*z[t-1] + phi2*z[t-2]  + randn(1)[1]
end
#y = z[21:(n+20)]
y = z
plot(y)
mean(y)
plot(y[1:498])


df = CSV.read("CPI.csv", DataFrame, header=true)
CPI = df.CPIAUCSL
CPI_Log = log.(CPI)
plot(CPI_Log)

dif_CPI = diff(CPI_Log)
plot(dif_CPI)

## AR(1) DGP
n = 2000
phi = 0.9
alpha = 2.0
#z = zeros(n+20)
z = zeros(n)
z[1] = 0.1
for t = 2:(n)
    z[t] = alpha + phi*z[t-1] + randn(1)[1]
end
#y = z[21:(n+20)]
y = z
plot(y)
mean(y)
plot(y[1:50])

## AR(2) DGP
n = 2000
phi1 = 0.6
phi2 = 0.4
alpha = 0.1
#z = zeros(n+20)
z = zeros(n)
z[1] = 0.1
z[2] = z[1] + randn(1)[1]
for t = 3:(n)
    z[t] = alpha + phi1*z[t-1] + phi2*z[t-2]  + randn(1)[1]
end
#y = z[21:(n+20)]
y = z
plot(y)
mean(y)
plot(y[1:50])

# ARMA(1,1) DGP
n = 200

y = zeros(n)
sig = 0.5
u = rand(Normal(0,sig),n)
plot(u)

# generate ARMA(1,1) psuedo-data:
function arma1_dgp(n,sig, a, phi, theta)
    e = rand(Normal(0,sig),n)
    y = zeros(n)
    y[1] = a + e[1]
    for t in 2:n
        y[t] = a + phi*y[t-1] + theta*e[t-1] + e[t]     # + 0.4*y[t-1] + 0.0*y[t-2] +
            # 0.4*e[t-1] + 0.0*e[t-2] + e[t]
    end
    return y
end

n = 500
a = 0.1
theta = 0.8
phi = 0.7
sig = 0.5  # adjust signal to noise ratio here
y = arma1_dgp(n,sig, a, phi, theta)

plot(y)
pz95 = quantile(y,[0.025,0.975])
hline!(pz95, linecolor = "green", label=false)
hline!([mean(y)], label=false, linecolor="black")
mean(y)
std(y)

# MA(1) model
@model ma1(x, ::Type{TV} = Vector{Float64}) where {TV} =begin
    n = length(x)
    s ~ Uniform(0.001, 5.0)   #InverseGamma(2,3)
    α ~ Normal(0, 3.0)
 #   ϕ ~ Normal(0, sqrt(s))
    θ ~ Normal(0, 1.0)
    u = TV(undef,n)
    for i in eachindex(x)  # NEED TO DEFINE u!
        if i == 1
            x[i] ~ Normal(α, s)
            u[i] = x[i] - α
        else #  i == 2
            x[i] ~ Normal(α + θ*u[i-1], s)
            u[i] = x[i] - α - θ*u[i-1]
#        else
#            x[i] ~ Normal(a + th1*u[i-1] + th2*u[i-2], sqrt(s))
       end
    end
end

@model arma11(x, ::Type{TV} = Vector{Float64}) where {TV} =begin
    n = length(x)
    s ~ Uniform(0.001, 5.0)   #InverseGamma(2,3)
    α ~ Normal(0, 3.0)
    ϕ ~ Normal(0, s)
    θ ~ Normal(0, 1.0)
    u = TV(undef,n)
    for i in eachindex(x)  # NEED TO DEFINE u!
        if i == 1
            x[i] ~ Normal(α, s)
            u[i] = x[i] - α
        else #  i == 2
            x[i] ~ Normal(α + ϕ*x[i-1] + θ*u[i-1], s)
            u[i] = x[i] - α - ϕ*x[i-1]- θ*u[i-1]
#        else
#            x[i] ~ Normal(a + th1*u[i-1] + th2*u[i-2], sqrt(s))
       end
    end
end


@model arma21(x, ::Type{TV} = Vector{Float64}) where {TV} =begin
    n = length(x)
    s ~ Uniform(0.001, 5.0)   #InverseGamma(2,3)
    α ~ Normal(0, 3.0)
    phi1 ~ Normal(0, s)
    phi2 ~ Normal(0, s)
    θ ~ Normal(0, 1.0)
    u = TV(undef,n)
    for i in eachindex(x)  # NEED TO DEFINE u!
        if i == 1
            x[i] ~ Normal(α, s)
            u[i] = x[i] - α
        elseif  i == 2
            x[i] ~ Normal(α + phi1*x[i-1] + θ*u[i-1], s)
            u[i] = x[i] - α - phi1*x[i-1]- θ*u[i-1]
        else
            x[i] ~ Normal(α + phi1*x[i-1] + phi2*x[i-2] + θ*u[i-1], s)
            u[i] = x[i] - α - phi1*x[i-1]- phi2*x[i-2]- θ*u[i-1]
       end
    end
end

@model arma12(x, ::Type{TV} = Vector{Float64}) where {TV} =begin
    n = length(x)
    s ~ Uniform(0.001, 5.0)   #InverseGamma(2,3)
    α ~ Normal(0, 3.0)
    phi1 ~ Normal(0, 5.0)
    theta2 ~ Normal(0, 5.0)
    θ ~ Normal(0, 1.0)
    u = TV(undef,n)
    for i in eachindex(x)  # NEED TO DEFINE u!
        if i == 1
            x[i] ~ Normal(α, s)
            u[i] = x[i] - α
        elseif  i == 2
            x[i] ~ Normal(α + phi1*x[i-1] + θ*u[i-1], s)
            u[i] = x[i] - α - phi1*x[i-1]- θ*u[i-1]
        else
            x[i] ~ Normal(α + phi1*x[i-1] + theta2*u[i-2] + θ*u[i-1], s)
            u[i] = x[i] - α - phi1*x[i-1]- theta2*u[i-2]- θ*u[i-1]
       end
    end
end

include("useful_functions_mcmc.jl")


# Estimate the model
model = ma1(y)
model = arma21(y)
model = arma12(y)

model = arma21(dif)
Turing.setprogress!(true)
iter = 10000
@time cc = sample(model, NUTS(0.65),iter)

cc

plot(cc)

model = arma12(dif_CPI)
Turing.setprogress!(true)
iter = 10000
@time cc = sample(model, NUTS(0.65),iter)

cc

plot(cc)

# one parameter (on MCMC chain) 
phi1 = get(cc,:phi1)
phi1_est = Array(phi1.phi1)
mean(phi1_est)
std(phi1_est)
plot(phi1_est, st=:density)

phi1_arma21 = phi1_est
phi1_arma12 = phi1_est

dif_phi1 = phi1_arma21 - phi1_arma12

plot(phi1_arma21, st=:density, fill=true, alpha=0.5, label="ARMA(2,1) model posterior") #, xlims=(-0,1.2), grid=false)
plot!(phi1_arma12, st=:density, fill=true, alpha=0.5, label="ARMA(1,2)model posterior")
dif_phi1 = phi1_arma21 - phi1_arma12

plot!(dif_phi1, st=:density, fill=true, label = "Difference in AR(1) coeffs posterior")
vline!([0.0], linecolor=:black, linewidth=2, label=false)
prob_le_zero = length(dif_phi1[dif_phi1 .<= 0.0])/length(dif_phi1)
p90 = quantile(dif_phi1, [0.05, 0.95])
vline!(p90, linecolor=:blue, linewidth=2, label=false)

# one parameter (on MCMC chain) 
θ = get(cc,:θ)
θ_est = Array(θ.θ)
mean(θ_est)
std(θ_est)
plot(θ_est, st=:density)

theta1_arma11 = θ_est
theta1_ma1 = θ_est

plot(theta1_arma11, st=:density, fill=true, alpha=0.5, label="ARMA(1,1) modle posterior") #, xlims=(-0,1.2), grid=false)
plot!(theta1_ma1, st=:density, fill=true, alpha=0.5, label="MA(1)model posterior")
dif_ar1 = theta1_arma11 - theta1_ma1

plot!(dif_ar1, st=:density, fill=true, label = "Difference in AR(1) coeffs posterior")
vline!([0.0], linecolor=:black, linewidth=2, label=false)
prob_le_zero = length(dif_ar1[dif_ar1 .<= 0.0])/length(dif_ar1)
p90 = quantile(dif_ar1, [0.05, 0.95])
vline!(p90, linecolor=:blue, linewidth=2, label=false)


std(theta1_arma11)
std(theta1_ma1)

ϕ = get(cc,:ϕ)
ϕ_est = Array(ϕ.ϕ)
mean(ϕ_est)
std(ϕ_est)
plot(ϕ_est, st=:density)


sigma = get(cc,:sigma)
sigma_est = Array(sigma.sigma)
mean(sigma_est)
std(sigma_est)
plot(sigma_est, st=:density)


b = get(cc,:beta1)
bs = Array(b.beta1)




# save value from AR(1) posterior
ar1_phi1 = bs
ar2_phi1 = bs

plot(ar1_phi1, st=:density, fill=true, alpha=0.5, label="AR(1) posterior") #, xlims=(-0,1.2), grid=false)
plot!(ar2_phi1, st=:density, fill=true, alpha=0.5, label="AR(2) posterior")
dif_ar1 = ar2_phi1 - ar1_phi1

plot!(dif_ar1, st=:density, fill=true, label = "Difference in AR(1) coeffs posterior")
vline!([0.0], linecolor=:black, linewidth=2)
prob_le_zero = length(dif_ar1[dif_ar1 .<= 0.0])/length(dif_ar1)
p90 = quantile(dif_ar1, [0.05, 0.95])
vline!(p90, linecolor=:blue, linewidth=2)



# several (k) parameters (MCMC chains) in a vector
k = 1
b = get(cc,:beta)
bs = zeros(iter,k)
for i in 1:k
    bs[:,i] = Array(b.beta[i])
end

summary_mcmc_array(bs,coefnames)
# bs2 = bs[:,j]
j = 1
plot(bs[:,j], st=:density, fill=true, alpha=0.5, label="HMC posterior") #, xlims=(-0,1.2), grid=false)
plot!(bdraws[:,j], st=:density, fill=true, alpha=0.5, label="Gibbs posterior")






################################################################

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


# Stan AR(1) model
@model stan_ar1(y, ::Type{TV}=Vector{Float64}) where {TV} =begin
  N = length(y)
  alpha ~ Normal(0,10)
  beta ~ Normal(0, 20.0)
  #beta ~ Truncated(Normal(0.0,20),-1.1,1.1)
  sigma ~ Uniform(0.01,10)
  for n in 2:N
    y[n] ~ Normal(alpha + beta * y[n-1], sigma);
  end
end


@model stan_ar2(y, ::Type{TV}=Vector{Float64}) where {TV} =begin
    N = length(y)
    alpha ~ Normal(0,10)
    beta1 ~ Normal(0, 20.0)
    beta2 ~ Normal(0, 20.0)
    #beta ~ Truncated(Normal(0.0,20),-1.1,1.1)
    sigma ~ Uniform(0.01,10)
    for t in 3:N
      y[t] ~ Normal(alpha + beta1 * y[t-1] + beta2 * y[t-2], sigma);
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


#model = simple_regression(yt, X)
#model = informative_prior_model(yt, X)
model = stan_ar2(y)
Turing.setprogress!(true)
iter = 6000
@time cc = sample(model, NUTS(0.65),iter)

cc

plot(cc)


# one parameter (on MCMC chain) 
sigma = get(cc,:sigma)
sigma_est = Array(sigma.sigma)
mean(sigma_est)
std(sigma_est)
plot(sigma_est, st=:density)


b = get(cc,:beta1)
bs = Array(b.beta1)




# save value from AR(1) posterior
ar1_phi1 = bs
ar2_phi1 = bs

plot(ar1_phi1, st=:density, fill=true, alpha=0.5, label="AR(1) posterior") #, xlims=(-0,1.2), grid=false)
plot!(ar2_phi1, st=:density, fill=true, alpha=0.5, label="AR(2) posterior")
dif_ar1 = ar2_phi1 - ar1_phi1

plot!(dif_ar1, st=:density, fill=true, label = "Difference in AR(1) coeffs posterior")
vline!([0.0], linecolor=:black, linewidth=2)
prob_le_zero = length(dif_ar1[dif_ar1 .<= 0.0])/length(dif_ar1)
p90 = quantile(dif_ar1, [0.05, 0.95])
vline!(p90, linecolor=:blue, linewidth=2)



# several (k) parameters (MCMC chains) in a vector
k = 1
b = get(cc,:beta)
bs = zeros(iter,k)
for i in 1:k
    bs[:,i] = Array(b.beta[i])
end

summary_mcmc_array(bs,coefnames)
# bs2 = bs[:,j]
j = 1
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
    y0 ~ Normal(0,10)    # Normal(y[1],std(y[1:5])*1.5)  # reasonable prior for y0?
    sig ~ Uniform(0.01,10)
    beta = TV(undef,(D))
    # for k in 1:(D)
    #     beta[k] ~ Normal(0, 20.0)
    # end
    beta[1] ~ Normal(0, 20.0)
    beta[2] ~ Normal(0, 20.0)
  #  beta[2] ~ Truncated(Normal(0, 10), -0.9999, 0.9999)
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

