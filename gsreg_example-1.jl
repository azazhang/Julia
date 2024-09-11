# gsres function example
# cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2022")

## data here:
using Distributions, Plots, StatsPlots
n = 100
x = randn(n)
y = 1.0 .+ 0.5.*x .+ 0.4.*randn(n)
plot(x,y,st=:scatter)

## AR(1) DGP
phi = 1.0
z = zeros(n+20)
for t = 2:(n+20)
    z[t] = 0.0 + phi*z[t-1] + randn(1)[1]
end
y = z[21:(n+20)]
plot(y)


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


plot(s2draws, st=:density, fill=true, alpha=0.5, title =  "σ squared posterior", label= false )



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
