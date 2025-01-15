# gibbs sampling for SUR using Turing.jl

# lags function: embed function in julia
# Creates matrix with contemporaneous + p lags of x
function embed(x,p)
    n = length(x)
    pp = p+1
    m = zeros(n-pp+1,pp)
    for i in 1:pp
        m[:,i] = x[(pp-i+1):(n-i+1)]
    end
    return m
end


"""
#### SURW appears to work.  Need sensible priors for covariance terms  #####
## Cannot get Wishart to work with MvNormal ##
"""
# y = stacked y variables, X = stacked X matrix including 1s for intercept
# Gaussian prior for coefficients b ~ MVN(b0,B0)
# Wishart prior for precision Sigma^{-1} ~ W(v0,R0)
# See Greenberg (2014), p.171
# m = number of equations
# yy = (y1, y2, ..., ym) tuple of ys
# XX = (X1, X2, ..., Xm) tuple of Xs
# start with 2 equation model:
    # values outside iterations
# Generalize to m equations
#    sse = 0.0
#    y = []
#    X = zeros()
#    for i in 1:m
#        yi = yy[i]
#        Xi = XX[i]
#        sse = sse + (y-X*b)'(y-X*b)
#    end
m = 2
tau = 1.0

using LinearAlgebra, Distributions, Random

iSigma = ones(m,m).*tau + I # no. of equations
function gssur2(yy,XX,m, iSigma; b0=0.0,iB0=0.0001) #; M=10000, burnin = Int(floor(M/10.0)),tau=1.0,iB0=0.0001,b0=0.0,d0=0.0001, a0=3.0)
    y1 = yy[1]
    y2 = yy[2]
    X1 = XX[1]
    X2 = XX[2]
    n, k1 = size(X1)
    n, k2 = size(X2)
    b0 = ones(k1+k2).*b0
    iB0 = (zeros((k1+k2),(k1+k2)) + I).*iB0
    XSX = zeros((k1+k2),(k1+k2))
    XSy = zeros(k1+k2)
#function we(XSX,XSy)
    for j in 1:n
        yj = [ y1[j]; y2[j] ]
        Xj = [ X1[j,:]' zeros(k2)'; zeros(k1)' X2[j,:]']
        XSX  = XSX + Xj'*iSigma*Xj
        XSy =  XSy + Xj'*iSigma*yj
    end
#    return XSX, XSy
#end

#XSX, XSy = we(XSX, XSy)

    B1 = inv(XSX + iB0)
    b1 = B1*(XSy + iB0*b0)
    bdraw = rand(MvNormal(b1,B1))
    return bdraw
end

yy = [y1 y2]
bdraw = gssur2(yy,XX,2, iSigma)


"""
    betas = vcat(beta1, beta2)
    ssej = 0.0
    for j in 1:n
        yj = [ y1[j]; y2[j] ]
        Xj = [ X1[j,:]' zeros(k2)'; zeros(k1)' X2[j,:]']
        ssej = ssej + (yj - Xj*betas)'*iSigma*(yj - Xj*betas)
    end

    n, k = size(X)
    a1 = a0 + n
    # default uninformative priors
    if b0 == 0.0
        b0 = zeros(k)
    end
    if iB0 == 0.0001     # coefficient mean and var.
        B0 = ones(k).*10000.0
        mB0 = Diagonal(B0)
        iB0 = inv(mB0)
    end

    bdraws = zeros(M,k)
    s2draws = zeros(M,1)
# Gibbs algorithm
    for i = 1:(M + burnin)

    # draw betas
        Db = inv(X'*X.*tau[1] + iB0)
        db = X'y.*tau[1] + iB0*b0
        H = cholesky(Hermitian(Db))
        betas = Db*db + H.U*randn(k,1)

    # draw sigma sq.
    #### N.B. second parameter is inverse of Greenberg/Koop defns.!
        d1 = d0 .+ (y-X*betas)'*(y-X*betas)
        tau = rand(Gamma(a1[1]/2,2/d1[1]),1)
        sig2 = 1/tau[1]

    # store draws
        if i > burnin
            j = i - burnin
            bdraws[j,:] = betas'
            s2draws[j] = sig2
        end

    end
    return bdraws,s2draws
end
"""

## simulate data for SUR model
Random.seed!(656)

## simulate data from SUR

# two equation SUR model
function sur2_dgp(beta1,beta2,Sigma,nobs)
    iota = ones(nobs)
    X1 = [iota rand(nobs)]
#    X2 = [iota rand(nobs) rand(nobs)]
    H = cholesky(Hermitian(Sigma))
    E = randn(4,nobs)'*H.U
    y1 = X1*beta1 + E[:,1]
    y2 = X1*beta2 + E[:,2]
    y3 = X1*beta1 + E[:,3]
    y4 = X1*beta2 + E[:,4]
    return y1, y2, y3, y4, X1
end

beta1 = [1, 2]
beta2 = [1, 2]
beta3 = [1, 2]
beta4 = [1, 2]
Sigma = [0.3 0.0 0.1 0.1; 0.0 0.3 0.1 0.1; 0.1 0.1 0.3 0.1; 0.1 0.1 0.1 0.3]
nobs = 200
y1, y2, y3, y4, XX = sur2_dgp(beta1,beta2,Sigma,nobs)

# put data in tuples (don't have to be same length) - why?
#yy = (y1, y2)
#XX = (XX, XX)
m = 2  # number of equations in SUR system

# identity matrix function
using LinearAlgebra
function eye(m)
    eye = zeros(m,m) + I
    return eye
end
x = eye(10)


using Turing, MCMCChains, Distributions, StatsPlots, BayesTesting

# Turing.jl SUR model
# X1 and X2 covariate matrices include 1s for intercept
@model SUR(y1, X1, y2, X2, n, k1, k2) = begin
    # Set variance prior.
    S = zeros(2,2) + I.*40.00
    Sigma ~ Wishart(60, S)
#    iSigma = Symmetric(Hermitian(Matrix{Float64}(inv(Sigma))))
    iSigma = Matrix(Hermitian(inv(Sigma)))
    # Set the priors on our coefficients.
#    b = Array{Real}(undef,k)
# intercepts
#    a1 ~ Normal(0, 10)
#    a2 ~ Normal(0, 10)
    # regression coeffs priors
    b1 = Array{Real}(undef, k1)
    b2 = Array{Real}(undef, k2)
     for i in 1:k1
         b1[i] ~ Normal(0, 10)
     end
     for i in 1:k2
         b2[i] ~ Normal(0, 10)
     end

    # Calculate all the mu terms.
    mu1 = X1 * b1
    mu2 = X2 * b2

 #  for i = 1:n
#       ss = cholesky(iSigma)
#       s1 = sqrt(ss.U[1,1]^2 + ss.U[1,2]^2)
#       y1[i] ~ Normal(mu1[i], s1)  # SD not variance in julia Normal
#  end

   for i = 1:n
       yy = [y1[i], y2[i]]
       mu = [mu1[i], mu2[i]]
       yy ~ MvNormal(mu, iSigma)
   end
end

## 4 variable VAR 10-2019
@model SUR4var(y1, y2, y3, y4, XX, n, k) = begin
    # Set variane prior.
    S = zeros(4,4) + I.*1.00
    Sigma ~ Wishart(6, S)
    iSigma = Matrix(Hermitian(inv(Sigma)))
    # Set the priors on our coefficients.
    b1 = Array{Real}(undef, k)
    b2 = Array{Real}(undef, k)
    b3 = Array{Real}(undef, k)
    b4 = Array{Real}(undef, k)
     for i in 1:k
         b1[i] ~ Normal(0, 100)
     end
     for i in 1:k
         b2[i] ~ Normal(0, 100)
     end
     for i in 1:k
         b3[i] ~ Normal(0, 100)
     end
     for i in 1:k
         b4[i] ~ Normal(0, 100)
     end
    # Calculate all the mu terms.
    mu1 = XX * b1
    mu2 = XX * b2
    mu3 = XX * b3
    mu4 = XX * b4
   for i = 1:n
       yy = [y1[i], y2[i], y3[i], y4[i]]
       mu = [mu1[i], mu2[i], mu3[i], mu4[i]]
       yy ~ MvNormal(mu, Symmetric(iSigma))
   end
end



# 4 VARIABLE VAR
@model SUR4(y1, y2, y3, y4, XX, n, k) = begin
#    n, k = size(XX)
#    k1 = k2 = k3 = k4 = k
    # Set variance prior.
#    S = Matrix(Hermitian(zeros(4,4) + I.*40.00))
#    S = zeros(4,4) + I.*40.00
    Sigma ~ InverseWishart(40, Matrix{Float64}(I.*10.0, 4, 4))
    #    Sigma ~ Wishart(200, S)
#    iSigma = Symmetric(Hermitian(Matrix{Float64}(inv(Sigma))))
##    iSigma = Matrix(Hermitian(inv(Sigma)))
    # Set the priors on our coefficients.
#    b = Array{Real}(undef,k)
    # regression coeffs priors
    b1 = Array{Real}(undef, k)
    b2 = Array{Real}(undef, k)
    b3 = Array{Real}(undef, k)
    b4 = Array{Real}(undef, k)
     for i in 1:k
         b1[i] ~ Normal(0, 10)
#     end
#     for i in 1:k2
         b2[i] ~ Normal(0, 10)
#     end
#     for i in 1:k3
         b3[i] ~ Normal(0, 10)
#     end
#     for i in 1:k4
         b4[i] ~ Normal(0, 10)
     end

    # Calculate all the mu terms. - same X for each equation
    mu1 = XX * b1
    mu2 = XX * b2
    mu3 = XX * b3
    mu4 = XX * b4

   for i = 1:n
       yy = [y1[i], y2[i], y3[i], y4[i]]
       mu = [mu1[i], mu2[i], mu3[i], mu4[i]]
#       yy ~ MvNormal(mu, iSigma)
        yy ~ MvNormal(mu, Symmetric(Sigma))
   end
end


@model SURW(y1, X1, y2, X2, n, k1, k2, ::Type{TV}=Vector{Float64}) where {TV} = begin
    # regression coeffs priors
    b1 = TV(undef, k1)
    b2 = TV(undef, k2)
     for i in 1:k1
         b1[i] ~ Normal(0, 100)
     end
     for i in 1:k2
         b2[i] ~ Normal(0, 100)
     end
#    b1 ~ [Normal(0, 100)]
#    b2 ~ [Normal(0, 100)]
    # Calculate all the mu terms.
    mu1 = TArray{Any}(2)
    mu2 = TArray{Any}(2)
#    mu1 = TV(2)
#    mu2 = TV(2)
    mu1 = X1 * b1
    mu2 = X2 * b2
    # Create sigma variables.
    sigma = TArray{Any}(2)
    sigma[1] ~ TruncatedNormal(0, 100, 0.001, 100)
    sigma[2] ~ TruncatedNormal(0, 100, 0.001, 100)

    # Create rho.
    rho ~ Truncated(Uniform(-1, 1), -0.999, 0.999)
#    s12 ~ Normal(0,100)

    # Generate covariance matrix.
    s12 = sigma[1]*sigma[2]*rho
    cv = [sigma[1]^2 s12;
            s12 sigma[2]^2]

    for i = 1:n
        v = [y1[i], y2[i]]
        mu = [mu1[i], mu2[i]]
        v ~ MvNormal(mu, cv)
    end
end


# MCMC for SUR model
n1, k1 = size(XX)
n2, k2 = size(XX)
n = n1
# model = SURW(y1, X1, y2, X2, n, k1, k2)
# 2 equation system:
model = SUR(y1, XX, y2, XX, n, k1, k2)
chain = sample(model, NUTS(2000, 0.65));
# chain = sample(model, HMC(3000, 0.01, 10));
@show(describe(chain))
plot(chain)


# 4 equation system:
Random.seed!(1358)
n, k = size(XX)
model = SUR4var(y1, y2, y3, y4, XX, n, k)
Turing.setadbackend(:reverse_diff)
Turing.setadbackend(:forward_diff)
chain = sample(model, NUTS(2000, 0.65));

@show(describe(chain))
plot(chain)

α = chain[:intercept][200:end]
betas  = chain[:coefficients][200:end]
beta1 = chain[200:end,["coefficients[1]"],:]
b1 = get_chain(beta1)
quantile(b1,[0.025,0.5,0.975])
beta2 = chain[200:end,["coefficients[1]"],:]
b2 = get_chain(beta2)
quantile(b1,[0.025,0.5,0.975])

b2 = get_chain(chain[200:end,["coefficients[1]"],:])
b3 = get_chain(chain[200:end,["coefficients[2]"],:])

plot(b3)
plot(b2,st=:histogram, bins=100,normalize=true)
plot!(b2,st=:density,linewidth=2)

mcodds(b2, h0=2.1)
bayespval(b2,h0=2.1)


## model for Wishart?
@model Wish(x) = begin
    # Set variance prior.
#    s2 ~ TruncatedNormal(0,100, 0, Inf)
     S = zeros(2,2) + I.*4.00
     Sigma ~ Wishart(60, S)
#    iSigma = Matrix{Float64}(inv(Sigma))
##    iS = zeros((k1+k2),(k1+k2)) + I./400.00
    m1 ~ Normal(0,100)
    m2 ~ Normal(0,100)
##   iSigma ~ InverseWishart(6, iS)
#     for i in 1:length(x[:,1])
#         x[i,1] ~ Normal(m1,Sigma[1,1])
#         x[i,2] ~ Normal(m2,(Sigma[2,2]*(1.0 - Sigma[2,1])^2))
#     end
    for i in 1:length(x[:,1])
        mu = [m1, m2]
        SS = [Sigma[1,1] Sigma[1,2];
              Sigma[2,1] Sigma[2,2]]
        x[i,:] ~ MvNormal(mu,SS)
    end
 end


n = 50
x = [randn(n) randn(n)]
model = Wish(x)
chain = sample(model, NUTS(1000, 0.65));
describe(chain[200:end])
plot(chain[200:end])

function SUR(y1, X1, y2, X2, n, k1, k2)
    # Set variance prior.
    S = zeros(2,2) + I.*400.00
    Sigma = rand(Wishart(6, S))
    iSigma = Symmetric(Hermitian(Matrix{Float64}(inv(Sigma))))
    # Set intercept prior.
#    intercept1 ~ Normal(0, 3)
#    intercept2 ~ Normal(0, 3)

    # Set the priors on our coefficients.
    coefficients1 = Array{Real}(undef, k1)
    coefficients2 = Array{Real}(undef, k2)
    coefficients1 ~ [Normal(0, 100)]
    coefficients2 ~ [Normal(0, 100)]

    # Calculate all the mu terms.
    mu1 = X1 * coefficients1
    mu2 = X2 * coefficients2

    for i = 1:n
        y = (y1[i], y2[i])
        mu = [mu1[i], mu2[i]]
        y ~ MvNormal(mu, iSigma)
#    y1[i] ~ Normal(mu1[i], iSigma[1,1])
    end
end;




beta1 = [1, 2]
beta2 = [1,-1,-2]
betas = vcat(beta1,beta2)


# function to draw betas|Sigma ~ MVN(b1,B1)
function b1B1(y1, y2, X1, X2,b0,iB0,iSigma)
    n, k1 = size(X1)
    n, k2 = size(X2)
    XSX = zeros((k1+k2),(k1+k2))
    XSy = zeros(k1+k2)
    for j in 1:n
        yj = [ y1[j]; y2[j] ]
        Xj = [ X1[j,:]' zeros(k2)'; zeros(k1)' X2[j,:]']
        XSX  = XSX + Xj'*iSigma*Xj
        XSy =  XSy + Xj'*iSigma*yj
    end
    B1 = Symmetric(Hermitian(Matrix{Float64}(inv(XSX + iB0))))
    b1 = B1*(XSy + iB0*b0)
    bdraw = rand(MvNormal(b1,B1))
    return bdraw
end

function sses(y1, y2, X1, X2, betas, iSigma)
    ssej = zeros(2,2)
    for j in 1:n
        yj = [ y1[j]; y2[j] ]
        Xj = [ X1[j,:]' zeros(k2)'; zeros(k1)' X2[j,:]']
        ssej = ssej + (yj - Xj*betas)*(yj - Xj*betas)'
    end
    return ssej
end

default priors and starting value for Sigma
function SUR_Gibbs(y1, y2, X1, X2, iters)
    # priors and starting value for Sigma
    S = zeros(2,2) + I.*400.00
    iSigma = Symmetric(Hermitian(Matrix{Float64}(rand(Wishart(6, S)))))
    Sigma = inv(iSigma)
    b0 = 0.0
    iB0 = 1000.00
    b0 = ones(k1+k2).*b0
    iB0 = (zeros((k1+k2),(k1+k2)) + I).*iB0
    v = n + 6
    iR0 = zeros(2,2) + I./400.00
    bdraws = zeros(iters,(k1+k2))
    s1draws = zeros(iters)
    s2draws = zeros(iters)
    s12draws = zeros(iters)
    for i in 1:iters
    # draw betas|Sigma
        bdraw = b1B1(y1, y2, X1, X2,b0,iB0,iSigma)
        bdraws[i,:] = bdraw'
        # draw Sigma|betas
        sse = sses(y1, y2, X1, X2, bdraw, iSigma)
        R1 = inv(iR0 + sse)
        Sigma = rand(Wishart(v, R1))
        s1draws[i] = Sigma[1,1]
        s2draws[i] = Sigma[2,2]
        s12draws[i] = Sigma[1,2]
        iSigma = Symmetric(Hermitian(Matrix{Float64}(inv(Sigma))))
    end
    return bdraws, s1draws, s2draws, s12draws
end

M = 10000
bdraws, s1draws, s2draws, s12draws = SUR_Gibbs(y1, y2, X1, X2,M)
plot(bdraws[:,1])
plot(density(bdraws[:,1]))
plot(bdraws[:,2])
plot(density(bdraws[:,2]))
plot(bdraws[:,3])
plot(density(bdraws[:,3]))
plot(bdraws[:,4])
plot(density(bdraws[:,4]))
plot(bdraws[:,5])
plot(density(bdraws[:,5]))
plot(s1draws)
plot(density(s1draws))
plot(s2draws)
plot(density(s2draws))
plot(s12draws)
plot(density(s12draws))



mu1 = mean(y1).*ones(length(y1))
mu2 = mean(y2).*ones(length(y2))
cv = [0.3 0.1; 0.1 0.2]
v = zeros(length(y1),2)
#for i = 1:n
#    v = [y1[i], y2[i]]
#    mu = [mu1[i], mu2[i]]
#    v[i,:] = rand(MvNormal(mu, cv))
#end
@show(v)


rand(MvNormal(mu1,cv))
cv = zeros(n,n) + I



using GLM
data = DataFrame(Y = y1, X = XX)
ols = lm(@formula(Y ~ X), data)

xx = XX[:,2]
blinreg(y1,xx)

bs = inv(XX'XX)*XX'y1
sig2hat = ((y1 - XX*bs)'*(y1 - XX*bs)) /(length(y1) - 2)
