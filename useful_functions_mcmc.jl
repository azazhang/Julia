# Packages used
using CSV, DataFrames, Plots, StatsPlots, Distributions, LinearAlgebra
using Turing, StatsFuns, Statistics, PrettyTables

# set path to working directory
# Mac
#cd("/Users/millsja/OneDrive - University of Cincinnati/Class/9011 2022")

# Windows
#cd("I:\\Econ 9011 2022")
#cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2022")

# Surface - local copy
#cd("C:\\Users\\millsjf\\Desktop\\Current Research\\FiESTAA_GI_AEs")
#cd("C:\\Users\\millsjf\\OneDrive for Business\\FIESTAA\\FiESTAA_GI_AEs")


function pval(vector)
    pval = 2*(1 - sum(vector .< 0.0)/(size(vector,1)))
    if pval > 1.0
        return pval = round(2-pval,digits=4)
    else
        return round(pval,digits=4)
    end
end

""" old version of pval function:
function pval(vector)
    if typeof(vector) == Matrix{Float64}
        pval = zeros(size(vector, 2))
        for i = 1:size(vector, 2)
            pval[i] = 2 * (1 - sum(vector[:, i] .< 0.0) / (size(vector[:, i], 1)))
            if pval[i] > 1.0
                pval[i] = round(2 - pval[i], digits = 4)
            else
                pval[i] = round(pval[i], digits = 4)
            end
        end
    else
        pval = 2 * (1 - sum(vector .< 0.0) / (size(vector, 1)))
        if pval > 1.0
            pval = round(2 - pval, digits = 4)
        else
            pval = round(pval, digits = 4)
        end
    end
    return pval
end
"""

function linreg(x, y)
    blinreg = x \ y
    n = length(y)
    k = length(blinreg)
    res = y - x * blinreg
    sse = sum(res .^ 2)
    sigma2_hat = sse / (n - k)
    covb = inv(x'x) * sigma2_hat
    b2se = diag(sqrt.(abs.(Diagonal(covb))))
    criu = blinreg .+ 1.96 .* b2se
    cril = blinreg .- 1.96 .* b2se
    tstats = blinreg ./ b2se
    pval1 = 2.0 .* cdf.(TDist(n - k), tstats)
    pval2 = 2.0 .* (1.0 .- cdf.(TDist(n - k), tstats))
    pvals = minimum([pval1 pval2], dims = 2)
    sst = sum((y .- mean(y)) .^ 2)
    Rsq = 1.0 - sse / sst
    return blinreg, b2se, tstats, pvals, Rsq, sigma2_hat, cril, criu
end

function rsquare(y, x, bhat)
    res = y  - x*bhat
    sse = sum(res.^2)
    sst = sum((y .- mean(y)).^2)
    Rsq = 1.0 - sse/sst
    n, k = size(x)
    bic = n*log(sse/n) + k*log(n)
    aic = n*log(sse/n) + 2*k
    return Rsq, aic, bic
end

function print_regression(blinreg, b2se, pvals, Rsq, cril, criu, coefnames)
    println(" Variable   coeff    s.e.    pval       CrI")
    for i = 1:length(blinreg)
        println(coefnames[i], "         ", round(blinreg[i], digits = 3), "     ", round(b2se[i], digits = 3), "     ", round(pvals[i], digits = 4), "     ", round(criu[i], digits = 4), "    ", round(cril[i], digits = 4))
    end
    println("Rsquared = ", round(Rsq, digits = 3))
end

"""
# hpdi - Compute high density region.
Derived from `hpd` in MCMCChains.jl.
By default alpha=0.05 for a 2-sided tail area of p < 0.025% and p > 0.975%.
"""
function hpdi(x::Vector{T}; alpha=0.05) where {T<:Real}
    n = length(x)
    m = max(1, ceil(Int, alpha * n))
    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)

    return [a[i], b[i]]
end

# requires hpdi function:
function table(cc)
    params = cc.name_map.parameters
    s = size(cc.value.data)[1]
    x = zeros(s, length(params))
    for i in 1:length(params)
        x[:, i] = parent(cc[params[i]])[:]
    end

    mx = round.(mean(x, dims=1)'[:], digits=3)
    stx = round.(std(x, dims=1)'[:], digits=3)
    px = pval(x)
    qx = [round.(quantile(x[:, i], [0.025, 0.975]), digits=3) for i in 1:size(x, 2)]
    hx = [round.(hpdi(x[:, i]), digits=3) for i in 1:size(x, 2)]

    df = [params mx stx px qx hx]
    header = ["Parameter" "Mean" "Std" "P-Val" "CI-95%" "HPD-95%"]
    tab = [header; df]
    (df=DataFrame(x, params), tab=pretty_table(tab[2:end, :], header=tab[1, :], alignment=:C))
    # return out,tab
end

function summary_mcmc_array(coeffs, coeffnames)
    if typeof(coeffs) == Vector{Float64}
        m = length(coeffs)
        k = 1
    else
        m, k = size(coeffs)
    end
    println(" coeff    ", "mean    ", "std    ", "p-value  ", " 0.95 interval")
    for i in 1:k
        bi = coeffs[:,i]
        pval1 = length(bi[bi .<= 0.0])/m
        pval2 = 1.0 - length(bi[bi .<= 0.0])/m
        pval = round(2*minimum([pval1 pval2]), digits = 4)
        println(coeffnames[i],"   ",  round(mean(bi), digits=3), "   ", round(std(bi), digits=3), "   ", pval, "   ", round.(quantile(bi, [0.025,0.975]), digits = 3))
    end
end


@model simple_logistic(y,X, ::Type{TV}=Vector{Float64}) where {TV} =begin
    n, D = size(X)
    #alpha ~ Normal(0,1)
    #sig ~ Uniform(0.01,3)
    #m ~ Truncated(Normal(-2,3),-999.9,0.999)
    beta = TV(undef,(D))
    # sd<10 too restrictive for beta coeffs 
    for k in 1:(D)
        beta[k] ~ Normal(0, 10.0)
    end
    #delta ~ Normal(0,3.0)
    mu = logistic.(Matrix(X) * beta)
    for i in 1:n
        v = Bernoulli(mu[i])
        y[i] ~ v
    end
end

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


### This xtlag function includes contemporaneous xt, the original Serial Correlation code function does not
function xtlag(X, laglength)
    n, k = size(X)
    w = zeros(n - laglength, (k) * (laglength + 1))
    p = collect(1:k:(k)*(1+laglength))
    for i in 1:(1+laglength)
        w[:, p[i]:(p[i]+k-1)] = X[(laglength+1-i+1):(n-i+1), :]
    end
    return w
end

# creates lags - same as embed function in R
function embed(x, p)
    n = length(x)
    pp = p + 1
    m = zeros(n - pp + 1, pp)
    for i in 1:pp
        m[:, i] = x[(pp-i+1):(n-i+1)]
    end
    return m
end

println("Functions loaded")

#dfmo = DataFrame(CSV.File("FiESTAA_07-20.csv")) # original data
#dfm = DataFrame(CSV.File("FiESTAA_07_20_cleaned.csv")) # cleaned data
#show(names(dfm))

#=

# example use of Turing for regression

@model lm(y, x, ::Type{TV}=Float64) where {TV} = begin
    n = length(y)

    sig ~ Uniform(0.001, 10.0)

    b = Vector{TV}(undef, k)

    [b[i] ~ Normal(0, 5) for i in 1:k]

    mu = x * b

    for i in 1:n
        y[i] ~ Normal(mu[i], sig)
    end

end

n = 50
b = [1; 1]
s = 1.0
x = [ones(n) randn(n)]
u = randn(n)
y2 = x * b .+ u
k = length(b)

# creating lags
p = 2
yp = embed(y2, p)
xt = xtlag(x, p)
X = [yp[:, 2:end] xt]
Y = yp[:, 1]

model_lm = lm(y2, x)

s = 2000
Turing.setprogress!(true)
# @time cc = sample(model, NUTS(0.65), s)
# @time cc_sc = sample(model_sc, NUTS(0.65), s)
@time cc_lm = sample(model_lm, NUTS(0.65), s)

table(cc_lm)[2]

# Is there a difference in these groups?
# What if we double the same size, but keep proportions the same?
## Drawing from beta
M = 10^6
sm = 25 *2
nm = 73 *2
sf = 13 *2
nf = 60 *2
sm/nm
sf/nf
mtrt = rand(Beta(sm+1, nm-sm+1), M)
mpbo = rand(Beta(sf+1, nf-sf+1), M)
plot(mtrt, st = :density, fill=true, alpha = 0.4, label = "treated")
plot!(mpbo, st = :density, fill=true, alpha = 0.4, label = "controls")
dif = mtrt .- mpbo
plot!(dif, st = :density, fill=true, alpha = 0.4, label = "difference")
vline!([0.0], linecolor = "black", label = false)
pval(dif)
mean(dif)
mean(mtrt)
mean(mpbo)


## SADrawing from beta
M = 10^6
sm = 2
nm = 136
sf = 13
nf = 137
sm/nm
sf/nf
mtrt = rand(Beta(sm+1, nm-sm+1), M)
mpbo = rand(Beta(sf+1, nf-sf+1), M)
plot(mtrt, st = :density, fill=true, alpha = 0.4, label = "treated")
plot!(mpbo, st = :density, fill=true, alpha = 0.4, label = "controls")
dif = mtrt .- mpbo
plot!(dif, st = :density, fill=true, alpha = 0.4, label = "difference")
vline!([0.0], linecolor = "black", label = false)
pval(dif)
mean(dif)
mean(mtrt)
mean(mpbo)


# Normal mean difference
sm = -1.34
nm = 136
msig = -(1.24 - 1.44)/2
sf = -1.21
nf = 137
fsig = -(1.11 - 1.31)/2
sm/nm
sf/nf
mtrt = rand(Normal(sm, msig), M)
mpbo = rand(Normal(sf, fsig), M)
plot(mtrt, st = :density, fill=true, alpha = 0.4, label = "treated")
plot!(mpbo, st = :density, fill=true, alpha = 0.4, label = "controls")
dif = mtrt .- mpbo
plot!(dif, st = :density, fill=true, alpha = 0.4, label = "difference")
vline!([0.0], linecolor = "black", label = false)
pval(dif)
mean(dif)
mean(mtrt)
mean(mpbo)

# three-ways to order the exact same chili:
vector = dif
sum(vector[:, 1] .< 0.0)
sum(vector .< 0.0)
length(vector[vector .< 0])

=#