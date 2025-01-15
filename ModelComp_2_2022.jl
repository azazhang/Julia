### Model Comparison - MMRM v BHM
# October 7th
using DataFrames, Turing, StatsBase, StatsPlots, Plots, LinearAlgebra, Random, CSV, PrettyTables
using BSON: @save,@load
# Run Functions in ModelComp.jl
### Comparing BHM and MMRM data generating processes under different model specifications
#= BHM - AR, Log Trend, Spectral
MMRM - Time Effects, Log Trend, Log Trend + Quadratic Trend with Error Structures - Compound Symmetry and Homogeneous AR1 Errors =#

### DGPs 
#= AR, Log Trend + IID Normal Error, Time Effects + Error Structures, TADS =#

# Time Effect + Compound Symmetry Error Structure - si, st are variances not SD.
function mix_cs((n, t, fes, si, st, rho),)

    om = zeros(t, t)
    for i in 1:t, j in 1:t
        if i == j
            om[i, j] = 1.0 * st
        else
            om[i, j] = st * rho
        end
    end

    x = rand(Bernoulli(0.5), n)

    xt = repeat(x, inner=t)

    ts = [repeat(Matrix(I, t, t)[:, i], outer=n) for i in 1:t]
    t_fes = reduce(hcat, ts)

    X = [ones(n * t) xt t_fes[:, 2:end] xt .* t_fes[:, 2:end]]
    xb = X * fes
    xb_s = xb .+ reduce(vcat, rand(MvNormal(fill(0.0, t), om), n)) .+ rand(Normal(0, sqrt(si)), (n * t))

    y0 = repeat(rand(5:7, n), inner=t)
    yt = zeros(n * t)
    yt[1:t:end] = y0[1:t:end]

    yt[Not(1:t:end)] = y0[Not(1:t:end)] .+ xb_s[Not(1:t:end)]

    return yt, DataFrame(X, :auto)
end

# Time Effect + IID Error Structure 
function mix((n, t, fes, sig),)

    x = rand(Bernoulli(0.5), n)

    xt = repeat(x, inner=t)

    e = sqrt(sig) .* randn(n * t)

    ts = [repeat(Matrix(I, t, t)[:, i], outer=n) for i in 1:t]
    t_fes = reduce(hcat, ts)

    X = [ones(n * t) xt t_fes[:, 2:end] xt .* t_fes[:, 2:end]]
    xb = X * fes
    xb_s = xb .+ e

    y0 = repeat(rand(5:7, n), inner=t)

    yt = zeros(n * t)
    yt[1:t:end] = y0[1:t:end]

    yt[Not(1:t:end)] = y0[Not(1:t:end)] .+ xb_s[Not(1:t:end)]

    return yt, DataFrame(X, :auto)
end

#= # Time Effect + Toeplitz Error Structure - si, st are variances not SD. rho is Vector with rho1 rho2 rho3 i.e., (t-1) Toeplitz has issues
function mix_t(n, t, b, si, st, rho)

    om = zeros(t, t)
    for i in 1:t, j in 1:t
        if i == j
            om[i, j] = 1.0 * st
        else
            om[i, j] = st * rho[abs(i - j)]
        end
    end

    x = randn(n)
    xt = repeat(x, inner=t)

    ts = [repeat(Matrix(I, t, t)[:, i], outer=n) for i in 1:t]
    t_fes = reduce(hcat, ts)

    X = [ones(n * t) xt t_fes[:, 2:end]]
    xb = X * b
    xb_s = xb .+ reduce(vcat, rand(MvNormal(fill(0.0, 4), om), n)) .+ rand(Normal(0, sqrt(si)), (n * t))

    y0 = repeat(rand(5:7, n), inner=4)

    y = y0 .+ xb_s
    y2 = copy(y)

    yt = y2

    return yt, DataFrame(X, ["intercept"; "x"; "t1"; "t2"; "t3"])
end =#

# Time Effect + AR1 Error Structure - si, st are variances not SD.
function mix_ar1((n, t, b, si, st, rho),)

    om = [st * rho^abs(i - j) for i in 1:t, j in 1:t]

    x = rand(Bernoulli(0.5), n)

    xt = repeat(x, inner=t)

    ts = [repeat(Matrix(I, t, t)[:, i], outer=n) for i in 1:t]
    t_fes = reduce(hcat, ts)

    X = [ones(n * t) xt t_fes[:, 2:end] xt .* t_fes[:, 2:end]]
    xb = X * b
    xb_s = xb .+ reduce(vcat, rand(MvNormal(fill(0.0, t), om), n)) .+ rand(Normal(0, sqrt(si)), (n * t))

    y0 = repeat(rand(5:7, n), inner=t)
    yt = zeros(n*t)
    yt[1:t:end] = y0[1:t:end]
    
    yt[Not(1:t:end)] = y0[Not(1:t:end)] .+ xb_s[Not(1:t:end)]

    return yt, DataFrame(X, :auto)
end

# Log Trend + IID Normal Errors
function lt_dgp((n, t, b, d, sig),)
    x = rand(Bernoulli(0.5), n)
    xt = repeat(x, inner=t)
    lt = repeat(log.(1:t), outer=n)

    xb = (xt .* lt) * b

    y0 = rand(5:7, n)

    yt = zeros(n * t)
    e = sqrt(sig) .* randn(n * t)

    for i in 2:n*t
        if rem(i, t) == 1
            yt[i] = 0.0
        else
            yt[i] = d * lt[i] + xb[i]
        end
    end

    return repeat(y0, inner=t) .+ yt .+ e, [lt xt]
end

# Log Trend + AR1 Errors 
function lt_ar1((n, t, b, d, si, st, rho),)
    x = rand(Bernoulli(0.5), n)
    xt = repeat(x, inner=t)
    lt = repeat(log.(1:t), outer=n)

    xb = (xt .* lt) * b

    y0 = rand(5:7, n)

    yt = zeros(n * t)

    om = [st * rho^abs(i - j) for i in 1:t, j in 1:t]
    e = reduce(vcat, rand(MvNormal(fill(0.0, t), om), n)) .+ rand(Normal(0, sqrt(si)), (n * t))

    for i in 2:n*t
        if rem(i, t) == 1
            yt[i] = 0.0
        else
            yt[i] = d * lt[i] + xb[i]
        end
    end

    return repeat(y0, inner=t) .+ yt .+ e, [lt xt]
end

# Log Trend + CS Errors
function lt_cs((n, t, b, d, si, st, rho),)
    x = rand(Bernoulli(0.5), n)
    xt = repeat(x, inner=t)
    lt = repeat(log.(1:t), outer=n)

    xb = (xt .* lt) * b

    y0 = rand(5:7, n)

    yt = zeros(n * t)

    om = zeros(t, t)
    for i in 1:t, j in 1:t
        if i == j
            om[i, j] = 1.0 * st
        else
            om[i, j] = st * rho
        end
    end
    e = reduce(vcat, rand(MvNormal(fill(0.0, t), om), n)) .+ rand(Normal(0, sqrt(si)), (n * t))


    for i in 2:n*t
        if rem(i, t) == 1
            yt[i] = 0.0
        else
            yt[i] = d * lt[i] + xb[i]
        end
    end

    return repeat(y0, inner=t) .+ yt .+ e, [lt xt]
end

#= # Log Trend + Toeplitz Errors
function lt_t(n, t, b, d, si, st, rho)
    x = randn(n)
    xt = repeat(x, inner=t)
    lt = repeat(log.(1:t), outer=n)

    xb = (xt .* lt) * b

    y0 = rand(5:7, n)

    yt = zeros(n * t)

    om = zeros(t, t)
    for i in 1:t, j in 1:t
        if i == j
            om[i, j] = 1.0 * st
        else
            om[i, j] = st * rho[abs(i - j)]
        end
    end
    e = reduce(vcat, rand(MvNormal(fill(0.0, 4), om), n)) .+ rand(Normal(0, sqrt(si)), (n * t))

    for i in 2:n*t
        if rem(i, t) == 1
            yt[i] = 0.0
        else
            yt[i] = d * lt[i] + xb[i]
        end
    end

    return repeat(y0, inner=t) .+ yt .+ e, [lt xt]
end
 =#
# AR1 
function ar_dgp((n, t, a, b, r, sig),)
    x = rand(Bernoulli(0.5), n)
    xt = repeat(x, inner=t)
    xb1 = xt * b[1]
    xb2 = xt * b[2]

    y0 = rand(5:7, n)

    yt = zeros(n * t)
    e = sqrt(sig) .* randn(n * t)

    yt[1:t:end] = y0

    for i in deleteat!(collect(1:n*t), collect(1:t:n*t))
        yt[i] = r * yt[i-1] + xb1[i] * yt[i-1] + xb2[i] + a
    end

    return yt .+ e, xt
end

#= Model Statements 
BHM - Spectral, Log Trend IID Errors, AR1 
=#

# Spectral Trend Model 
@model bhm_onefreq(y, X1, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1) # TORDIA 

    sig ~ Uniform(0.01, 5.0)

    w ~ Uniform(0, 1)
    a_s ~ Normal(0, 5)
    b_s ~ Normal(0, 5)
    # alp ~ Normal(0, 5)

    b = Vector{TV}(undef, (k1 - 1))

    for i = 1:(k1-1)
        b[i] ~ Normal(0, 1)
    end

    xb = Matrix(X1[:, 2:end]) * b
    mu1 = a_s .* sin.(2π .* w .* X1[:, 1]) .+ b_s .* cos.(2π .* w .* X1[:, 1]) .+ xb

    for i = 1:n1
        y[i] ~ Normal(mu1[i], sqrt(sig))
    end

end

# Log Trend Model
@model bhm_logt(y, X1, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1)

    sig ~ Uniform(0.01, 5.0)

    d1 ~ TruncatedNormal(-1, 1.0, -5, 0)

    b = Vector{TV}(undef, (k1 - 1))

    for i = 1:(k1-1)
        b[i] ~ Normal(0, 5)
    end

    mu1 = d1 .* X1[:, 1] .+ (Matrix(X1[:, 2:end])) * b

    for i = 1:n1
        y[i] ~ Normal(mu1[i], sqrt(sig))
    end

end

# AR1 Model
@model bhm_ar1(y, X1, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1)

    sig ~ Uniform(0.01, 5.0)

    rho ~ Uniform(-1, 1)
    b = Vector{TV}(undef, k1)

    for i = 1:k1
        b[i] ~ Normal(0, 3)
    end

    a ~ Normal(0,3)

    mu1 = rho .* X1[:, 1] .+ [X1[:, 1] .* Matrix(X1[:, 2:end]) X1[:,2]] * b .+ a

    for i = 1:n1
        y[i] ~ Normal(mu1[i], sqrt(sig))
    end

end

#=
MMRM - Time Effects with AR1, CS, Toeplitz, 
Log Trend  with AR1, CS, Toeplitz
=#

# MMRM Time Effects with AR1 Errors
@model mmrm_t_ar1(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t

    s_i ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)
    rho ~ Uniform(0.0, 0.999)

    b = Vector{TV}(undef, k)

    for i in 1:k
        b[i] ~ Normal(0.0, 5.0)
    end

    omega = [s_t * rho^abs(i - j) for i in 1:t, j in 1:t] .+ Diagonal(fill(s_i, t))

    xb = x * b

    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(xb[(i-1)*t+1:i*t], omega)
    end

end

# MMRM Time Effects with CS Errors
@model mmrm_t_cs(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t

    s_i ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)
    rho ~ Uniform(0.0, 0.999)

    b = Vector{TV}(undef, k)

    for i in 1:k
        b[i] ~ Normal(0.0, 5.0)
    end

    omega = Matrix{TV}(undef, t, t)

    for i in 1:t, j in 1:t
        if i == j
            omega[i, j] = s_t + s_i
        else
            omega[i, j] = s_t * rho
        end
    end

    xb = x * b

    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(xb[(i-1)*t+1:i*t], omega)
    end

end

#= # MMRM Time Effects with Toeplitz Errors
@model mmrm_t_t(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t

    s_i ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)

    r = Vector{TV}(undef, t-1)

    for i in 1:t-1
        r[i] ~ Uniform(0.0, 0.999)
    end

    b = Vector{TV}(undef, k)

    for i in 1:k
        b[i] ~ Normal(0.0, 2.0)
    end

    omega = Matrix{TV}(undef, t, t)

    for i in 1:t, j in 1:t
        if i == j
            omega[i, j] = s_t + s_i
        else
            omega[i, j] = s_t * r[abs(i - j)]
        end
    end

    xb = x * b

    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(xb[(i-1)*t+1:i*t], omega)
    end

end =#

# MMRM Log Trend with AR1 Errors
@model mmrm_l_ar1(y, X1, t, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1)

    ind = n1 ÷ t

    s_t ~ Uniform(0.01, 10.0)
    s_i ~ Uniform(0.01, 10.0)
    rho ~ Uniform(0.0, 0.999)

    omega = [s_t * rho^abs(i - j) for i in 1:t, j in 1:t] .+ Diagonal(fill(s_i, t))

    d1 ~ TruncatedNormal(-1, 1.0, -5, 0)

    b = Vector{TV}(undef, (k1 - 1))

    for i = 1:(k1-1)
        b[i] ~ Normal(0, 5)
    end

    mu1 = d1 .* X1[:, 1] .+ (Matrix(X1[:, 2:end])) * b

    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(mu1[(i-1)*t+1:i*t], omega)
    end
end

# MMRM Log Trend with CS Errors
@model mmrm_l_cs(y, X1, t, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1)

    ind = n1 ÷ t

    s_t ~ Uniform(0.01, 5.0)
    s_i ~ Uniform(0.01, 5.0)
    rho ~ Uniform(0.0, 0.999)

    omega = Matrix{TV}(undef, t, t)

    for i in 1:t, j in 1:t
        if i == j
            omega[i, j] = s_t + s_i
        else
            omega[i, j] = s_t * rho
        end
    end

    d1 ~ TruncatedNormal(-1, 1.0, -5, 0)

    b = Vector{TV}(undef, (k1 - 1))

    for i = 1:(k1-1)
        b[i] ~ Normal(0, 5)
    end

    mu1 = d1 .* X1[:, 1] .+ (Matrix(X1[:, 2:end])) * b

    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(mu1[(i-1)*t+1:i*t], omega)
    end
end

# MMRM Log Trend with CS Errors
@model te(y, X1, t, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1)

    sig ~ Uniform(0.01, 5.0)

    b = Vector{TV}(undef, k1)

    for i = 1:k1
        b[i] ~ Normal(0, 5)
    end

    mu1 = Matrix(X1) * b

    for i in 1:n1
        y[i] ~ Normal(mu1[i], sqrt(sig))
    end
end

#= # MMRM Log Trend with Toeplitz Errors
@model mmrm_l_t(y, X1, t, ::Type{TV}=Float64) where {TV} = begin
    n1, k1 = size(X1)

    ind = n1 ÷ t

    s_t ~ Uniform(0.01, 5.0)
    s_i ~ Uniform(0.01, 5.0)
    r = Vector{TV}(undef, t-1)

    for i in 1:t-1
        r[i] ~ Uniform(0.0, 0.999)
    end

    omega = Matrix{TV}(undef, t, t)

    for i in 1:t, j in 1:t
        if i == j
            omega[i, j] = s_t + s_i
        else
            omega[i, j] = s_t * r[abs(i - j)]
        end
    end

    d1 ~ TruncatedNormal(-1, 1.0, -5, 0)

    b = Vector{TV}(undef, (k1 - 1))

    for i = 1:(k1-1)
        b[i] ~ Normal(0, 1)
    end

    mu1 = d1 .* X1[:, 1] .+ (Matrix(X1[:, 2:end]) .* X1[:, 1]) * b

    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(mu1[(i-1)*t+1:i*t], omega)
    end
end =#

#= Data Generation from all 8 DGPs =#
# Individuals = 100, Time Measurements = 4, sig = 1.0, si = 1.0, st = 1.0, rho = 0.65, r1 = 0.65, r2 = 0.5, r3 = 0.4
# time effects = -1.0,-1.5,-2.5, d logt = -2.5, ar = 0.75, intercept = 0.0, beta = -0.5

# Global DGP variables
n = 100
n2 = 50
t = 4
M = 2000

fes = [0.0; -0.3; -1.0; -1.5; -2.5; -0.3; -0.3; -0.3] # Intercept beta t1 t2 t3 for the Time Effect dgps
b = -0.3 # Beta for the Ar and Log Trend dgps
ar = 0.75 # Ar coeffcient
d = -2.0 # Delta for Log Trend

sig = 1.0 # Ar, Log Trend IID DGP VARIANCE
si = 1.0 # measurement noise
st = 1.0 # random process noise 
r = [0.35; 0.2; 0.15] # Toeplitz covariances
rho = 0.65 # AR covariance and CS covariance


#= Log Trend AR DGP =#
y, x = ar_dgp(n, t, b, ar, sig)
y2, x2 = ar_dgp(n2, t, b, ar, sig) # Out of Sample

yt = reshape(y, (t, length(y) ÷ t))

plot(yt, label=false, title="AR DGP", dpi=600)
savefig("AR.png")

#= y
x

y = y_test
x = x_test
n = length(y) ÷ t =#

ts = [repeat(Matrix(I, t, t)[:, i], outer=n) for i in 1:t]
t_fes = reduce(hcat, ts)

#= Model Statements =#
# BHM - Spectral
dy = y .- repeat(y[1:4:end], inner=t)
# x_sp = [repeat(1:t, outer=length(y) ÷ t) x[:,2]]
x_sp = [repeat(1:t, outer=length(y) ÷ t) x]
model_sp = bhm_onefreq(dy, x_sp)
cc = sample(model_sp, NUTS(0.65), M)
table(cc)[2]
bh_sp = mean.(eachcol(table(cc)[1][:, end-3:end]))

mean((dy .- (bh_sp[2] .* sin.(bh_sp[1] .* x_sp[:, 1]) .+ bh_sp[3] .* cos.(bh_sp[1] .* x_sp[:, 1]) .+ x_sp[:, 2] .* bh_sp[4])) .^ 2)
# In Sample RMSE = 5.498781
# beta = -0.738, p-val = 0.0, True = -0.5
dy2 = y2 .- repeat(y2[1:4:end], inner=t)
# x_sp2 = [repeat(1:t, outer=n2) x2[:, 2]]
x_sp2 = [repeat(1:t, outer=n2) x2]
mean((dy2 .- (bh_sp[2] .* sin.(bh_sp[1] .* x_sp2[:, 1]) .+ bh_sp[3] .* cos.(bh_sp[1] .* x_sp2[:, 1]) .+ x_sp2[:, 2] .* bh_sp[4])) .^ 2)
# Out Sample RMSE = 6.365449


# BHM - Log Trend
# x_lt = [repeat(log.(1:t), outer=n) x[:,2]]
x_lt = [repeat(log.(1:t), outer=n) x]
dy = y .- repeat(y[1:4:end], inner=t)
model_lt = bhm_logt(dy, x_lt)
cc = sample(model_lt, NUTS(0.65), M)
table(cc)[2]

mean((dy .- x_lt[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_lt, 2)+1:end]))[1] .+ x_lt[:, 1] .* x_lt[:, 2] .* mean.(eachcol(table(cc)[1][:, end-size(x_lt, 2)+1:end]))[2]) .^ 2)
# In Sample RMSE = 2.077773
# beta = 0.2, p-val = 0.008????, True = -0.5
# n2 = length(x2)÷4

# x_lt2 = [repeat(log.(1:t), outer=n2) x2[:, 2]]
x_lt2 = [repeat(log.(1:t), outer=n2) x2]
dy2 = y2 .- repeat(y2[1:4:end], inner=t)
mean((dy2 .- x_lt2[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_lt2, 2)+1:end]))[1] .+ x_lt2[:, 1] .* x_lt2[:, 2] .* mean.(eachcol(table(cc)[1][:, end-size(x_lt2, 2)+1:end]))[2]) .^ 2)
# Out Sample RMSE = 2.180812

# BHM - AR1
y
y_ar = y[Not(1:4:end)]
# x_ar = [y[Not(4:4:end)] x[Not(4:4:end),2]]
x_ar = [y[Not(4:4:end)] x[Not(4:4:end)]]
model_ar = bhm_ar1(y_ar, x_ar)
cc = sample(model_ar, NUTS(0.65), M)
table(cc)[2]

mean((y_ar .- x_ar[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_ar, 2)+1:end]))[1]) .^ 2)
# In Sample RMSE = 2.108816
# beta = -0.124, p-val = 0.167, True = -0.5

y_ar2 = y2[Not(1:4:end)]
# x_ar2 = [y2[Not(4:4:end)] x2[Not(4:4:end),2]]
x_ar2 = [y2[Not(4:4:end)] x2[Not(4:4:end)]]

mean((y_ar2 .- x_ar2 * mean.(eachcol(table(cc)[1][:, end-size(x_ar2, 2)+1:end]))) .^ 2)
# Out Sample RMSE = 2.341054

# MMRM - Time Effects CS
# X = [ones(n * t) x[:, 2] t_fes[:, 2:end]]
X = [ones(n * t) x t_fes[:, 2:end] x .* t_fes[:, 2:end]]
model_t_cs = mmrm_t_cs(y, X, t)
cc = sample(model_t_cs, NUTS(0.65), M)
table(cc)[2]

mean((y .- X * mean.(eachcol(table(cc)[1][:, end-size(X, 2)+1:end]))) .^ 2)
# In Sample RMSE = 2.283926
# beta = -0.368, p-val = 0.009, True = -0.5

# X2 = [ones(n2 * t) x2[:, 2] t_fes[1:n2*t, 2:end]]
X2 = [ones(n2 * t) x2 t_fes[1:n2*t, 2:end]]

mean((y2 .- X2 * mean.(eachcol(table(cc)[1][:, end-size(X2, 2)+1:end]))) .^ 2)
# Out Sample RMSE = 2.652892

# MMRM - Time Effects AR
# X = [ones(n * t) x[:, 2] t_fes[:, 2:end]]
X = [ones(n * t) x t_fes[:, 2:end]]
model_t_ar = mmrm_t_ar1(y, X, t)
cc = sample(model_t_ar, NUTS(0.65), M)
table(cc)[2]

mean((y .- X * mean.(eachcol(table(cc)[1][:, end-size(X, 2)+1:end]))) .^ 2)
# In Sample RMSE = 2.283669
# beta = -0.374, p-val = 0.005, True = -0.5
# X2 = [ones(n2 * t) x2[:, 2] t_fes[1:n2*t, 2:end]]
X2 = [ones(n2 * t) x2 t_fes[1:n2*t, 2:end]]

mean((y2 .- X2 * mean.(eachcol(table(cc)[1][:, end-size(X2, 2)+1:end]))) .^ 2)
# Out Sample RMSE = 2.653316

# MMRM - Log Trend CS
dy = y .- repeat(y[1:t:end], inner=t)
# x_l_cs = [repeat(log.(1:t), outer=n) x[:, 2]]
x_l_cs = [repeat(log.(1:t), outer=n) x]
model_l_cs = mmrm_l_cs(dy, x_l_cs, t)
cc = sample(model_l_cs, NUTS(0.65), M)
table(cc)[2]

mean((dy .- x_l_cs[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_cs, 2)+1:end]))[1] .+ x_l_cs[:, 1] .* x_l_cs[:, 2] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_cs, 2)+1:end]))[2]) .^ 2)
# In Sample RMSE = 2.075241
# beta = 0.196, p-val = 0.015. True = -0.5

dy2 = y2 .- repeat(y2[1:t:end], inner=t)
# x_l_cs2 = [repeat(log.(1:t), outer=n2) x2[:, 2]]
x_l_cs2 = [repeat(log.(1:t), outer=n2) x2]
mean((dy2 .- x_l_cs2[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_cs2, 2)+1:end]))[1] .+ x_l_cs2[:, 1] .* x_l_cs2[:, 2] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_cs2, 2)+1:end]))[2]) .^ 2)
# Out Sample RMSE = 2.162925

# MMRM - Log Trend AR1
dy = y .- repeat(y[1:t:end], inner=t)
# x_l_ar = [repeat(log.(1:t), outer=n) x[:,2]]
x_l_ar = [repeat(log.(1:t), outer=n) x]
model_l_ar = mmrm_l_ar1(dy, x_l_ar, t)
cc = sample(model_l_ar, NUTS(0.65), M)
table(cc)[2]

mean((dy .- x_l_ar[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_ar, 2)+1:end]))[1] .+ x_l_ar[:, 1] .* x_l_ar[:, 2] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_ar, 2)+1:end]))[2]) .^ 2)
# In Sample RMSE = 2.076148
# beta = 0.197, p-val = 0.024. True = -0.5

dy2 = y2 .- repeat(y2[1:t:end], inner=t)
# x_l_ar2 = [repeat(log.(1:t), outer=n2) x2[:, 2]]
x_l_ar2 = [repeat(log.(1:t), outer=n2) x2]
mean((dy2 .- x_l_ar2[:, 1] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_ar2, 2)+1:end]))[1] .+ x_l_ar2[:, 1] .* x_l_ar2[:, 2] .* mean.(eachcol(table(cc)[1][:, end-size(x_l_ar2, 2)+1:end]))[2]) .^ 2)
# In Sample RMSE = 2.160160


#### TORDIA Application
rest = DataFrame(CSV.File("combined_data_final_9_2020.csv"))
tads = DataFrame(CSV.File("tads_wtht.csv"))
torhtwt = DataFrame(CSV.File("Tordia htwt.csv"))
cdr = DataFrame(CSV.File("Tordia_Final.csv"))

tor_ht = torhtwt[:, 1]
tor_wt = torhtwt[:, 2]

replace!(tor_ht, 999 => missing)
replace!(tor_wt, 999 => missing)

torht = tor_ht .* 2.54
torwt = tor_wt .* 0.45

tordia = (rest[rest.cams.==0, :])[(rest[rest.cams.==0, :]).tads.==0, :]
tordia.ht = repeat(torht, inner=4)
tordia.wt = repeat(torwt, inner=4)

tordia.cdrs = cdr.cdrsrtot

@show(names(tads))
tads.weeks

@show(names(tordia))

tads.nonwh = ifelse.(tads.white .== 1, 0, 1)
tordia.ht

tads.sex = ifelse.(tads.gender .== "F", 1, 0)
tordia.age = floor.(tordia.agem ./ 12)

t1 = tads[:, [:weeks, :cgis, :sex, :age, :anx, :snri, :nonwh, :height_met, :weight_met, :tads_cbt]]
t2 = tordia[:, [:wk, :cgis, :sex, :age, :anx, :snri, :nonwh, :ht, :wt, :cbt]]

rename!(t1, [1 => :wk, 8 => :ht, 9 => :wt, 10 => :cbt])

t1.tordia = zeros(size(t1, 1))
t2.tordia = ones(size(t2, 1))
t1.tads = ones(size(t1, 1))
t2.tads = zeros(size(t2, 1))

tt2 = vcat(t2, t1)

tt2.lt = repeat(log.([1, 2, 3, 4]), outer=Int64(size(tt2, 1) / 4))

tt2.bmi = tt2.wt ./ (tt2.ht ./ 100) .^ 2

tt2[tt2.lt.==0, :cgis]
tt2.base = repeat(tt2[tt2.lt.==0, :cgis], inner=4)

tt = tt2[tt2.base.>2, :]

tt.baselt = tt.base .* tt.lt
tt.sexlt = tt.sex .* tt.lt
tt.anxlt = tt.anx .* tt.lt
tt.snrilt = tt.snri .* tt.lt
tt.ssrilt = (1 .- tt.snri) .* tt.lt
tt.nonwhlt = tt.nonwh .* tt.lt
tt.agelt = tt.age .* tt.lt
tt.bmilt = tt.bmi .* tt.lt
tt.nonwhlt = tt.nonwh .* tt.lt
tt.torlt = tt.tordia .* tt.lt
tt.tadslt = tt.tordia .* tt.lt
tt.resp = ifelse.(tt.cgis .<= 2, 1, 0)

tt.anxsnrilt = tt.anxlt .* tt.snri
tt.sexsnrilt = tt.sexlt .* tt.snri

tt.chngbase = tt.cgis .- tt.base

tt.cbtlt = tt.cbt .* tt.lt

names(tt)

tortd = tt[:, [:cgis, :lt, :nonwh]]

samp = sample(Random.seed!(12), 1:541, floor(Int, 541 * 0.6), replace=false, ordered=true)

train = (repeat(samp, inner=4) .* 4) .- repeat([3; 2; 1; 0], outer=324)
test = repeat(deleteat!(collect(1:541), samp), inner=4) .* 4 .- repeat([3; 2; 1; 0], outer=541 - 324)

y_test = tortd[train, :cgis]
x_test = tortd[train, :nonwh]
y2 = tortd[test, :cgis]
x2 = tortd[test, :nonwh]

