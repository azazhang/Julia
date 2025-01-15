# Bayesian MMRM example
using DataFrames, Turing, StatsBase, StatsPlots, Plots, LinearAlgebra, Random, CSV, PrettyTables


# some useful functions
function pval(vector)
    if typeof(vector) == Matrix{Float64}
        pval = zeros(size(vector, 2))
        for i = 1:size(vector, 2)
            pval[i] = 2 * (1 - sum(vector[:, i] .< 0.0) / (size(vector[:, i], 1)))
            if pval[i] > 1.0
                pval[i] = round(2 - pval[i], digits=3)
            else
                pval[i] = round(pval[i], digits=3)
            end
        end
    else
        pval = 2 * (1 - sum(vector .< 0.0) / (size(vector, 1)))
        if pval > 1.0
            pval = round(2 - pval, digits=3)
        else
            pval = round(pval, digits=3)
        end
    end
    return pval
end


function hpdi(x::Vector{T}; alpha=0.05) where {T<:Real}
    n = length(x)
    m = max(1, ceil(Int, alpha * n))

    y = sort(x)
    a = y[1:m]
    b = y[(n-m+1):n]
    _, i = findmin(b - a)

    return [a[i], b[i]]
end

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


### DATA GENERATING PROCESSES

# Generate pseudo-data with AR(1) error structure:
# Vikram's DGP:
# function mix_dgp(n, t, fs, si, st, rho)
#     x = randn(n)   # generate explanatory variables
#     xt = repeat(x, inner=t)
#     ts = [repeat(Matrix(I, t, t)[:, i], outer=n) for i in 1:t]
#     t_fes = reduce(hcat, ts)  # time dummies for each individual
#     xb = [t_fes[:, 2:end] xt] * fs
#     # y0 = rand(5:7, n)
#     y0 = ones(n) .+ 6
#     yt = zeros(n * t)
#     for i in 2:n*t
#         if rem(i, t) == 1
#             yt[i] = 0.0
#         else
#             yt[i] = xb[i]
#         end
#     end
#     om = [st * rho^abs(i - j) for i in 1:t, j in 1:t]
#     e = zeros(n * t)
#     for i in 1:n
#         e[(i-1)*t+1:i*t] .= rand(MvNormal(zeros(t), om)) .+ repeat([si * randn()], outer=t)
#     end
#     # e[1:t:end] .= 0.0
#     return repeat(y0, inner=t) .+ yt .+ e, [t_fes[:, 2:end] xt]
# end

# Simpler DGP:
function mixed_dgp2(s_i, s_t, rho, t, n, b)
    xi = randn(n, length(b)) # = size(x)
    x = repeat(xi, inner=(t, 1))
    omega = [s_t * rho^abs(i - j) for i in 1:t, j in 1:t] .+ Diagonal(fill(s_i, t))
    xb = x * b
    y = zeros(n * t)
    for i in 1:n
        y[(i-1)*t+1:i*t] = rand(MvNormal(xb[(i-1)*t+1:i*t], omega))
    end
    return y, x
end


#### Model specification for estimation:
@model bhm_simple(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t
    #   s_i ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)
    rho ~ Uniform(0.0, 0.999)
    mb ~ Normal(1.0, 1.0)    # hierarchical prior
    b = Vector{TV}(undef, k)
    for i in 1:k
        b[i] ~ Normal(mb, 1.0)
    end
    xb = x * b
    s = s_t * 1 * Matrix(I, t, t)
    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(xb[(i-1)*t+1:i*t], s)
    end
end

@model bhm_bhm(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t
    si ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)
    rho ~ Uniform(0.0, 0.999)
    d ~ Normal(0.0, 2.0)

    di = Vector{TV}(undef, n)
    for i in 1:n
        di[i] ~ Normal(d, si)
    end

    b = Vector{TV}(undef, k)
    for i in 1:k
        b[i] ~ Normal(0.0, 2.0)
    end
    trend = collect(1:t)

    xb = x * b + trend * di
    s = s_t * 1 * Matrix(I, t, t)
    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(xb[(i-1)*t+1:i*t] + trend * di[i], s)
    end
end


@model bhm_mixed(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t
    s_i ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)
    rho ~ Uniform(0.0, 0.999)
    b = Vector{TV}(undef, k)
    for i in 1:k
        b[i] ~ Normal(0.0, 2.0)
    end
    omega = [s_t * rho^abs(i - j) for i in 1:t, j in 1:t] .+ Diagonal(fill(s_i, t))
    xb = x * b
    for i in 1:ind
        y[(i-1)*t+1:i*t] ~ MvNormal(xb[(i-1)*t+1:i*t], omega)
    end
end

@model bhm_cs(y, x, t, ::Type{TV}=Float64) where {TV} = begin
    n, k = size(x)
    ind = n ÷ t
    s_i ~ Uniform(0.001, 5.0)
    s_t ~ Uniform(0.001, 5.0)
    rho ~ Uniform(0.0, 0.999)
    b = Vector{TV}(undef, k)
    for i in 1:k
        b[i] ~ Normal(0.0, 2.0)
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


### Data Generation
Random.seed!(123)
n = 30
t = 4
rho = 0.7
si = 0.2
st = 0.2
b = [1.0; 2.0]
y2, x2 = mixed_dgp2(si, st, rho, t, n, b)

y = reshape(y2, (t, n))
plot(y, label=false)

M = 2000
model_m = bhm_mixed(y2, x2, t)

model_m = bhm_simple(y2, x2, t)

@time cc = sample(model_m, NUTS(0.65), M)


table(cc)[2]



1 + 1 == sqrt(2)




####################################

#### Using Vikram's DGP:

n = 25   # number of individuals
t = 4   # number of time n_time_periods

# DGP parameter values
si = 0.1
st = 0.5
rho = 0.003
fs = [0.5; 1.0; 1.5; 1.5] #ones(t)  #[-0.8; -1.5; -2.5; -0.5]

yt, xt = mix_dgp(n, t, fs, si, st, rho)

subject = repeat(collect(1:t), outer=n)
### Plotting data
y = reshape(yt, (t, n))

plot(y, label=false)

M = 2000
model_m = bhm_mixed(yt, xt, t)

@time cc = sample(model_m, NUTS(0.65), M)
table(cc)[2]



