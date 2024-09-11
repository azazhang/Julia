 
# Comparison of means taking into account potential nonzero covariance between
# the variables and allowing variances to differ

using Distributions, DataFrames, Turing, StatsBase, StatsPlots, Plots, LinearAlgebra, Random, CSV, PrettyTables 

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

# Turing model:
@model means(yx, ::Type{TV}=Float64) where {TV} = begin
    # assumes a balanced sample (n_x = n_y)
    n = length(yx[:,1]) # TORDIA 
    sigy ~ Uniform(0.001, 3.0)
    sigx ~ Uniform(0.001, 3.0)
    rho ~ Truncated(Normal(0, 5), -0.99, 0.99)
    muy ~ Normal(0, 5)
    mux ~ Normal(0, 5)
    mu = [muy; mux] 
    om = [sigy^2 rho*sigy*sigx; rho*sigy*sigx sigx^2]
    for i = 1:n
        yx[i,:] ~ MvNormal(mu, om)
    end
end

# Generate data:
Random.seed!(12894)
n = 30
rho = -0.6
sigy = 0.3
sigx = 0.5
muy = 1.1
mux = 1.0

om = [sigy^2 rho*sigy*sigx; rho*sigy*sigx sigx^2]
xy = rand(MvNormal([muy; mux], om),n)' 

y = xy[:,1]
x = xy[:,2]

mean(xy[:,1])
mean(xy[:,2])

std(xy[:,1])
std(xy[:,2])

cor(xy)

plot(x, y, st=:scatter)

#  Estimate
M = 3000

yx = [y x]
model = means(yx)
@time cc = sample(model, NUTS(0.65), M)
table(cc)[2]

dif_xy = table(cc)[1].muy .- table(cc)[1].mux

plot(table(cc)[1].rho, st=:density, label=false)

plot(table(cc)[1].muy, st=:density, label="mu_y", fill = true, alpha = 0.4)
plot!(table(cc)[1].mux, st=:density, label="mu_x", fill = true, alpha = 0.4)
plot!(dif_xy, st=:density, label="mu_y - mu_x", fill = true, alpha = 0.4)
vline!([0.0], linewidth=3, linecolor=:black, label=false)

# two-tailed p-value
pval(dif_xy)
# one tail area
length(dif_xy[dif_xy .< 0.0])/length(dif_xy)

# Coefficients are correlated
plot(table(cc)[1].muy, table(cc)[1].mux, st = :scatter, label=false, alpha=0.3, xlabel="mu_y", ylabel="mu_x")


## Comparison of means ignoring covariance

function post_t(x)
    n_x = length(x)
    mn_x = mean(x)
    sd_x = std(x)
    x_draws = rand(TDist(n_x-1), 10^6).*(sd_x/sqrt(n_x)) .+ mn_x
    return x_draws, mn_x, sd_x, n_x
end

y_draws, mn_y, sd_y, n_y = post_t(y)
x_draws, mn_x, sd_x, n_x = post_t(x)

plot(y_draws, st=:density, label="posterior for mean of y")
plot!(x_draws, st=:density, label="posterior for mean of x")

# We can compute the posterior density for the DIFFERENCE of the 
# means by looking at the difference of the MC draws!
diff_yx = y_draws .- x_draws
plot!(diff_yx, st=:density, label="posterior for difference of means")
vline!([0.0], label=false, linecolor=:black, linewidth=2)

# Compute a "Bayesian p-value"
pval2 = length(diff_yx[diff_yx .< 0.0])/length(diff_yx)


## Assuming same variance (standard ANOVA)

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

function print_regression(blinreg, b2se, pvals, Rsq, cril, criu, coefnames)
    println(" Variable        coeff     s.e.     pval        CrI")
    for i = 1:length(blinreg)
        println(coefnames[i], "         ", round(blinreg[i], digits = 3), "     ", round(b2se[i], digits = 3), "     ", round(pvals[i], digits = 4), "     ", round(criu[i], digits = 4), "    ", round(cril[i], digits = 4))
    end
    println("Rsquared = ", round(Rsq, digits = 3))
end

yy = vcat(y, x)
d = vcat(ones(length(y)), zeros(length(x)))
X = [ones(length(yy)) d]
blinreg, b2se, tstats, pvalsr, Rsq, sigma2_hat, cril, criu = linreg(X,yy)
pvalsr[2]

coefs = ["intecept" "diff_ind"]
print_regression(blinreg, b2se, pvals, Rsq, cril, criu, coefs)