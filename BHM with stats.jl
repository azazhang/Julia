using CSV, DataFrames, Turing, StatsBase, StatsPlots, Plots, LinearAlgebra, Random, PrettyTables, Distributions, MCMCChains

# Set random seed for reproducibility
Random.seed!(1234)

# Load the dataset
df = CSV.read("C:/Users/hudso/OneDrive/ECON 9011/Data/airlines.csv", DataFrame)

# Extract variables
ID = df.I
T = 15  # Number of time periods
Y = df.C
Q = df.Q
PF = df.PF
LF = df.LF

# Convert IDs to integers
airlines = unique(ID)
airline_map = Dict(a => i for (i, a) in enumerate(airlines))
ID_int = [airline_map[a] for a in ID]
N = length(airlines)

# Define the non-hierarchical model
@model function panel_data_model_no_hierarchical(Y, Q, PF, LF)
    alpha ~ Normal(0, 10)
    beta_Q ~ Normal(0, 1)
    beta_PF ~ Normal(0, 1)
    beta_LF ~ Normal(0, 1)
    sigma ~ InverseGamma(2, 3)
    mu = alpha .+ beta_Q .* Q .+ beta_PF .* PF .+ beta_LF .* LF
    Y .~ Normal.(mu, sigma)
end

# Define the hierarchical model
@model function panel_data_model_with_hierarchical(Y, Q, PF, LF, ID, N)
    mu_alpha ~ Normal(0, 10)
    tau_alpha ~ InverseGamma(2, 3)
    beta_Q ~ Normal(0, 1)
    beta_PF ~ Normal(0, 1)
    beta_LF ~ Normal(0, 1)
    sigma ~ InverseGamma(2, 3)
    alpha_individual = Vector{Real}(undef, N)
    for i in 1:N
        alpha_individual[i] ~ Normal(mu_alpha, tau_alpha)
    end
    mu = alpha_individual[ID] .+ beta_Q .* Q .+ beta_PF .* PF .+ beta_LF .* LF
    Y .~ Normal.(mu, sigma)
end

# Sample from the non-hierarchical model
model_no_hierarchical = panel_data_model_no_hierarchical(Y, Q, PF, LF)
chain_no_hierarchical = sample(model_no_hierarchical, NUTS(), 2000; warmup=1000)

# Sample from the hierarchical model
model_with_hierarchical = panel_data_model_with_hierarchical(Y, Q, PF, LF, ID_int, N)
chain_with_hierarchical = sample(model_with_hierarchical, NUTS(), 2000; warmup=1000)

### Utility functions

function pval(vector)
    s = size(vector, 1)
    pv = 2 * (1 - sum(vector .< 0.0)/s)
    pv = pv > 1.0 ? 2 - pv : pv
    return round(pv, digits=3)
end

function pval_matrix(x::Matrix{Float64})
    # Returns vector of p-values if you have a matrix of samples for multiple params
    ps = similar(x, size(x,2))
    for i in 1:size(x, 2)
        ps[i] = pval(x[:, i])
    end
    return ps
end

function hpdi(x::Vector{T}; alpha=0.05) where {T<:Real}
    n = length(x)
    m = max(1, ceil(Int, alpha * n))
    y = sort(x)
    a = y[1:m]
    b = y[(n-m+1):n]
    _, i = findmin(b .- a)
    return [a[i], b[i]]
end

function create_summary_table(chain)
    params = chain.name_map.parameters
    s = size(chain.value.data)[1]
    x = zeros(s, length(params))
    for i in 1:length(params)
        x[:, i] = parent(chain[params[i]])[:]
    end

    mx = round.(mean(x, dims=1)'[:], digits=3)
    stx = round.(std(x, dims=1)'[:], digits=3)
    px = pval_matrix(x)
    qx = [round.(quantile(x[:, i], [0.025, 0.975]), digits=3) for i in 1:size(x, 2)]
    hx = [round.(hpdi(x[:, i]), digits=3) for i in 1:size(x, 2)]

    # Convert everything to strings
    table_matrix = Array{String}(undef, length(params), 6)
    for i in 1:length(params)
        table_matrix[i, :] = [
            string(params[i]),
            string(mx[i]),
            string(stx[i]),
            string(px[i]),
            "[" * string(qx[i][1]) * ", " * string(qx[i][2]) * "]",
            "[" * string(hx[i][1]) * ", " * string(hx[i][2]) * "]"
        ]
    end

    header = ["Parameter", "Mean", "Std", "P-Val", "CI-95%", "HPD-95%"]
    # Pass header as a keyword argument:
    pretty_table(table_matrix; header=header, alignment=:c)
end




# Print summaries
println("Summary for Non-Hierarchical Model:")
create_summary_table(chain_no_hierarchical)

println("\nSummary for Hierarchical Model:")
create_summary_table(chain_with_hierarchical)




####################### POSTERIOR STUFF  ################################

using StatsPlots

# Get parameter lists and find common parameters
params_no_hier = chain_no_hierarchical.name_map.parameters
params_hier = chain_with_hierarchical.name_map.parameters
common_params = intersect(params_no_hier, params_hier)

# For each parameter, overlay the densities of the two models
for p in common_params
    samples_no_hier = parent(chain_no_hierarchical[p])[:]
    samples_hier = parent(chain_with_hierarchical[p])[:]

    plt = density(samples_no_hier; label="Non-Hierarchical", xlabel="Parameter Value", ylabel="Density", title="Posterior Distribution Comparison for $(p)")
    density!(plt, samples_hier; label="Hierarchical")

    display(plt)
end


###########################         EVERYTHING ABOVE HERE WORKS        TESTING SHIT BELOW           ###############################




