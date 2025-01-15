

using CSV, DataFrames, Turing, StatsBase, StatsPlots, Plots, LinearAlgebra, Random, PrettyTables, Distributions, MCMCChains

##################### MAIN MODEL ###############################


# Set random seed for reproducibility
Random.seed!(1234)

####################    DATA IMPORT/INFO 

# Load the dataset
df = CSV.read("airlines.csv", DataFrame)

#Data Set contains Cost Data for U.S. Airlines, 90 Observations On 6 Firms For 15 Years, 1970-1984
# Data from: https://www.kaggle.com/datasets/sandhyakrishnan02/paneldata?resource=download

#  I = Airline
#  T = Year
#  Q = Output, in revenue passenger miles
#                Revenue passenger miles (RPMs) are a metric used in the airline industry to measure the number of miles traveled by paying passengers. 
#                One RPM is equal to one paying passenger traveling one mile.
#  PF = Fuel price
#  LF = Load factor, the average capacity utilization of the fleet 
#                 No idea what load factor was. See this for basic idea: https://www.investopedia.com/ask/answers/041515/how-can-i-use-load-factor-indicator-profitability-airline-industry.asp
#  
#  C = Total cost, in $1000




######################### BASIC SUMMARY STATS

using Statistics, PrettyTables


# Extract variables
ID = df.I
T = 15  # Number of time periods
Y = df.C
Q = df.Q
PF = df.PF
LF = df.LF


# Original variables in the dataset
variables = Dict(
    "Y (Total Cost)" => Y,
    "Q (Output)" => Q,
    "PF (Fuel Price)" => PF,
    "LF (Load Factor)" => LF
)

# Function to calculate summary statistics
function calculate_summary_stats(variables)
    stats = DataFrame(
        Variable = String[],
        Mean = Float64[],
        StdDev = Float64[],
        Min = Float64[],
        Max = Float64[]
    )
    
    for (name, values) in variables
        push!(stats, (
            Variable = name,
            Mean = mean(values),
            StdDev = std(values),
            Min = minimum(values),
            Max = maximum(values)
        ))
    end
    
    return stats
end

# Generate and display summary statistics
summary_stats = calculate_summary_stats(variables)
pretty_table(summary_stats, header=["Variable", "Mean", "StdDev", "Min", "Max"], alignment=:c)






################ DATA TRANSFORMATION





# Log-transform the dependent variable Y
Y_log = log.(Y)  # Apply element-wise log to Y

#### begin of shitty stuff ####

# Y_log = Y / 100000

##### end of shitty stuff ####

# Standardize predictors
Q_std = (Q .- mean(Q)) ./ std(Q)
PF_std = (PF .- mean(PF)) ./ std(PF)
LF_std = (LF .- mean(LF)) ./ std(LF)

# Convert IDs to integers
airlines = unique(ID)
airline_map = Dict(a => i for (i, a) in enumerate(airlines))
ID_int = [airline_map[a] for a in ID]
N = length(airlines)



#############      DEFINING THE PANEL/HIERARCHICAL MODELS 


# Define the non-hierarchical model with diagonal covariance on log(Y) and standardized predictors
@model function panel_data_model_no_hierarchical_with_diag(Y_log, Q_std, PF_std, LF_std, ID, N, T)
    alpha ~ Normal(0, 5)
    beta_Q ~ Normal(0, 0.5)
    beta_PF ~ Normal(0, 0.5)
    beta_LF ~ Normal(0, 0.5)
    s_i ~ Uniform(0.1, 1.0)
    s_t ~ Uniform(0.1, 1.0)
    sigma ~ InverseGamma(3, 3)
    rho ~ Uniform(0.0, 1.0)

    omega = [s_t * rho^abs(i - j) for i in 1:T, j in 1:T] .+ Diagonal(fill(s_i, T))

    for i in 1:N
        idx = findall(ID .== i)
        X = hcat(Q_std[idx], PF_std[idx], LF_std[idx])
        mu = alpha .+ X * [beta_Q, beta_PF, beta_LF]
        # Model log(Y), assuming a normal distribution on the log scale
        Y_log[idx] ~ MvNormal(mu, omega)
    end
end

######## shitty stuff below ########
@model function panel_data_model_no_hierarchical_with_diag_n_lag(Y_log, Q_std, PF_std, LF_std, ID, N, T)
    alpha ~ Normal(0, 5)
    beta_Q ~ Normal(0, 0.5)
    beta_PF ~ Normal(0, 0.5)
    beta_LF ~ Normal(0, 0.5)
    phi ~ Normal(0, 5)
    s_i ~ Uniform(0.1, 1.0)
    s_t ~ Uniform(0.1, 1.0)
    sigma ~ InverseGamma(3, 3)
    rho ~ Uniform(-1.0, 1.0)
    Y0 = Vector{Real}(undef, N)
    Y_lag = Vector{Real}(undef, N*T)
    for i in 1:N
        mu = Y_log[findfirst(ID .== i)]
        sd = std(Y_log[findall(ID .== i)])
        Y0[i] ~ Normal(mu, sd)
        Y_lag[(i-1)*T+1] = Y0[i]
        Y_lag[(i-1)*T+2:i*T] = Y_log[(i-1)*T+1:i*T-1]
    end

    omega = [s_t * rho^abs(i - j) for i in 1:T, j in 1:T] .+ Diagonal(fill(s_i, T))

    for i in 1:N
        idx = findall(ID .== i)
        X = hcat(Q_std[idx], PF_std[idx], LF_std[idx], Y_lag[idx])
        mu = alpha .+ X * [beta_Q, beta_PF, beta_LF, phi]
        # Model log(Y), assuming a normal distribution on the log scale
        Y_log[idx] ~ MvNormal(mu, omega)
    end
end
######## end of shitty stuff  ########

# Define the hierarchical model with diagonal covariance on log(Y) and standardized predictors
@model function panel_data_model_with_hierarchical_and_diag(Y_log, Q_std, PF_std, LF_std, ID, N, T)
    mu_alpha ~ Normal(0, 5)
    tau_alpha ~ InverseGamma(3, 3)
    beta_Q ~ Normal(0, 0.5)
    beta_PF ~ Normal(0, 0.5)
    beta_LF ~ Normal(0, 0.5)
    s_i ~ Uniform(0.1, 1.0)
    s_t ~ Uniform(0.1, 1.0)
    sigma ~ InverseGamma(3, 3)
    rho ~ Uniform(0.0, 1.0)

    alpha_individual = Vector{Real}(undef, N)
    for i in 1:N
        alpha_individual[i] ~ Normal(mu_alpha, tau_alpha)
    end

    omega = [s_t * rho^abs(i - j) for i in 1:T, j in 1:T] .+ Diagonal(fill(s_i, T))

    for i in 1:N
        idx = findall(ID .== i)
        X = hcat(Q_std[idx], PF_std[idx], LF_std[idx])
        mu = alpha_individual[i] .+ X * [beta_Q, beta_PF, beta_LF]
        Y_log[idx] ~ MvNormal(mu, omega)
    end
end



################### SAMPLING 

# Sample from the non-hierarchical model with standardized predictors and log(Y)
model_no_hier_with_diag = panel_data_model_no_hierarchical_with_diag(Y_log, Q_std, PF_std, LF_std, ID_int, N, T)
chain_no_hier_with_diag = sample(model_no_hier_with_diag, NUTS(), 2000; warmup=1000)

# Sample from the hierarchical model with standardized predictors and log(Y)
model_with_hierarchical_and_diag = panel_data_model_with_hierarchical_and_diag(Y_log, Q_std, PF_std, LF_std, ID_int, N, T)
chain_with_hierarchical_and_diag = sample(model_with_hierarchical_and_diag, NUTS(), 2000; warmup=1000)

# Sample from the non-hierarchical model with lag and standardized predictors and log(Y)
model_no_hier_with_diag_n_lag = panel_data_model_no_hierarchical_with_diag_n_lag(Y_log, Q_std, PF_std, LF_std, ID_int, N, T)
chain_no_hier_with_diag_n_lag = sample(model_no_hier_with_diag_n_lag, NUTS(), 2000; warmup=1000)


##############################  FUNCTIONS FOR SUMMARIES AND COMPARISONS


function pval(vector)
    s = size(vector, 1)
    pv = 2 * (1 - sum(vector .< 0.0) / s)
    pv = pv > 1.0 ? 2 - pv : pv
    return round(pv, digits=3)
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
    px = [pval(x[:, i]) for i in 1:size(x, 2)]
    qx = [round.(quantile(x[:, i], [0.025, 0.975]), digits=3) for i in 1:size(x, 2)]
    hx = [round.(hpdi(x[:, i]), digits=3) for i in 1:size(x, 2)]

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
    pretty_table(
        table_matrix;
        header=header,
        alignment=:c,
        crop=:none  # Disable cropping to ensure all rows are displayed
    )
end

function compute_residuals_hier(chain, Y_log, Q_std, PF_std, LF_std, ID, N, T)
    # Extract posterior means for parameters
    mu_alpha = mean(parent(chain[:mu_alpha])[:])
    tau_alpha = mean(parent(chain[:tau_alpha])[:])
    beta_Q = mean(parent(chain[:beta_Q])[:])
    beta_PF = mean(parent(chain[:beta_PF])[:])
    beta_LF = mean(parent(chain[:beta_LF])[:])
    
    # Calculate individual intercepts
    alpha_individual = [mu_alpha + tau_alpha * randn() for i in 1:N]
    
    # Compute residuals
    residuals = zeros(length(Y_log))
    for i in 1:N
        idx = findall(ID .== i)
        X = hcat(Q_std[idx], PF_std[idx], LF_std[idx])
        mu = alpha_individual[i] .+ X * [beta_Q, beta_PF, beta_LF]
        residuals[idx] = Y_log[idx] .- mu
    end
    
    return residuals
end


###################### PRINTING RESULTS + POSTERIORS



# Print summaries
println("\nSummary for Non-Hierarchical Model with Diagonal Covariance (Log(Y), Standardized Predictors):")
create_summary_table(chain_no_hier_with_diag)

println("\nSummary for Hierarchical Model with Diagonal Covariance (Log(Y), Standardized Predictors):")
create_summary_table(chain_with_hierarchical_and_diag)

println("\nSummary for Non-Hierarchical Model with Diagonal Covariance and Lagged Dependent Variable (Log(Y), Standardized Predictors):")
create_summary_table(chain_no_hier_with_diag_n_lag)

# Posterior Comparisons
println("\nComparing Posterior Distributions Across Models")
params_no_hier = chain_no_hier_with_diag.name_map.parameters
params_hier = chain_with_hierarchical_and_diag.name_map.parameters
common_params = intersect(params_no_hier, params_hier)

# Overlay posterior distributions
for p in common_params
    samples_no_hier = parent(chain_no_hier_with_diag[p])[:]
    samples_hier = parent(chain_with_hierarchical_and_diag[p])[:]

    plt = density(samples_no_hier; label="Non-Hierarchical", xlabel="Parameter Value", ylabel="Density", title="Posterior Comparison for $(p)")
    density!(plt, samples_hier; label="Hierarchical")
    display(plt)
end










################################### "EXPERIMENTAL" STUFF


#################################### AUTOCORRLEATION STUFF

using StatsBase, StatsPlots

# Compute autocorrelation for residuals
function compute_acf(residuals, max_lag)
    mean_res = mean(residuals)
    var_res = var(residuals)
    acf = [sum((residuals[1:end-l] .- mean_res) .* (residuals[1+l:end] .- mean_res)) / 
           ((length(residuals) - l) * var_res) for l in 0:max_lag]
    return acf
end

# Define the maximum lag
max_lag = 10  # Adjust as necessary

# Compute residuals (assuming residuals_hier contains the residuals)
residuals_hier = compute_residuals_hier(chain_with_hierarchical_and_diag, Y_log, Q_std, PF_std, LF_std, ID, N, T)

# Compute autocorrelations
acf_values = compute_acf(residuals_hier, max_lag)

# Plot autocorrelation
bar(0:max_lag, acf_values, xlabel="Lag", ylabel="Autocorrelation", title="Residual Autocorrelation", legend=false)









println(chain_with_hierarchical_and_diag)
println(chain_with_hierarchical_and_diag.name_map.parameters)











# Generate posterior predictive samples
function generate_posterior_predictive(chain, Q_std, PF_std, LF_std, ID, N, T)
    mu_alpha_samples = chain[:mu_alpha][:]
    beta_Q_samples = chain[:beta_Q][:]
    beta_PF_samples = chain[:beta_PF][:]
    beta_LF_samples = chain[:beta_LF][:]
    sigma_samples = chain[:sigma][:]
    alpha_individual_samples = [chain[Symbol("alpha_individual[$i]")][:] for i in 1:N]

    predictive_samples = zeros(length(mu_alpha_samples), length(Q_std))

    for i in 1:length(mu_alpha_samples)
        mu_alpha = mu_alpha_samples[i]
        beta_Q = beta_Q_samples[i]
        beta_PF = beta_PF_samples[i]
        beta_LF = beta_LF_samples[i]
        sigma = sigma_samples[i]

        for j in 1:N
            idx = findall(ID .== j)
            X = hcat(Q_std[idx], PF_std[idx], LF_std[idx])
            alpha_j = alpha_individual_samples[j][i]
            mu = alpha_j .+ X * [beta_Q, beta_PF, beta_LF]
            predictive_samples[i, idx] .= rand.(Normal.(mu, sigma))
        end
    end

    return predictive_samples
end

# Compute residuals
function compute_residuals(Y_log, predictive_samples)
    mean_predictive = mean(predictive_samples, dims=1)[:]
    residuals = Y_log - mean_predictive
    return residuals
end

# Plot posterior predictive checks
function plot_posterior_predictive(Y_log, predictive_samples, T, ID)
    # Compute mean and quantiles of posterior predictive samples
    mean_predictive = mean(predictive_samples, dims=1)[:]
    lower_bound = mapslices(x -> quantile(x, 0.025), predictive_samples; dims=1)[:]
    upper_bound = mapslices(x -> quantile(x, 0.975), predictive_samples; dims=1)[:]

    # Group by time
    time_mean = [mean(Y_log[findall(ID .== t)]) for t in 1:T]
    time_predictive = [mean(mean_predictive[findall(ID .== t)]) for t in 1:T]
    lower_time_bound = [mean(lower_bound[findall(ID .== t)]) for t in 1:T]
    upper_time_bound = [mean(upper_bound[findall(ID .== t)]) for t in 1:T]

    # Plot observed vs. predictive
    plot(1:T, time_mean, label="Observed Mean (Y_log)", lw=2, xlabel="Time Period", ylabel="Log(Cost)")
    plot!(1:T, time_predictive, ribbon=(lower_time_bound, upper_time_bound), label="Posterior Predictive Mean & 95% CI", lw=2)
end

# Run posterior predictive check
posterior_predictive_samples = generate_posterior_predictive(chain_with_hierarchical_and_diag, Q_std, PF_std, LF_std, ID, N, T)
plot_posterior_predictive(Y_log, posterior_predictive_samples, T, ID)




function compute_residuals_hier(chain, Y_log, Q_std, PF_std, LF_std, ID, N, T)
    # Extract posterior means for parameters
    mu_alpha = mean(parent(chain[:mu_alpha])[:])
    tau_alpha = mean(parent(chain[:tau_alpha])[:])
    beta_Q = mean(parent(chain[:beta_Q])[:])
    beta_PF = mean(parent(chain[:beta_PF])[:])
    beta_LF = mean(parent(chain[:beta_LF])[:])
    
    # Calculate individual intercepts
    alpha_individual = [mu_alpha + tau_alpha * randn() for i in 1:N]
    
    # Compute residuals
    residuals = zeros(length(Y_log))
    for i in 1:N
        idx = findall(ID .== i)
        X = hcat(Q_std[idx], PF_std[idx], LF_std[idx])
        mu = alpha_individual[i] .+ X * [beta_Q, beta_PF, beta_LF]
        residuals[idx] = Y_log[idx] .- mu
    end
    
    return residuals
end


