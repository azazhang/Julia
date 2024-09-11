using Distributions, StatsBase, Random, Plots, StatsPlots #, PrettyTables
Random.seed!(1249)
#cd("C:\\Users\\millsjf\\OneDrive - University of Cincinnati\\Class\\9011 2022")
cd("C:\\Users\\tszan\\OneDrive - University of Cincinnati\\Class\\8013 Bayesian econometrics\\8013 2024")


# The data:
x_trt = [-0.51, 1.86, 0.91, 1.42, 0.87, 1.23, 0.02, 0.28, 0.14, 2.24, 2.63, 
-0.02, 0.83, -0.4, 2.33, -0.56, 1.27, 0.16, 0.6, 2.41]
x_pbo = [-0.8748, 1.5307, 0.1469, -2.0674, 1.3385, -0.4815, 1.2451, -1.4923, 1.0757, 2.099, -0.1703, -0.924, -2.181, 0.01667, 4.2496, -0.8309, -0.9772, -2.5074, -1.6458]



y_trt = [0.37, 1.52, 2.46, 2.89, 2.17, 1.86, 1.96, 0.06, 2.07, 7.06, 12.74, 
4.04, 3.09, 4.13, 5.16, 0.24, 4.9, 2.07, 0.8, 5.39]

y_pbo = [-1.004, 1.43, 0.273, 1.309, -0.584, 1.346, -1.451, 0.889, 2.153, 
-0.088, -0.744, -2.004, 0.137, 4.398, -0.723, -0.81, -2.458, -1.709]



n = length(x_trt)

mean(x_trt)
std(x_trt)

sdtrt_mn = std(x_trt)/sqrt(n)

#x_pbo = 0.48 .+ 1.3.*randn(n-1)
np = length(x_pbo)
sdpbo_mn = std(x_pbo)/sqrt(np)

mutrt_draws = rand(TDist(n-1), 10^6).*sdtrt_mn .+ mean(x_trt)
mupbo_draws = rand(TDist(np-1), 10^6).*sdpbo_mn .+ mean(x_pbo)
plot(mutrt_draws, st = :density, label = "Treatment x", color = "red", fill=true, alpha=0.4)
plot!(mupbo_draws, st = :density, label = "Control x", color = "green", fill=true, alpha=0.4)

diff_trt_pbo = mutrt_draws .- mupbo_draws
plot!(diff_trt_pbo, st = :density, label = "Efficacy x: treat - control", color="blue", fill=true, alpha=0.4)
vline!([0.0], linecolor = "black", linewidth = 2, label = false)

mean(diff_trt_pbo)
p95 = quantile(diff_trt_pbo, [0.025, 0.975])
prob_le_zero = length(diff_trt_pbo[diff_trt_pbo .<= 0.0])/length(diff_trt_pbo)
vline!(p95, linecolor = "blue", label = "prob<0 = $prob_le_zero")

# y treat - control


ny = length(y_trt)
sdtrty_mn = std(y_trt)/sqrt(ny)
mutrty_draws = rand(TDist(ny-1), 10^6).*sdtrty_mn .+ mean(y_trt)
#plot(mutrty_draws, st = :density, label = "Treatment y", color="blue", fill=true, alpha=0.4)


#y_pbo = rand(Uniform(-0.2,0.2),18) + [-0.8748, 1.5307, 0.1469, 1.3385, -0.4815386413790482, 1.2450813219379055, -1.4923656651449668, 1.0757, 2.0994, -0.17027253010137622, -0.9239921594998992, -2.181218825311208, 0.016666739241215478, 4.249622767557066, -0.8309544957559665, -0.9772, -2.5074, -1.6458]
#show(round.(y_pbo, digits = 3))

nyp = length(y_pbo)
sdpboy_mn = std(y_pbo)/sqrt(nyp)
mupboy_draws = rand(TDist(nyp-1), 10^6).*sdpboy_mn .+ mean(y_pbo)
#plot!(mupboy_draws, st = :density, label = "Control y", color="darkgreen", fill=true, alpha=0.4)


plot!(mutrty_draws, st = :density, label = "Treatment y", color = "darkred", fill=true, alpha=0.3)
plot!(mupboy_draws, st = :density, label = "Control y", color = "darkgreen", fill=true, alpha=0.3)

diff_trt_pboy = mutrty_draws .- mupboy_draws
plot!(diff_trt_pboy, st = :density, label = "Efficacy y: treat - control", color="darkblue", fill=true, alpha=0.4)
#vline!([0.0], linecolor = "black", linewidth = 2, label = false)

mean(diff_trt_pboy)
p95 = quantile(diff_trt_pboy, [0.025, 0.975])
prob_le_zeroy = length(diff_trt_pboy[diff_trt_pboy .<= 0.0])/length(diff_trt_pbo)
vline!(p95, linecolor = "blue", label = "prob<0 = $prob_le_zeroy")

# Difference in Ave. Efficacy (ATE beyond placebo effect)
plot(diff_trt_pbo, st = :density, label = "Efficacy x: treat - control", color="blue", fill=true, alpha=0.3)
vline!([0.0], linecolor = "black", linewidth = 2, label = false)
plot!(diff_trt_pboy, st = :density, label = "Efficacy y: treat - control", color="purple", fill=true, alpha=0.3)
diff_effic_y_x = diff_trt_pboy .- diff_trt_pbo

plot!(diff_effic_y_x, st = :density, label = "Efficacy y - x", color="green", fill=true, alpha=0.5)
mean(diff_effic_y_x)
p95 = quantile(diff_effic_y_x, [0.025, 0.975])
prob_le_zeroyx = length(diff_effic_y_x[diff_effic_y_x .<= 0.0])/length(diff_effic_y_x)
vline!(p95, linecolor = "green", label = "prob<0 = $prob_le_zeroyx")

# The above figure gives the probability density for a difference of a difference 
# with different numbers of observations
# with difference variances
# no need for any asymptotic assumptions beyond normality of a sample mean
# with uninformative priors (so using no additional info other than the samples)

# We can compare the variances in the same way (use the inverted-Gamma, or better
# still, the Gamma for the precision).  Replace the Normal draws with draws from
# the gamma(s/2, vs^2/2). E.g., to compare the precision or SD of treatments x and y
vx = length(x_trt) - 1
vs2x = vx*var(x_trt)
check = sum((x_trt .- mean(x_trt)).^2)

vy = ny - 1
vs2y = vy*var(y_trt)
#### WARNING: Julia flips the second parameter - "rate" instead of "scale"
prec_trtx_draws = rand(Gamma(vx/2, 2/vs2x), 10^6)  
prec_trty_draws = rand(Gamma(vy/2, 2/vs2y), 10^6)
plot(prec_trty_draws, st = :density, label = "Treatment y", color="blue", fill=true, alpha=0.5, title = "Precision")
plot!(prec_trtx_draws, st = :density, label = "Treatment x", color="green", fill=true, alpha=0.5)

# SD instead of precision:
sd_trtx_draws = 1 ./sqrt.(prec_trtx_draws)
sd_trty_draws = 1 ./sqrt.(prec_trty_draws)
plot(sd_trty_draws, st = :density, label = "Treatment y", color="blue", fill=true, alpha=0.5, title = "Standard Deviation")
plot!(sd_trtx_draws, st = :density, label = "Treatment x", color="green", fill=true, alpha=0.5)
vline!([std(x_trt) std(y_trt)], label = false)

std(x_trt)
std(y_trt)
var(x_trt)
var(y_trt)

# var instead of precision:
var_trtx_draws = 1 ./prec_trtx_draws
var_trty_draws = 1 ./prec_trty_draws
plot(var_trty_draws, st = :density, label = "Treatment y", color="blue", fill=true, alpha=0.5, title = "Variance", xlims = (0, 25))
plot!(var_trtx_draws, st = :density, label = "Treatment x", color="green", fill=true, alpha=0.5)



# We can compare proportions (chance of success or failure) using a Beta posterior density.
# so if we have data that are successes in n trials, we just replace the 4 sets of
# random draws from the Normal with draws from the Beta(s+1, n-s+1), which is the posterior
# resulting from a uniform prior on the proportion and the binomial likelihood, e.g. suppose we get
# 11 successful treatments out of 18 patients in the treatment group, then
sy = 11
ny = 18
prop_trty_draws = rand(Beta(sy+1, ny-sy+1), 10^6)
plot(prop_trty_draws, st = :density, label = "Treatment y", color="blue", fill=true, alpha=0.5, title = "Proportion of Successes")


# We can do the same thing for regression parameters (and parameters of any model), as well as
# predicted values, etc.