using Distributions, Plots, Random
Random.seed!(9011)

function generate_ar1(phi, sigma, n)
    y = zeros(n)
    y[1] = rand(Normal(0, sigma))
    for i in 2:n
        y[i] = phi*y[i-1] + sigma*rand(Normal(0, 1))
    end
    return y
end

n = 100
phi_stationary = 0.8
phi_nonstationary = 1.1
sigma = 1.0
y_stationary = generate_ar1(phi_stationary, sigma, n)
y_nonstationary = generate_ar1(phi_nonstationary, sigma, n)

p1 = plot(y_stationary, label="Stationary AR(1)")
p2 = plot(y_nonstationary, label="Nonstationary AR(1)")
plot(p1, p2, layout=(2,1), size=(800, 400))
