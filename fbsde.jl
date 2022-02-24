# Proof of concept
# problems with NaN

# Following
# 2019 - Kebiri Hartmann
# Adaptive importance sampling with forward-backward stochastic differential equations

# Whats new here is
# a) we use a NN for learning the value function
# b) instaed of learning V_t iterativelu backwards in time
#    we learn it for all times at once, improving our trajectories while learning.

using Flux
using Plots

# Multi layer Perceptron with sigmoid activation and non-sigmoid output
function mlp(x=[2,2,1])
    Chain([Dense(x[i], x[i+1], σ) for i=1:length(x)-2]..., Dense(x[end-1], x[end]))
end

doublewell(x) = ((x[1])^2 - 1) ^ 2

defproblem() = (
    b = x->-gradient(doublewell,x)[1],
    sigma = 0.6,
    n = 2000,
    h = 1/20,
    f = x -> 0,
    g = x -> sum(x), #sum(abs2, x .- 1) < 1/2,
    )

function eulermaruyama(x0, b, sigma, n, h)
    x = x0
    X = similar([x0], 0)
    noise = randn(n)
    for i in 1:n
        x = x + h * b(x) .+ sqrt(h) * sigma * noise[i]
        push!(X, x)
    end
    X, noise
end

using LinearAlgebra
norm(x) = sum(abs2, x)

function backwardpass(X, h, f, g, Z, noise)
    n = length(X)
    Y = zeros(n)
    Y[end] = g(X[n])
    for t in n:-1:2
        Y[t-1] = Y[t] + h * (f(X[t]) - 1/2 * norm(Z[t])^2 ) - sqrt(h) * dot(Z[t], noise[t])
        # train V_t
        # compute Z
    end
    return Y
end

function forwardbackward(x0; b, sigma, n, h, V, f, g)
    X, noise = eulermaruyama(x0, b, sigma, n, h)

    Z = similar(X)
    for i in 1:length(X)
        t = i / length(X) # reparametrize time to [0,1] for easier learning
        dV = gradient(x->V([t; x])[1], X[i])[1]
        if any(isnan.(dV))
            @show X, dV
            error()
        end
        Z[i] = - sigma * dV
    end



    Y = backwardpass(X, h, f, g, Z, noise)

    plt = true
    if plt
        plot(Flux.stack(X, 1))
        plot!(Y)
        plot!(Flux.stack(Z, 1)) |> display
    end
    return X, Y, Z
end

"""
    turn n-vector of m-samples into (m x n) matrix with [0,1] timestamps in the first row
"""
function stackinput(X::Vector{<:Array{T}}) where {T}
    n = length(X)
    input = Flux.stack(X, 2)
    ts = collect(1:n) / n
    input = vcat(ts', input) :: Array{T}
end


"""
    do a single neural network update with `n` sample paths
"""
function step!(model, problem, opt, n = 100)

    Xs = []
    Ys = []
    for i in 1:n
        x0 = randn(1)
        X, Y, Z = forwardbackward(x0, V = model; problem...)

        X = stackinput(X)
        push!(Xs, X)
        push!(Ys, Y)
    end

    T = length(Ys[1])
    weights = (0:1/T:1)[2:end]

    ps = Flux.params(model)
    grads = Flux.gradient(ps) do
        loss = 1/(n*T) * sum(zip(Xs, Ys)) do (X, Y)
            y = model(X)[1,:]
            sum(abs2, (y-Y) .* weights)
        end
        sqnorm(x) = sum(abs2, x)
        reg = sum(sqnorm, ps) * 0.01
        @show loss
        @show reg
        loss + reg
    end

    Flux.Optimise.update!(opt, ps, grads)
end

function test()
    model = mlp([2, 3, 3, 1])
    opt = Flux.Optimise.Descent(0.00001)
    for i in 1 : 1000
        step!(model, defproblem(), opt)
        params(model)
        if i % 1 == 0
            plot(x->model([0, x])[1])
            plot!(x->model([1, x])[1]) |> display
        end
    end
    model
end


function estimator(x0; b, V, sigma, n, h, f, g, _...)
    # TODO: we work with fixed time here
    tmid = n * h / 2
    u(x) = -sigma * gradient(x->V([tmid; x])[1], x)[1]

    bu(x) = b(x) + u(x)
    x, noise = eulermaruyama([-1.], bu, sigma, n, h)

    gamma = sum(f.(x) + 1/2 * (norm.(u.(x)).^2)) * h + g(x[end])

end
