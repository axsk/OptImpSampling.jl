# simpler, bigger, better, stronger
# ISOKANN2 - multidimensional ISOKANN in abstract form

import Flux

using Zygote: ignore_derivatives, withgradient
using LinearAlgebra: norm

include("langevin.jl")

function loss(xs, ys, chi, K)
    d, n, m = size(ys)
    @assert size(xs) == (d, n)

    K = K ./ sum(K, dims=2)  # enforce rowsum = 1

    # using the power iteration we dont diff through koopman
    # however: why not?
    chix = chi(xs)
    chiy = ignore_derivatives(chi)(ys)

    # mc estimate koopman over time propagation
    koopchi = mean(chiy, dims=3)[:,:,1]

    # TODO: which one is appropriate?
    # norm(chi(xs) - K^-1 * koopchi)
    norm(K * chix - koopchi) / n
end

DIM = 3

# softmax enforces χ-properties
chinet(dims, nchi) = Flux.Chain(Flux.Dense(dims,32), Flux.Dense(32,nchi), softmax)

sample(dynamics, n) = randn(DIM, n)
propagate(dynamics::Nothing, xs, n) = repeat(xs, outer=[1,1,n])

function propagate(dynamics::Langevin, xs, n)
    sde = SDEProblem(dynamics)
    X = repeat(xs, outer=[1,1,n])
    ys = mapslices(X, dims=1) do x0
        solve(sde, u0 = x0)[end]
    end
    return ys
end

function test_train()
    train(chinet(DIM, 2), rand(2,2), nothing, 100, 10, 10)
    train(chinet(DIM, 2), rand(2,2), Langevin(), 100, 10, 10)
end

# should we call n_startpoints = batch, n_iter = epochs?
function train(chi, K, dynamics, n_iter, n_startpoints, n_koopman)
    ls = []
    for i in 1:n_iter
        xs = randn(1, n_startpoints)
        ys = propagate(dynamics, xs, n_koopman)

        l, dl = withgradient(Flux.params(chi, K)) do
            loss(xs, ys, chi, K)
        end
        push!(ls, l)
        println(l)

        # opt, params = Optimisers.update(opt, params, dl[1])
        # optk, K = Optimisers.upate(optk, K, dl[2])

        Flux.update!(ADAM(), Flux.params(chi, K), dl)
    end

    return chi, K, ls
end
