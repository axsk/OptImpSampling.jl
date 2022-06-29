using Flux
using Flux.Optimise: update!
using Statistics
using Plots
using StatsBase
import Zygote

function mlp(x=[1,3,3], sig=true)
    last = sig ? Dense(x[end], 1, σ) : Dense(x[end], 1)
    Chain([Dense(x[i], x[i+1], σ) for i = 1:length(x)-1]..., last, first)
end

# shiftscale which also scales stds and returns shift and rate
function shiftscale!(ys, stds)
    a, b = extrema(ys)
    ys .= (ys .- a) ./ (b - a)
    stds ./= (b-a)
    λ = b-a  # inferred eigenvalue
    s = a / (a + 1 - b)  # inferred shift
    return λ, s
end

""" shift scaled koopman sampling """
function SK(ocp, xs::AbstractVector, nmc=10)
    kxs = zeros(length(xs))
    stds = zeros(length(xs))
    # TODO @threads zip(xs, 1:nmc) -> matrix
    for i in 1:length(xs)
        kxs[i], stds[i] = mean_and_std(evaluate(ocp, xs[i]) for j in 1:nmc)
    end
    λ, b = shiftscale!(kxs, stds)
    stds ./= sqrt(nmc)
    kxs, stds, λ, b
end

@with_kw mutable struct Isokann1
    nx = 10
    nmc = 10
    poweriter = 100
    learniter = 1
    opt = Nesterov(.1, .9)
    model = mlp()
    forcing = 0.
    dt = .01
end

Isokann = Isokann1

function run(iso::Isokann)
    (;nx, nmc, poweriter, learniter, opt, model, forcing, dt) = iso
    ls = Float64[]
    stds = Float64[]
    q = -1.
    b = 0.
    for i in 1:poweriter
        chi = statify(model)
        ocp = ProblemOptChi5(chi=chi, q=q, b=b, forcing=forcing, dt=dt)
        #xs = [rand(1) * 4 .- 2 for i in 1:nx]
        xs = map(x->[x], range(-2, 2, nx))
        target, std, λ, b = SK(ocp, xs, nmc)
        q = min(log(λ), 0)  # dont allow positive rates

        for j in 1:learniter
            loss = learnstep!(model, xs, target, opt)
            push!(stds, mean(std))
            push!(ls, loss)
        end
        cbplot(model, ls, xs, target, stds, std, iso)
    end
    model, ls
end

""" single supervised learning iteration """
function learnstep!(model, xs, target, opt)
    ps = Flux.params(model)
    loss, back = Zygote.pullback(ps) do
        predict = model.(xs)
        mean(abs2, target - predict)
    end
    grad = back(one(loss))
    update!(opt, ps, grad)
    loss
end

function string(iso::Isokann)
    (;nx, nmc, poweriter, learniter, opt, model, forcing, dt) = iso
    "nx=$nx nmc=$nmc piter=$poweriter liter=$learniter f=$forcing dt=$dt"
end

function cbplot(model, loss, xs, target, stds, std, iso)
    length(loss) % 1 == 0 || return
    p1=plot(loss, yaxis=:log, title="loss", label="loss", legend=:bottomleft)
    plot!(p1, stds, label="std")
    p2=plot(x->model([x]), -3:.1:3, ylims=(-.1,1.1), title="fit", label="χ", legend=:best)
    scatter!(p2, reduce(vcat, xs), target, yerror=std, label="SKχ")
    plot(p1, p2, title=string(iso)) |> display
end


# we use this to create a copy which uses StaticArrays, for faster d/dx gradients
statify(x::Any) = x
statify(c::Chain) = Chain(map(statify, c.layers)...)
function statify(d::Dense)
    w = d.weight
    W = SMatrix{size(w)...}(w)
    b = d.bias
    B = SVector{length(b)}(b)
    Dense(W, B, d.σ)
end
