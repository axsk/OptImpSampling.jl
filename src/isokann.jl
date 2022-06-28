using Flux
using Flux.Optimise: update!
using Statistics
using Plots
using StatsBase
import Zygote

function mlp(x=[1,3,3], sig=false)
    last = sig ? Dense(x[end], 1, σ) : Dense(x[end], 1)
    Chain([Dense(x[i], x[i+1], σ) for i = 1:length(x)-1]..., last, first)
end

# shiftscale which also scales stds
function shiftscale!(ys, stds)
    a, b = extrema(ys)
    ys .= (ys .- a) ./ (b - a)
    stds ./= (b-a)
end

""" shift scaled koopman sampling """
function SK(ocp, xs::AbstractVector, nmc=10)
    kxs = zeros(length(xs))
    stds = zeros(length(xs))
    for i in 1:length(xs)
        kxs[i], stds[i] = mean_and_std(evaluate(ocp, xs[i]) for j in 1:nmc)
    end
    shiftscale!(kxs, stds)
    kxs, stds ./ sqrt(length(xs))  # return the scaled uncertainty
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
    for i in 1:poweriter
        chi = statify(model)
        ocp = ProblemOptChi5(chi=chi, q=-1.0, b=0.0, forcing=forcing, dt=dt)
        #xs = [rand(1) * 4 .- 2 for i in 1:nx]
        xs = map(x->[x], range(-2, 2, nx))
        target, stds = SK(ocp, xs, nmc)
        for j in 1:learniter
            loss = learnstep!(model, xs, target, opt)
            push!(ls, loss)
        end
        cbplot(model, ls, xs, target, stds, iso)
    end
    model, ls
end

function string(iso::Isokann)
    (;nx, nmc, poweriter, learniter, opt, model, forcing, dt) = iso
    "nx=$nx nmc=$nmc piter=$poweriter liter=$learniter f=$forcing dt=$dt"
end

function cbplot(model, loss, xs, target, stds, iso)
    length(loss) % 1 == 0 || return
    p1=plot(loss, yaxis=:log)
    p2=plot(x->model([x]), -3:.1:3, ylims=(0,1))
    @show mean(stds)
    scatter!(p2, reduce(vcat, xs), target, yerror=stds)
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
