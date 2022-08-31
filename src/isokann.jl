#export Isokann, run, mlp

using Flux

function mlp(x=[1,3,3], sig=true)
    last = sig ? Dense(x[end], 1, σ) : Dense(x[end], 1)
    Chain([Dense(x[i], x[i+1], σ) for i = 1:length(x)-1]..., last, first)#, x->(x+1)/2)
end

# shiftscale which also scales stds and returns shift and rate
function shiftscale!(ys, stds, eps = 0)
    a, b = extrema(ys)
    ys .= (ys .- a) ./ (b - a) .+ eps
    stds ./= (b-a)
    λ = min(b-a, 3)  # inferred eigenvalue
    s = a / (a + 1 - b) + eps  # inferred shift

    return λ, s
end

""" shift scaled koopman sampling """
function SK(ocp, xs::AbstractVector, nmc=10)
    kxs = zeros(length(xs))
    stds = zeros(length(xs))
    # TODO @threads zip(xs, 1:nmc) -> matrix
    Threads.@threads for i in 1:length(xs)
        kxs[i], stds[i] = mean_and_std(evaluate(deepcopy(ocp), xs[i], nmc))
        yield()
    end
    λ, b = shiftscale!(kxs, stds)
    stds ./= sqrt(nmc)  # monte carlo variance
    kxs, stds, λ, b
end

abstract type AIsokann end
@with_kw mutable struct Isokann3 <: AIsokann
    nx = 10
    nmc = 10
    poweriter = 100
    learniter = 1
    opt = Nesterov(.1, .9)
    model = mlp()
    forcing = 0.
    dt = .01
    ls = Float64[]
    stds = Float64[]
end

@with_kw mutable struct Isokann4 <: AIsokann
    nx = 10
    nmc = 10
    poweriter = 100
    learniter = 10
    opt = Flux.ADAM(0.01)
    model = mlp()
    forcing = 0.
    dt = .01
    ls = Float64[]
    stds = Float64[]
    potential = doublewell
end

@with_kw mutable struct UniformSampler
    min::Float64 = -2.
    max::Float64 = 2.
    n::Int = 10
end

@with_kw mutable struct GridSampler
    min::Float64 = -2.
    max::Float64 = 2.
    n::Int = 10
end

sample(s::UniformSampler, dim=1) = [rand(dim) * (s.max-s.min) .+ s.min for i in 1:s.n]
sample(s::GridSampler, dim=1) = [rand(dim) * (s.max-s.min) .+ s.min for i in 1:s.n]


Isokann = Isokann4
converging() = Isokann(poweriter=1000, learniter=100, nmc=100, forcing=1, opt=ADAM(0.001), model=mlp([1,3,3], false), dt=.01)
happy1() = Isokann(nx=30, poweriter=100, learniter=100, nmc=3, forcing=1., opt= ADAM(0.01), dt=0.01)
basic2d() = Isokann(model=mlp([2,5,5]), potential=triplewell)
better2d() = Isokann(model=mlp([2,5,5,5]), potential=triplewell, forcing=1, dt=0.001)

function run(iso::AIsokann; liveplot=false)
    (;nx, nmc, poweriter, learniter, opt, model, forcing, dt, ls, stds) = iso
    q = -1.
    b = 0.
    λ = 0
    learnrate = 1
    local plt
    for i in 1:poweriter
        xs = sample(UniformSampler(-2,2,nx), dim(iso.potential))
        chi = statify(model)
        ocp = ProblemOptChi(chi=chi, q=q, b=b, forcing=forcing, dt=dt, potential=iso.potential)

        #xxs = humboldtsample(xs, ocp, 2)
        target, std, λ1, b1 = SK(ocp, xs, nmc)  # new trajectories

            λ = λ * (1-learnrate) + λ1 * learnrate
            b = b * (1-learnrate) + b1 * learnrate
            q = min(log(λ), 0)  # dont allow positive rates

        # SGD Loop
        for j in 1:learniter
            loss = learnstep!(model, xs, target, opt)
            push!(stds, mean(std))
            push!(ls, loss)
        end
        plt = cbplot(model, ls, xs, target, stds, std, iso)
        liveplot && display(plt)
    end
    display(plt)
    iso, (ls, stds)
end

""" single supervised learning iteration """
function learnstep!(model, xs, target, opt)
    ps = Flux.params(model)
    xs
    loss, back = Zygote.pullback(ps) do
        predict = model.(xs)
        mean(abs2, target - predict)
    end
    grad = back(one(loss))
    Flux.update!(opt, ps, grad)
    loss
end

function string(iso::AIsokann)
    (;nx, nmc, poweriter, learniter, opt, model, forcing, dt) = iso
    "nx=$nx nmc=$nmc piter=$poweriter liter=$learniter f=$forcing dt=$dt"
end

function cbplot(model, loss, xs, target, stds, std, iso)
    length(loss) % 1 == 0 || return

    p1=plot(sqrt.(loss), yaxis=:log, title="loss", label="loss", legend=:bottomleft)
    plot!(p1, stds, label="std")
    #dim(iso.potential) > 1 && return p1
    if length(xs) > 0
        if length(xs[1]) > 1  # hacky way to plot first dim

            p2 = contour(-2:.1:2, -2:.1:2, (x,y)->model([x,y]), fill=true)
            @show l = map(model, xs) - target
            xs = reduce(hcat, xs)'
            scatter!(p2, xs[:,1], xs[:,2], markersize=l.^2 * 100)
        else
            p2=plot(x->model([x]), -5:.1:5, ylims=(-.1,1.1), title="fit", label="χ", legend=:best)
            scatter!(p2, reduce(vcat, xs), target, yerror=std, label="SKχ")
        end

    end
    plot(p1, p2, title=string(iso))
end

function plot_description(iso::AIsokann)
    p=plot()
    annotate!(string(iso))
    p
end


# we use this to create a copy which uses StaticArrays, for faster d/dx gradients
statify(x::Any) = x
statify(c::Chain) = Flux.Chain(map(statify, c.layers)...)
function statify(d::Dense)
    w = d.weight
    W = SMatrix{size(w)...}(w)
    b = d.bias
    B = SVector{length(b)}(b)
    Dense(W, B, d.σ)
end


" subsample_uniform(ys, n)

Return up to `n` indices into `ys` such that these elements each
lie in one of `n` uniform partitions of the unit interval.
Does return less indices if not all partitions were hit "
function subsample_uniform(ys, n)
    p = sortperm(ys)
    s = ys[p]
    picks = []
    first = 1
    for i in 1:n
        last = findlast(x->x<=i/n, s)
        (isnothing(last) || (last < first)) && continue  # no element found in box
        push!(picks, rand(first:last))
        first = last + 1
    end
    p[picks]
end


function humboldtsample(xs, ocp, branch)
    ocp = deepcopy(ocp)
    ocp.forcing = 0.
    nxs = copy(xs)
    for x in xs
        for i in 1:branch
            s = msolve(ocp, x)[1:end-1]
            push!(nxs, s)
        end
    end

    ys = map(ocp.chi, nxs)
    is = subsample_uniform(ys, length(xs))

    return xs[is]
end
