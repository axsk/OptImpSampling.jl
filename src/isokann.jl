import Base.run

### legacy implementation



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
    ys = zeros(length(xs), length(xs[1]), nmc)
    chis = zeros(length(xs), nmc)
    # TODO @threads zip(xs, 1:nmc) -> matrix
    #Threads.@threads
    @floop for i in eachindex(xs)
        yy, cc, ws = prop_and_evaluate(deepcopy(ocp), xs[i], nmc)
        ys[i,:,:] = yy
        chis[i, :]  = cc
        kxs[i], stds[i] = mean_and_std(chis[i,:] .* ws)
        yield()
    end
    λ, b = shiftscale!(kxs, stds)
    stds ./= sqrt(nmc)  # monte carlo variance
    kxs, stds, λ, b, ys
end

abstract type AIsokann end

@with_kw mutable struct Isokann <: AIsokann
    nx = 10
    nmc = 10
    poweriter = 100
    learniter = 10
    opt = Flux.Adam(0.01)
    model = fluxnet1()
    forcing = 1.
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

sample(s::UniformSampler, dim=1) = [rand(dim) * (s.max-s.min) .+ s.min for i in 1:s.n]

converging() = Isokann(poweriter=1000, learniter=100, nmc=100, forcing=1, opt=Adam(0.001), model=fluxnet1([1,3,3,1], false), dt=.01)
happy1() = Isokann(nx=30, poweriter=100, learniter=100, nmc=3, forcing=1., opt= Adam(0.01), dt=0.01)
basic2d() = Isokann(model=fluxnet1([2,5,5,1]), potential=triplewell)
better2d() = Isokann(model=fluxnet1([2,5,5,5,1]), potential=triplewell, forcing=1, dt=0.001)

function run(iso::AIsokann; liveplot=0, humboldt=true, hotfixbnd=false)
    (;nx, nmc, poweriter, learniter, opt, model, forcing, dt, ls, stds) = iso
    q = -1.
    b = 0.
    λ = 0
    learnrate = 1
    local plt
    xs = sample(UniformSampler(-2,2,nx), dim(iso.potential))
    local ocp
    for i in 1:poweriter
        if hotfixbnd
            xs = [xs; [[-2.], [2.]]]  # hotfix for infering λ, b with small samplings
        end
        chi = statify(model)
        ocp = OptChiControl(chi=chi, q=q, b=b, forcing=forcing, dt=dt, potential=iso.potential)

        if humboldt
            xs = humboldtsample(xs, ocp, nx, 10)
        else
            xs = sample(UniformSampler(-2,2,nx), dim(iso.potential))
        end
        # TODO: SK now also exports the simulated endpoints, re-use them for humboldt
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
        if liveplot > 0 && i % liveplot == 0
            plt = cbplot(model, ls, xs, target, stds, std, iso)
          display(plt)
        end
    end
    # TODO: find means to plot on demand
    #display(plt)
    iso, (ls, stds), ocp
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

            p2 = contour(-2:.1:2, -2:.1:2, (x,y)->model([x,y]), fill=true, alpha=.1)
            l = map(model, xs) - target
            xs = reduce(hcat, xs)'
            scatter!(p2, xs[:,1], xs[:,2], markersize=l.^2 * 100)
        else
            p2=plot(x->model([x]), -3:.1:3, ylims=(-.1,1.1), title="fit", label="χ", legend=:best)
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
