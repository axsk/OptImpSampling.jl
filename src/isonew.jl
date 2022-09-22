# cleaner and simpler reimplementation of ISOKANN (1)
import Flux
import StatsBase

function densenet(layers=[1,3,3,1])
    Flux.Chain([Flux.Dense(layers[i], layers[i+1], Flux.sigmoid) for i in 1:length(layers)-1]...)
end

densenet(dynamics::AbstractLangevin, layers=[5,5]) = densenet([dim(dynamics); layers; 1])

# @time isokann(Doublewell) == 1sec

function isokann(dynamics; model=densenet(dynamics),
                 nx=10, nkoop=10, poweriter=100, learniter=10, dt=0.01, alg=SROCK2(),
                 opt=Flux.Adam(0.01),
                 sec=Inf, cb=Flux.throttle(plot_callback,sec,leading=false, trailing=true))

    xs = randx0(dynamics, nx)
    sde = SDEProblem(dynamics, dt = dt, alg=alg)

    ps = Flux.params(model)
    stds = Float64[]
    ls = Float64[]
    local S, cde
    control = nocontrol

    for _ in 1:poweriter

        cde = ControlledSDE(sde, control)

        # evaluate koopman
        ys, ws = girsanovbatch(cde, xs, nkoop)
        cs = model(ys)
        ks, std = vec.(StatsBase.mean_and_std(cs[1,:,:].*ws, 2))
        #ks = mean(cs[1,:,:] .* ws, dims=2) |> vec

        # estimate shift scale
        S = Shiftscale(ks)
        target = invert(S, ks)
        std = std ./ exp(S.q) / sqrt(nkoop)

        # train network
        for _ in 1:learniter
            loss() = mean(abs2, (model(xs)|>vec) .- target)
            l, grad = Flux.withgradient(loss, ps)
            Flux.update!(opt, ps, grad)
            push!(ls, l)
            push!(stds, mean(std))
        end

        cb(;losses=ls, model, xs, target, stds, std, cs)

        # update controls
        control = optcontrol(statify(model), S, sde)

        # resample xs uniformly along chi
        xys = hcat(xs, reshape(ys, size(xs, 1), :))
        cs = model(xys) |> vec
        xs = humboldtsample(xys, cs, nx)
    end
    return (;model, ls, S, sde, cde, xs, dynamics)
end

function plot_callback(; kwargs...)
    (;losses, model, xs, target, std, stds) = NamedTuple(kwargs)

    p1=plot(yaxis=:log, title="loss", legend=:bottomleft)
    plot!(p1, sqrt.(losses), label="loss")
    plot!(p1, vec(stds), label="std")

    if length(xs) > 0
        if size(xs, 1) == 1
            p2=plot(ylims=(-.1,1.1), title="fit",  legend=:best)
            plot!(p2, x->model([x])[1], -3:.1:3, label="χ")
            scatter!(p2, vec(xs), vec(target), yerror=vec(std), label="SKχ")
        else
            p2 = contour(-2:.1:2, -2:.1:2, (x,y)->model( [x,y])[1], fill=true, alpha=.1)
            l = vec(mapslices(model, xs, dims=1)) - target
            #xs = reduce(hcat, xs)'
            scatter!(p2, xs[:,1], xs[:,2], markersize=l.^2 * 100)
        end
    end

    plot(p1, p2) |> display
end
