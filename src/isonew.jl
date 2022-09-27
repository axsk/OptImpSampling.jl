# cleaner and simpler reimplementation of ISOKANN (1)
import Flux
import StatsBase
import Optimisers


#@assert @elapsed isokann(Doublewell()) < 2

fluxnet(dynamics::AbstractLangevin, layers=[5,5]) = fluxnet([dim(dynamics); layers; 1])

# 10-3 in about 30s
#isokann(Doublewell(), sec=3, poweriter=100000, learniter=100, opt=Flux.Adam(0.001), dt=0.001, nx=10, nkoop=10, keepedges=true);

function isokann(;dynamics=Doublewell(), model=fluxnet(),
                 nx::Int=10, nkoop::Int=10, poweriter::Int=100, learniter::Int=10, dt::Float64=0.01, alg=SROCK2(),
                 opt=Optimisers.Adam(0.01), keepedges::Bool=true,
                 sec=Inf, cb=Flux.throttle(plot_callback,sec,leading=false, trailing=true)
                 )

    xs = randx0(dynamics, nx)
    sde = SDEProblem(dynamics, dt = dt, alg=alg)

    opt = Optimisers.setup(opt, model)
    stds = Float64[]
    ls = Float64[]
    local S, cde
    control = nocontrol

    for _ in 1:poweriter

        cde = GirsanovSDE(sde, control)

        # evaluate koopman
        ys, ws = girsanovbatch(cde, xs, nkoop) :: Tuple{Array{Float64, 3},  Array{Float64, 2}}
        cs = model(ys)
        ks, std = vec.(StatsBase.mean_and_std(cs[1,:,:].*ws, 2))

        # estimate shift scale
        S = Shiftscale(ks)
        target = invert(S, ks)
        std = std ./ exp(S.q) / sqrt(nkoop)

        # train network
        for _ in 1:learniter
            l, grad = let xs=xs  # this let allows xs to not be boxed
                Zygote.withgradient(model) do model
                    sum(abs2, (model(xs)|>vec) .- target) / length(target)
                end
            end
            Optimisers.update!(opt, model, grad[1])
            push!(ls, l)
            push!(stds, mean(std))
        end

        cb(;losses=ls, model, xs, target, stds, std, cs)

        # update controls
        control = optcontrol(statify(model), S, sde)

        # resample xs uniformly along chi
        xys = hcat(xs, reshape(ys, size(xs, 1), :))
        cs = model(xys) |> vec
        xs = humboldtsample(xys, cs, nx; keepedges)
        #xs = randx0(dynamics, nx)
    end
    return (;model, ls, S, sde, cde, xs, dynamics)
end

function plot_callback(; kwargs...)
    (;losses, model, xs, target, std, stds) = NamedTuple(kwargs)

    let sqrloss = sqrt(losses[end]), std=stds[end]
        #@show sqrloss, std
        #@show log10(std)
    end

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
