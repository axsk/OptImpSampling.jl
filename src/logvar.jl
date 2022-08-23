using LinearAlgebra
using StaticArrays
using Zygote
using DifferentialEquations, StochasticDiffEq, SciMLSensitivity
using StatsBase
using Lux, Random
using ReverseDiff

# dX = b + σ * v + σ * dB
# dY = -f - u'v + u'u/2
# Goal: find u sth that Var[log(Y(T))] is minimized (=0)
# Motivation: The optimal u-controlled process gives is a 0-variance estimator for
# W = ∫ f(X(t)) dt =

# control problem problem
b(X, t) = zero(X)
v(X, t) = zero(X)
f(x, T) = 1.
sigma(X, t) = collect(I(length(X)))
sigma(X::SVector{N, T}, t) where {N, T} = SMatrix{N, N, T}(I)

function mydense(in=1)
    m = Lux.Chain(Lux.Dense(in,10,tanh), Lux.Dense(10,1))
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, m)
    return m, ps, st
end

function drift(xy, t, b, sigma, u, v, f)
    X = allbutlast(xy)
    let b = b(X, t),  # is this beautiful or horrible :D?
            v = v(X, t),
            u = u(X, t),
            σ = sigma(X, t),
            f = f(X, t)

        dX = b + σ * v  # v controlled process
        dY = - f - dot(u, v) + dot(u, u) / 2  # runnig cost and u-v-W reweighting
        vcat(dX, dY)
    end
end

function drift(dxy, xy, t, b, sigma, u, v, f)
    #dxy[:] = drift(xy,t,b,sigma,u,v,f)
    X = @view xy[1:end-1]
    let b = b(X, t),  # is this beautiful or horrible :D?
        v = v(X, t),
        u = u(X, t),
        σ = sigma(X, t),
        f = f(X, t)

        dxy[1:end-1] .= b + σ * v
        dxy[end] = - f - dot(u, v) + dot(u, u) / 2
    end
end

function noise(dxy, xy, t, sigma, u)
    #dxy .= noise(xy, t, sigma, u)
    dxy .= 0
    X = @view xy[1:end-1]
    dxy[1:end-1, 1:end-1] .= sigma(X, t)
    dxy[end, 1:end-1] .= u(X,t)
end

allbutlast(s) = s[1:end-1]
allbutlast(s::SVector) = similar_type(s, Size(length(s)-1))(s[1:end-1]) # xy[1:end-1], but ::SVector
statify(X, x) = similar_type(X, Size(size(X)))(x)

""" construct the noise matrix consisting of the blocks
( Σ(X,t)  0 )
( u'(X,t) 0 )
"""
function noise(xy, t, sigma, u)
    #@show typeof(xy)
    X = allbutlast(xy)
    #@show typeof(X)
    dx = sigma(X, t) # SMatrix{1,1, Float64}
    uu = u(X,t)  # u is a Lux Chain, returning a Vector{Float64}
    dy = statify(X,uu) # converting to SVector
    dxy = vcat(dx, dy')
    ds = hcat(dxy, zero(dxy)) # returns SMatrix
end

# stop after first component of trajectory crosses lower or upper bound
termination(ub) = ContinuousCallback((u,t,int)->(u[1]-lb) * (ub-u[1]), terminate!)

function LogVarProblem(x0=@SVector([0.]), T=1., luxmodel=mydense())
    model, ps, st = luxmodel
    xy0 = vcat(x0, 0.)

    p = Lux.ComponentArray(ps)
    u(p) = (X, t) -> model(X, p, st)[1]  # ::: (X,t) -> R

    noise_proto = noise(xy0, 0., sigma, u(p))
    _drift(xy, p, t) = drift(xy, t, b, sigma, u(p), v, f)
    _noise(xy, p, t) = noise(xy, t, sigma, u(p))
    # note that we store the nn params `p` in the SDEProblem for later AD
    SDEProblem(_drift, _noise, xy0, T, p, noise_rate_prototype = noise_proto)
end


function MLogVarProblem(x0=@MVector([0.]), T=1., luxmodel=mydense())
    model, ps, st = luxmodel
    xy0 = vcat(x0, 0.)

    p = Lux.ComponentArray(ps)
    u(p) = (X, t) -> model(X, p, st)[1]  # ::: (X,t) -> R

    noise_proto = noise(xy0, 0., sigma, u(p))
    _drift(dxy, xy, p, t) = drift(dxy, xy, t, b, sigma, u(p), v, f)
    _noise(dxy, xy, p, t) = noise(dxy, xy, t, sigma, u(p))
    # note that we store the nn params `p` in the SDEProblem for later AD
    SDEProblem(_drift, _noise, xy0, T, p, noise_rate_prototype = noise_proto)
end

## @benchmark msolve(l::Prob{Vector,  Lux.Chain}) = 401 us
## @benchmark msolve(l::Prob{SVector, Lux.Chain}) = 163 us
function msolve(prob ;ps=prob.p, salg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), noisemixing=true), dt=0.01)
    prob = remake(prob, p=ps)
    s = solve(prob, EM(), sensealg=salg, dt=dt)[end][end]
end


## @benchmark msens(ls::Prob{SVector, Flux.Chain}) = 37ms
## @benchmark msens(l::Prob{Vector, Flux.Chain}) = 38ms
## @benchmark msens(l::Prob{Vector, Lux.Chain}) = 25ms
function msens(prob; kwargs...)
    Zygote.gradient(ps->msolve(prob;ps=ps, kwargs...), prob.p) |> first
end

function benchmark()
    l = LogVarProblem([0.])
    ls = LogVarProblem(@SVector [0.])

    @show @benchmark msolve(l)
    @show @benchmark msolve(ls)
    @show @benchmark msens(l)
    @show @benchmark msens(ls)
end

function logvar(p, n=100)
    var(msolve(p) for i in 1:n)
end

function dlogvar(p, n=100)
    Zygote.gradient(()->logvar(p, n), Flux.params(p.p)) |> first
end

function learnoptcontrol(p, n, m)
    data = Iterators.repeated((), m)
    opt = ADAM(0.1)
    Flux.train!(()->logvar(p,n), Flux.params(p.p), data, opt)
end

function test_ad_systems()
    senses = [BacksolveAdjoint, InterpolatingAdjoint, QuadratureAdjoint,
    ReverseDiffAdjoint,
    ForwardDiffSensitivity,
    ForwardSensitivity,
    ZygoteAdjoint,
    TrackerAdjoint]
    jvs = [false, true, ZygoteVJP(), ReverseDiffVJP(true), ReverseDiffVJP(), TrackerVJP()]
    i = 0
    for prob in [LogVarProblem([0.]),
        LogVarProblem(@SVector([0.])),
        MLogVarProblem([0.]),
        MLogVarProblem(@MVector([0.]))]
        i+=1
        for s in senses
            for j in jvs
                try
                    @show @benchmark msens($prob, salg=$(s(autojacvec=j)))
                    println("$i $s $j")
                catch e
                    println("$i  $s $j fail")
                end
            end
        end
    end
end

#=
Trial(84.003 ms) oop Array InterpolatingAdjoint ZygoteVJP(false)
Trial(23.119 ms) oop Array InterpolatingAdjoint ReverseDiffVJP{false}()

Trial(48.637 ms) inplace Array BacksolveAdjoint ReverseDiffVJP{false}()
Trial(17.011 ms) inplace Array InterpolatingAdjoint false
Trial(23.284 ms) inplace Array InterpolatingAdjoint ReverseDiffVJP{false}()

Trial(25.789 ms) inplace MArray InterpolatingAdjoint false
Trial(32.074 ms) inplace MArray InterpolatingAdjoint ReverseDiffVJP{false}()
=#

#=
### hence we compute the adjoint / derivatives manually :|

function AdjointLogVarProblem(x0::StaticArray, T, force, dudp)
    N = length(x0)
    M = size(dudp(x0), 2)
    s = vcat(x0, @SVector(zeros(M)))

    function ddrift(s::SVector, p, t)
        X = s[SOneTo(N)]  # indexing like this to keep it an SVector

        bb = b(X, t)  # is this beautiful or horrible :D?
        vv = v(X, t)
        uu = force(X)
        ss = sigma(X, t)
        ff = f(X, t)
        du = dudp(X)

        dX = bb + ss * vv  # v controlled process
        dY = du' * (uu - vv)
        vcat(dX, dY)

    end

    function dnoise(s, p, t)
        X = s[SOneTo(N)]
        ds = zero(MMatrix{N+M,N+M})
        ds[1:N, 1:N] = sigma(X, t)
        dd = dudp(X)
        ds[N+1:N+M, 1:N] .= dd'
        SMatrix(ds)
    end

    noise_proto = dnoise(s, 0, 0)

    SDEProblem{false}(ddrift, dnoise, s, T, noise_rate_prototype = noise_proto)
end

# return dY/dp (x) where Y is the cost functional and p the ann parameters
# next step would be to define the chain rule for this and use it in the variance minim.
function adjoint(x=rand(1), par=p1)
    x = SVector{length(x)}(x)
    u(x) = re(par)(x)
    ps = Flux.params(par)
    dudp(x) = Zygote.jacobian(()->u(x), ps) |> first
    dudp(x) :: AbstractMatrix
    p = AdjointLogVarProblem(x, 1., u, dudp)
    sol = solve(p, EM(), dt=.1)
    sol[end][length(x)+1:end]
end



## backlog

# use a terminal condition
termination = ContinuousCallback((u,t,int)->u[1]-1, terminate!)

function compare_static_mutating()
    p = LogVarProblem([0.], 1.)
    @time s = solve(p, EM(), dt=.001)
    p = LogVarProblem(@SVector[0.], 1.)
    @time s = solve(p, EM(), dt=.001);
end


""" mutating variant, was around 10x slower """
function LogVarProblemMutating(x0, T)

    function drift(ds, s, p, t)
        X = @view s[1:end-1]

        let b = b(X, t),  # is this beautiful or horrible :D?
            v = v(X, t),
            u = u(X, t),
            σ = sigma(X, t),
            f = f(X, t)

            ds[1:end-1] .= b + σ * v  # v controlled process
            ds[end]     = - f - dot(u, v) + dot(u, u) / 2
        end
        nothing
    end

    function noise(ds, s, p, t)
        X = @view s[1:end-1]

        dWX = ds[1:end, 1:end] .= sigma(X, t)
        dWY = ds[end, 1:end-1] .= u(X, t)
        nothing
    end
    Sigma(X) = similar(X, size(X).+1 ...)
    SDEProblem{true}(drift, noise, [x0; 0], (0, T), p, noise_rate_prototype = Sigma(x0))
end

=#
