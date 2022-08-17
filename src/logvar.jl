using LinearAlgebra
using StaticArrays
using Flux, Zygote
using DifferentialEquations, StochasticDiffEq, SciMLSensitivity
using StatsBase

# control problem problem
dim = 1
sigma(X, t) = ones(dim,dim)
b(X, t) = zero(X)
v(X, t) = zero(X)
u(X, t, p) = anneval(p, X)
u(X::SVector{T,S}, t, p) where {T,S} = SVector{T,S}(anneval(p, X)) :: SVector{T,S}
f(x, T) = 1.
sigma(X::SVector{N, T}, t) where {N, T} = SMatrix{N, N, T}(I)
x0 = [0.]

struct UModel{T, U, V}
    ann::T
    p::U
    re::V
end

UModel(n) = UModel(Chain(Dense(1,10,tanh), Dense(10,1)))
UModel(ann::Chain) = UModel(ann, destructure(ann)...)
(u::UModel)(x, t) = u.re(u.p)(x)
params(u::UModel) = Flux.params(u.ann)

# neural network modeling force u
ann = Chain(Dense(1,10,tanh), Dense(10,1))
anneval(p, X) = re(p)(X) # evaluation at X with parameters p
p1, re = Flux.destructure(ann)
ps = Flux.params(p1)

allbutlast(s) = s[1:end-1]
allbutlast(s::SVector) = s[SOneTo(length(s)-1)]

function drift(s, p, t)
    X = allbutlast(s)
    let b = b(X, t),  # is this beautiful or horrible :D?
            v = v(X, t),
            u = u(X, t, p),
            σ = sigma(X, t),
            f = f(X, t)

        dX = b + σ * v  # v controlled process
        dY = - f - dot(u, v) + dot(u, u) / 2  # runnig cost and u-v-W reweighting
        vcat(dX, dY)
    end
end

function noise(s, p, t)
    X = allbutlast(s)
    dx = sigma(X, t)
    dy = u(X, t, p)'
    dxy = vcat(dx, dy)
    ds = hcat(dxy, zeros(length(s)))
end

function LogVarProblem(x0=x0, T=1., p=p1)
    s = vcat(x0, 0)
    noise_proto = noise(s, p, 0.)
    SDEProblem{false}(drift, noise, s, T, p, noise_rate_prototype = noise_proto)
end

# normal solve works
function msolve(;salg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(), noisemixing=true), x0=x0)
    p = LogVarProblem(x0)
    alg = EM()
    s = solve(p, alg, sensealg=salg, dt=.01)[end][end]
end

# sensivity doesnt
function msens(;kwargs...)
    p = LogVarProblem()
    Zygote.gradient(()->msolve(;kwargs...), ps) |> first
end

function logvar(n=100; kwargs...)
    var(msolve() for i in 1:n)
end

function dlogvar()
    Zygote.gradient(()->logvar(), ps) |> first
end

function learnoptcontrol()
    data = Iterators.repeated((), 100)
    opt = ADAM(0.1)
    Flux.train!(logvar, ps, data, opt)
end

function test()
    senses = [BacksolveAdjoint, InterpolatingAdjoint, QuadratureAdjoint,
    ReverseDiffAdjoint,
    ForwardDiffSensitivity,
    ForwardSensitivity,
    ZygoteAdjoint,
    TrackerAdjoint]
    jvs = [false, true, ZygoteVJP(), ReverseDiffVJP(true), ReverseDiffVJP(), TrackerVJP()]
    for s in senses
        for j in jvs
            @time try
                msens(salg=s(autojacvec=j))
                println("$s $j okay")
            catch e
                println("  $s $j fail")
            end
        end
    end
end

#=

ReverseDiffAdjoint(): ERROR: MethodError: no method matching *(::Vector{Float64}, ::Vector{Float64})
ZygoteAdjoint(): try catch in EM()
ForwardDiffSensitivity(): 0.02s
QuadratureAdjoint(): Incompatible problem+solver pairing.
InterpolatingAdjoint(): 0.05s
BacksolveAdjoint(): 0.1s
BacksolveAdjoint(autojacvec=false): 0.08
BacksolveAdjoint(autojacvec=true): type Nothing has no field t
BacksolveAdjoint(autojacvec=ZygoteVJP):
BacksolveAdjoint(autojacvec=ReverseDiffVJP()): ERROR: MethodError: no method matching mul!(::Vector{Float64}, ::Matrix{Float64}, ::Adjoint{Float64, Vector{Float64}}, ::Bool, ::Bool)
InterpolatingAdjoint(autojacvec=ZygoteVJP())): 0.05
InterpolatingAdjoint(autojacvec=ReverseDiffVJP()): 0.03
InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))): no method matching compile(::Nothing)



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
