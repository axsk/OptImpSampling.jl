using StochasticDiffEq
using LinearAlgebra
using StaticArrays

dim = 1
sigma(X, t) = ones(dim,dim)
Sigma(X) = similar(X, size(X).+1 ...)
b(X, t) = zero(X)
v(X, t) = zero(X)
u(X, t) = zero(X)
f(x, T) = 1
sigma(X::SVector{N, T}, t) where {N, T} = SMatrix{N, N, T}(I)

""" immutable """
function drift(s::SVector, p, t)
    N = length(s) - 1
    X = s[SOneTo(N)]  # indexing like this to keep it an SVector
    let b = b(X, t),  # is this beautiful or horrible :D?
        v = v(X, t),
        u = anneval(p, X),
        σ = sigma(X, t),
        f = f(X, t)

        dX = b + σ * v  # v controlled process
        dY = - f - dot(u, v) + dot(u, u) / 2  # runnig cost and u-v-W reweighting
        vcat(dX, dY)
    end
end

function noise(s::SVector, p, t)
    N = length(s) - 1
    X = s[SOneTo(N)]
    ds = zero(MMatrix{N+1,N+1})
    ds[1:N, 1:N] = sigma(X, t)
    ds[N+1, 1:N] .= anneval(p, X)
    SMatrix(ds)
end

ann = Chain(Dense(1,10,tanh), Dense(10,1))
anneval(p, X) = re(p)(X) # evaluation at X with parameters p
p1,re = Flux.destructure(ann)
ps = Flux.params(p1)

LogVarProblem(x0=[0.], T=1., p=p1) = LogVarProblem(SVector(x0...), T, p)

function LogVarProblem(x0::StaticArray, T, p)
    s = vcat(x0, 0)
    noise_proto = noise(s, p, 0)
    SDEProblem{false}(drift, noise, s, T, p, noise_rate_prototype = noise_proto)
    ODEProblem{false}(drift, s, T, p)
end

function msolve()
    p = LogVarProblem()
    s = solve(p, dt=.001)
end

function msens()
    p = LogVarProblem()
    Zygote.gradient(()->msolve()[end][2], ps)
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


""" mutating variant """
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

    SDEProblem{true}(drift, noise, [x0; 0], (0, T), p, noise_rate_prototype = Sigma(x0))
end
