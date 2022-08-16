using LinearAlgebra
using StaticArrays
using Flux, Zygote
using DifferentialEquations, StochasticDiffEq, SciMLSensitivity

# control problem problem
dim = 1
sigma(X, t) = ones(dim,dim)
Sigma(X) = similar(X, size(X).+1 ...)
b(X, t) = zero(X)
v(X, t) = zero(X)
u(X, t, p) = anneval(p, X)
f(x, T) = 1.
sigma(X::SVector{N, T}, t) where {N, T} = SMatrix{N, N, T}(I)
x0 = @SVector [0.]

# neural network modeling force u
ann = Chain(Dense(1,10,tanh), Dense(10,1))
anneval(p, X) = re(p)(X) # evaluation at X with parameters p
p1,re = Flux.destructure(ann)
ps = Flux.params(p1)

# went for immutable SVectors since they were much faster in normal sde solve
function drift(s::SVector, p, t)
    N = length(s) - 1
    X = s[SOneTo(N)]  # indexing like this to keep it an SVector
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

function noise(s::SVector, p, t)
    N = length(s) - 1
    X = s[SOneTo(N)]

    dx = sigma(X, t)
    dy = u(X, t, p)
    z  = zero(SVector{N+1})

    ds = hcat(vcat(dx, dy), z)  # non mutating way to construct the noise matrix
    SMatrix{N+1, N+1}(ds)
end

## using Zygote.Buffer to allow mutation of array, doesnt help though
function noise_buffered(s::SVector, p, t)
    N = length(s) - 1
    X = s[SOneTo(N)]
    #ds = zero(MMatrix{N+1,N+1})
    @show uu = u(X, t, p)
    ds = Zygote.Buffer(s, N+1, N+1) # do we need to zero this?
    ss = sigma(X, t)
    for i in 1:N
        for j in 1:N
            ds[i, j] = ss[i, j]
        end
    end


    for j in 1:N
        ds[N+1, j] = uu[j]
    end
    SMatrix{N+1,N+1}(copy(ds))
end

# default problem
LogVarProblem(x0=[0.], T=1., p=p1) = LogVarProblem(SVector(x0...), T, p)

function LogVarProblem(x0::StaticArray, T, p)
    s = vcat(x0, 0)
    noise_proto = noise(s, p, 0)
    SDEProblem{false}(drift, noise, s, T, p, noise_rate_prototype = noise_proto)
end

# normal solve works
function msolve()
    p = LogVarProblem()
    alg = EM()
    salg = InterpolatingAdjoint(checkpointing=true)
    s = solve(p, alg, dt=.01, sensealg=salg)[end][end]
end

# sensivity doesnt
function msens()
    p = LogVarProblem()
    Zygote.gradient(()->msolve(), ps)
end

function test()
    msens()
end


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

    SDEProblem{true}(drift, noise, [x0; 0], (0, T), p, noise_rate_prototype = Sigma(x0))
end

=#
