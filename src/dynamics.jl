# reconstruct the x-dynamics from the chi-kinetics

using LinearAlgebra: pinv
import ForwardDiff

"""
let χ = π(x), χ̇ = Q χ
then
χ̇ = ∇π(x) ẋ = Q π(x)
̇x = ∇π⁻¹(x) Q π(x)
"""

function dynamics(x, chi, Q)
    π  = chi(x)
    ∇π = jacobian(chi, x)
    ∇π⁻¹ = pinv(∇π)
    ẋ = ∇π⁻¹ * Q * π
    return ẋ
end

function dynamics(x, chi, S::Shiftscale)
    c = chi(x) :: AbstractVector{<:Number} # with 1 elem.
    pi  = [c[1]; 1-c[1]] :: AbstractVector{<:Number}
    dc = jacobian(chi, x)[1] :: AbstractMatrix
    dpi = [dc[1,1]; -dc[1,1]] :: AbstractVector{<:Number}
    return pinv(dpi) * Q(S) * pi
end

function Q(S::Shiftscale)
    q, a = S.q, S.a
    [q -a*q
     0 0]
end

function test_ss(S::Shiftscale=Shiftscale(.5,-1); chi=.4, T=1)

    s1 = S(chi, T)
    s2 = (exp(T*Q(S)) * [chi, 1])[1]
    if !isapprox(s1, s2)
        @show s1, s2
        error()
    end
end
