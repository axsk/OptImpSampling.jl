using Plots
using LaTeXStrings
include("../src/sqra.jl")

doublewell(x) = ((x[1])^2 - 1) ^ 2
grid = -3:.05:3

Q=sqra(reshape([doublewell(x) for x in grid], :, 1), 1)
A=eigen(collect(Q), sortby=-)
v1 = A.vectors[:,1] / A.vectors[1,1]
v2 = A.vectors[:,2] / A.vectors[1,1]

chi = v2
chi = chi .- minimum(chi)
chi = chi / maximum(chi)
chi1=chi
chi2=1 .- chi

kchi = (chi .- 1/2) * 0.4 .+ 1/2

default(size=(300*1.6,300), title="", dpi=300)

plot(grid, doublewell, ylims=[0, 2], labels="U", yticks=[0,1])
savefig("plots/potential.png")

plot(grid, [v1 v2], labels=[L"v_1" L"v_2"], yticks=[-1,0,1])
savefig("plots/eigenfun.png")

plot(grid, [chi1 chi2], labels=[L"\chi_1" L"\chi_2"], yticks=[-1,0,1,.5])
plot!(grid, kchi, linestyle=:dash, alpha=.5, label=L"\mathbf{K}\chi_1")
savefig("plots/chifun.png")

scatter([chi1 v1], [chi2  v2], label=["χ" "v"], xticks=[0,1], yticks=[-1,1], xlabel="first component", ylabel="second component")
savefig("plots/compscat.png")
