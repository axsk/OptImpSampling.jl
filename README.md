# OptImpSampling.jl

A WIP collection of algorithms related to optimal importance sampling (OIS)

- ociso: OIS for Koopman eigen- and chi functions
- control.jl: simpler rewrite of ociso
- isokann: ISOKANN using OIS for the chi approximations
- isonew: rewrite of isokann, uses humboldtsampling
- isokann2: ISOKANN in higher dimensions
- sqra: Square root approximation for the generators of Ito diffusions
- logvar: OIS  of path functionals with control variates (implements Richter, Nusken 2021)
- reinforce: OIS of functions via REINFORCE
- reinforcepath: like reinfore but on path space (equivalent to logvar)
- fbsde: Forward backward SDE with Neural Controls, learning all times at once (like Kebiri, Hartmann 2019)
