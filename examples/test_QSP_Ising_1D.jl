using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

"""
 Here we compute the quantum signal as: C(t) = <psi_0 | O(0) O(t) | psi_0>
 where O(0) is an initial operator and O(t) the operator evolved at time t.
 psi_0 is an initial state. The evolution is performed for the 1D transverse field Ising model
 H = -J * (Sum_{<i,j>}(Z_i Z_j) + g * Sum_j(X_j)).
"""
function trott_unitary_sequence_1D_TFIM(o::Union{Pauli{N}, PauliSum{N}}; J=1.0, g=.1, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])

    # Loop over trotter steps
    for ki in 1:k
        ## ZZ layer
        # e^{i π/2 P2} e^{i π/2 P1}|ψ>
        for i in 1:N-1
            pi = Pauli(N, Z=[i, i + 1])
            push!(generators, pi)
            push!(parameters, -J/2)
        end
        #pbc 
        #pi = Pauli(N, Z=[N, 1])
        #push!(generators, pi)
        #push!(parameters, π/2)

        ## X layer
        # e^{i αn (-X) / 2}
        for i in 1:N
            Pi = Pauli(N, X=[i])
         #   pi = Pauli{N}(-pi.s, pi.z, pi.x) # this accounts for the fact that the papers have -X and positive ZZ
            push!(generators, Pi)
            push!(parameters, -0.5*J*g)
        end
    end

    return generators, parameters
end

#
#- - - Hamiltonian - - -
#
function hamiltonian_1D_TFIM(N, J=1.0, g=.01)
    H = PauliSum(N, Float64)
    ## ZZ layer
    for i in 1:N-1
        H += Pauli(N, Z=[i, i + 1])
    end

    ## X layer
    for i in 1:N
        H += g*Pauli(N, X=[i])
    end

    H *= -J
    coeff_clip!(H)

    return H
end

"""
 Evolve function from DBF code
"""
function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
            O[p] *= _cos
        end
    end
    sum!(O, sin_branch)
    return O 
end

"""
Pauli-propagation time series with pruning controls.
Arguments:
- generators, angles  : from UnitaryPruning.heisenberg_1D(...)
- o                   : PauliSum (used as both V and W by default)
- ket                 : state for estimator (your current approach)
- thresh              : magnitude threshold (|coeff|)

"""
function evolution_op(J, g, n_intervals, dt, trott_steps,
                      o::PauliSum{N}, ket;
                      thresh::Float64=1e-3) where {N}

    Wt = deepcopy(o)        # evolve W ≡ U*OU
    W  = deepcopy(o)        # initial operator 
    nsamp = Int(n_intervals) + 1
    rCtvals = Vector{Float64}([])#(undef, nsamp) # vector for C(t) values storing
    iCtvals = Vector{Float64}([])#(undef, nsamp)

    # t = 0 
    WW = W*W
    expval0 = expectation_value(WW, ket)

    C0real = real(expval0)
    C0imag = imag(expval0)

    push!(rCtvals, C0real)
    push!(iCtvals, C0imag)

    # Evolve W under Trotterization
    for ki in 1:n_intervals
        # Trotter terms U_1 U_2, ..., U_k
        generators, angles = trott_unitary_sequence_1D_TFIM(o, J=J, g=g, k=trott_steps)
        
        angles = dt * angles 
        nt = length(angles)

        # Access to the evolution of the operator by H = Sum(theta_i * P_i)
        for i in 1:nt
            Pi  = generators[i]
            theta = angles[i]
            pb = PauliBasis(Pi)
            evolve!(Wt, pb, theta)

            # --- POST pruning ---
            coeff_clip!(Wt, thresh=thresh)
            # -----------------------------
        end

        WWt = W * Wt # OTOC-like product
        expval = expectation_value(WWt, ket) # Contraction with reference ket
        Ctreal = real(expval) # Real part of C(t) = <O(0) * (U_i^ O U_i)>
        Ctimag = imag(expval)
        push!(rCtvals, Ctreal)
        push!(iCtvals, Ctimag)
    end

    tgrid = collect(range(0.0, stop=k*dt, length=length(rCtvals)))

    return rCtvals, iCtvals, tgrid
end



# Define model parameters
J = 1.0
g = 0.00#-0.01
N = 4 #Total number of qubits
H = hamiltonian_1D_TFIM(N, J, g)
@printf("1D-TFIM Hamiltonian (J= %.2f, g= %.2f): \n", J, g)
display(H)

Hmat = Matrix(H)
println("Lowest 5 Eigenvalues (of 16 for N=4):")
for (idx, eig) in enumerate(eigvals(Hmat)[1:5])
    @printf("  Eig %1i: %.6f \n", idx, eig)
end

# Define time evolution parameters
# Circuit divided in k layers
# Thus total time (t) is divided in dt = t/k time intervals
k = 100
t = 2.50
dt = t/k
trott_steps = 4
# Define initial ket and Initial operator to be evolved under the circuit
#ket = Ket(N,0)
ket = Ket{N}(12)
#o = Pauli(N, X=[1])
o = Pauli(N, X=[3,4])

# Call evolution_op. And get the evolved O(t) operator
threshold = 1e-4 #pruning threshold based on coeff.
rRES, iRES, tgrid = evolution_op(J, g, k, dt, trott_steps, PauliSum(o), ket; thresh=threshold)

# Number of snapshots actually returned
nsnap = length(rRES)
println("* * * * Number of snapshots collected: $nsnap")

# Print C(t) results
plt = plot(tgrid, rRES, lw=2, seriestype=:scatter,
          label="Re(C(t), th=$threshold")
plt = plot!(tgrid, iRES, lw=2, seriestype=:scatter,
          label="Im(C(t), th=$threshold")

xlabel!(plt, "Time"); ylabel!(plt, "< X_1(0)X_1(t) >")
title!(plt, "N=$N, J=$J, g=$g,dt=$dt")
savefig(plt, "QSP_ising_1D.pdf")

println("- - - Sanity Check: |C(t)|^2 - - - ")
@printf("idx    dt     Re(C(t))    Im(C(t))   |C(t)|^2   |C(t)|\n")
for (i,interval) in enumerate(tgrid)
    normop2 = rRES[i]^2 + iRES[i]^2
    normop = sqrt(normop2)
    @printf("%.2s    %.4f    %.6f   %.6f   %.6f  %.6f\n", i, interval, rRES[i], iRES[i], normop2, normop)
end

println("Initial operator O:")
display(o)
println("Initial eigenstate |0> ")
display(ket)
display(Vector(ket))
println("- - - H|0> - - - -")
Hpsi0 = Hmat*Vector(ket) 
display(Hpsi0) 
println("- - - <0|H|0> - - - -")
function compute_ref_expval(H, k)
    ref_expval = 0
    for (p, c) in H
        ref_expval += c * expectation_value(p, ket)
    end
    return ref_expval
end

ref_expval = compute_ref_expval(H, k)
println(ref_expval)



# SIGNAL PROCESSING multiply by exp(-iE_0t) to correct signal
signal = rRES .+ 1im * iRES;
phase = exp.(-1im .* tgrid); #-1 is the eigenvalue associated with the eigenvector (|0>)

#corrected signal F(t) = exp(-iE_0t)*C(t)
F = phase .* signal

# Print C(t) results
plt2 = plot(tgrid, real(F), lw=2, seriestype=:scatter,
          label="Re(F(t), th=$threshold")
plt2 = plot!(tgrid, imag(F), lw=2, seriestype=:scatter,
          label="Im(F(t), th=$threshold")

xlabel!(plt2, "Time"); ylabel!(plt2, "exp(-iE_0t) * < X_1(0)X_1(t) >")
title!(plt2, "N=$N, J=$J, g=$g,dt=$dt")
savefig(plt2, "QSP_ising_1D_corrct.pdf")

println("- - - Compare signals  - - - ")
@printf("dt   Re(C(t))    Im(C(t))   Re(F(t))    Im(F(t))\n")
for (i,interval) in enumerate(tgrid)
    normop2 = rRES[i]^2 + iRES[i]^2
    normop = sqrt(normop2)
    @printf("%.4f     %.6f    %.6f    %.6f   %.6f\n", interval, rRES[i], iRES[i], real(F[i]), imag(F[i]))
end

plt3 = plot(tgrid, rRES, lw=2, seriestype=:scatter,
          label="Re(C(t), th=$threshold")
plt3 = plot!(tgrid, iRES, lw=2, seriestype=:scatter,
          label="Im(C(t), th=$threshold")
plt3 = plot!(tgrid, real(F), lw=2, seriestype=:scatter,
          label="Re(F(t), th=$threshold")
plt3 = plot!(tgrid, imag(F), lw=2, seriestype=:scatter,
          label="Im(F(t), th=$threshold")

xlabel!(plt3, "Time"); ylabel!(plt3, "exp(-iE_0t) * < X_1(0)X_1(t) >")
title!(plt3, "Signal Comparison N=$N, J=$J, g=$g,dt=$dt")
savefig(plt3, "QSP_ising_1D_comparison.pdf")
