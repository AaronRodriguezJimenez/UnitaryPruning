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

# return a PauliOperators Ket equivalent to a given bitstring
function string_to_ket(bits::String)
    b = collect(bits)
    v = parse.(Int128, b)
    N = length(v)
    out = 0
    count = 0

    for bit in v
        if bit%2 == 1
            out += 2^count
        end
        count +=1
    end
    ket = Ket{N}(out)
    return ket, out
end

"""
 Here we compute the quantum signal as: C(t) = <psi_0 | O(0) O(t) | psi_0>
 where O(0) is an initial operator and O(t) the operator evolved at time t.
 psi_0 is an initial state. The evolution is performed for the 1D Heisenverg model
 H = -J * (Sum_{<i,j>}(X_i X_j + Y_i Y_j) -Jz * (Sum_{<i,j>}(Z_i Z_j) + gx * Sum_j(X_j)) gy * Sum_j(Y_j)) + gz * Sum_j(Z_j)).

"""
function trott_unitary_sequence_Heisenberg(o::Union{Pauli{N}, PauliSum{N}}; Jx=1.0, 
                                   Jy=1.0, Jz=1.0, gx=0.1, gy=0.1, gz= 0.1, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])

    # Loop over trotter steps
    for ki in 1:k
        ## XX + YY layer
        for i in 0:N-1
            piX = Pauli(N, X=[i+1,(i+1)%(N)+1])
            push!(generators, piX)
            push!(parameters, -1.0*Jx)

            piY = Pauli(N, Y=[i+1,(i+1)%(N)+1])
            push!(generators, piY)
            push!(parameters, -1.0*Jy) #Positive to account for a change of sign
        end

        ## ZZ layer
        # e^{i π/2 P2} e^{i π/2 P1}|ψ>
        for i in 0:N-1
            piZ = Pauli(N, Z=[i+1,(i+1)%(N)+1])
            push!(generators, piZ)
            push!(parameters, -1.0*Jz)
        end
        
        if gx != 0.0
            ## x layer
            # e^{i αn (-X) / 2}
            for i in 0:N
                Pi = Pauli(N, X=[i])
                push!(generators, Pi)
                push!(parameters, -1.0*gx)
            end
        end

        if gy != 0.0
            ## y layer
            # e^{i αn (-Y) / 2}
            for i in 0:N
                Pi = Pauli(N, Y=[i])
                push!(generators, Pi)
                push!(parameters, -1.0*gy)
            end
        end

        if gz != 0.0
            ## z layer
            # e^{i αn (-X) / 2}
            for i in 0:N
                Pi = Pauli(N, X=[i])
                push!(generators, Pi)
                push!(parameters, -1.0*gz)
            end
        end

    end

    return generators, parameters
end

#
#- - - Hamiltonian - - -
#
function heisenberg_1D(N, Jx, Jy, Jz; x=0, y=0, z=0)
    H = PauliSum(N, Float64)
    for i in 0:N-1
        H += -Jx * Pauli(N, X=[i+1,(i+1)%(N)+1])
        H += -Jy * Pauli(N, Y=[i+1,(i+1)%(N)+1])
        H += -Jz * Pauli(N, Z=[i+1,(i+1)%(N)+1])
    end 
    for i in 1:N
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
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

# Helper: extract PauliBasis operators and coefficients from PauliSum
function extract_hamiltonian_coeffs_and_ops(H::PauliSum{N,T}) where {N,T}
    ops = PauliBasis{N}[]
    coeffs = Float64[]
    for (p, c) in H
        push!(ops, p)
        push!(coeffs, float(c))
    end
    return ops, coeffs
end

"""
Pauli-propagation time series with pruning controls.
Arguments:
- generators, angles  : from UnitaryPruning.heisenberg_1D(...)
- o                   : PauliSum (used as both V and W by default)
- ket                 : state for estimator (your current approach)
- thresh              : magnitude threshold (|coeff|)
- n_intervals : number of intervals,  n_intervals =tot_time/dt
"""
function evolution_op(ket, o::PauliSum{N,T}, H::PauliSum{N,T}, n_intervals, dt;
                      thresh::Float64=1e-3) where {N,T}

    #err = Vector{Float64}([])
    Wt = deepcopy(o)        # evolve W ≡ U*OU
    W  = deepcopy(o)        # initial operator 
    rCtvals = Vector{Float64}([])#(undef, nsamp) # vector for C(t) values storing
    iCtvals = Vector{Float64}([])#(undef, nsamp)

    # t = 0 
    WW = W*W
    expval0 = expectation_value(WW, ket)

    C0real = real(expval0)
    C0imag = imag(expval0)

    push!(rCtvals, C0real)
    push!(iCtvals, C0imag)

    # Extract Pauli strings and coefficients
    generators, angles = extract_hamiltonian_coeffs_and_ops(H)
    #@printf("Hamiltonian has %d terms \n", length(coeffs))

    nt = length(angles)                            
    println("Total Rotations:", nt * n_intervals)
    # Evolve W under Trotterization

    #accumulated_error = 0
    for ki in 1:n_intervals
            # Trotter terms U_1 U_2, ..., U_k
            accumulated_error = 0

            for j in 1:nt
                # Access to the evolution of the operator by H = Sum(theta_i * P_i)
                Pi  = generators[j]
                theta = 2 * dt * angles[j]
                pb = PauliBasis(Pi)
                evolve!(Wt, pb, theta)
                
                coeff_clip!(Wt, thresh=1.0e-12)
                e1 = expectation_value(W * Wt, ket)

                # --- POST pruning ---
                coeff_clip!(Wt, thresh=thresh)
                #WWt = W * Wt 
                e2 = expectation_value(W * Wt, ket)
                accumulated_error += e2-e1
            end           
            
            WWt = W * Wt #(at time interval)
            expval = expectation_value(WWt, ket) + accumulated_error # Contraction with reference ket
            Ctreal = real(expval) # Real part of C(t) = <O(0) * (U_i^ O U_i)>
            Ctimag = imag(expval)
            push!(rCtvals, Ctreal)
            push!(iCtvals, Ctimag)
    end

    tgrid = collect(range(0.0, stop=n_intervals*dt, length=length(rCtvals)))
    
    return rCtvals, iCtvals, tgrid
end

# Define model parameters
Jx = 1.0
Jy = 1.0
Jz = 2.0
gx = 0.0
gy = 0.0
gz = 0.0
g = 0.00#-0.01
N = 10 #Total number of qubits
H = heisenberg_1D(N, Jx, Jy, Jz)
@printf("1D-Heisenberg Hamiltonian (J= %.2f, Jz= %.2f): \n", Jx, Jz)
display(H)

# Define time evolution parameters
# Circuit divided in k layers
n_intervals = 100
t = 50.0 #Total Time Evolution
dt = t/n_intervals
threshold = 1e-4 #pruning threshold based on coeff.

# Define initial ket and Initial operator to be evolved under the circuit
ket = Ket(N,1)
#ket = Ket{N}(12)
#ket, _ = string_to_ket("1000")
#ket, _ = string_to_ket("1000000000")
o = Pauli(N, X=[1])
o = PauliSum(o)
#o = Pauli(N, X=[3,4])
#o += Pauli(N,X=[1,2]) #Amplitude modulation
#o += Pauli(N,X=[1,2,3,4]) #mixed signal
#o += Pauli(N,X=[3])  #mixed signal
# += Pauli(N,X=[2,3]) #Imaginary suppression


t1 = time()
rRES, iRES, tgrid = evolution_op(ket, o, H, n_intervals, dt; thresh=threshold)

                                  # Code block to measure
elapsed_time = time() - t1
println("Elapsed time: ", elapsed_time, " seconds")

# Number of snapshots actually returned
nsnap = length(rRES)
println("* * * * Number of snapshots collected: $nsnap")

# Print C(t) results
plt = plot(tgrid, rRES, lw=2, seriestype=:scatter,
          label="Re(C(t), th=$threshold")
plt = plot!(tgrid, iRES, lw=2, seriestype=:scatter,
          label="Im(C(t), th=$threshold")

xlabel!(plt, "Time"); ylabel!(plt, "< O(0)O(t) >")
title!(plt, "N=$N, J=$Jx, Jz=$Jz,dt=$dt")
savefig(plt, "/Users/admin/VSCProjects/UnitaryPruning/QSP_XXZ_1D/$N-Q_QSP_XXZ.pdf")

println("- - - Sanity Check: |C(t)|^2 - - - ")
@printf("idx    dt     Re(C(t))    Im(C(t))   |C(t)|^2   |C(t)|\n")
for (i,interval) in enumerate(tgrid)
    normop2 = rRES[i]^2 + iRES[i]^2
    normop = sqrt(normop2)
    @printf("%.2s    %.4f    %.6f   %.6f   %.6f  %.6f\n", i, interval, rRES[i], iRES[i], normop2, normop)
end

println("Initial operator O:")
display(o)
println("Initial state |Psi0> ")
display(ket)
#display(Vector(ket))

println("- - - <0|H|0> - - - -")
function compute_ref_expval(H, psi0)
    ref_expval = 0
    for (p, c) in H
        ref_expval += c * expectation_value(p, psi0)
    end
    return ref_expval
end

ref_expval = compute_ref_expval(H, ket)
println(ref_expval)


#- - - - - PHASE CORRECTION (ONLY WHEN REF STATE IS ALSO AN EIGENSTATE) - - - - -
# SIGNAL PROCESSING multiply by exp(-iE_0t) to correct signal
#Ek = 0.0#-8 #Eigenvalue associated to ref state in case of correction.
#signal = rRES .+ 1im * iRES;
#phase = exp.(1im * Ek .* tgrid); #-1 is the eigenvalue associated with the eigenvector (|0>)
#corrected signal F(t) = exp(-iE_0t)*C(t)
#F = phase .* signal
# Print C(t) results
#plt2 = plot(tgrid, real(F), lw=2, seriestype=:scatter,
#          label="Re(F(t), th=$threshold")
#plt2 = plot!(tgrid, imag(F), lw=2, seriestype=:scatter,
 #         label="Im(F(t), th=$threshold")

#xlabel!(plt2, "Time"); ylabel!(plt2, "exp(-iE_t) * < O(0)O(t) >")
#title!(plt2, "N=$N, J=$Jx, Jz=$Jz,dt=$dt")
#savefig(plt2, "/Users/admin/VSCProjects/UnitaryPruning/QSP_XXZ_1D/$N-Q_QSP_XXZ_corrct.pdf")

using Printf

open("/Users/admin/VSCProjects/UnitaryPruning/QSP_XXZ_1D/$N-Q_signals.txt", "w") do io
    println(io, "- - - XXZ $N qubits output signals  - - - ")
    @printf(io, "dt   Re(C(t))    Im(C(t))   Re(F(t))    Im(F(t))\n")

    for (i, interval) in enumerate(tgrid)
        #normop2 = rRES[i]^2 + iRES[i]^2
        #normop = sqrt(normop2)
        @printf(io, "%.4f     %.6f    %.6f\n",
                interval, rRES[i], iRES[i])
    end
end

