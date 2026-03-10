"""
 Here we get the total number of rotations needed in a deterministic calculation
 given a Hamiltonian
"""
#
using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random
using StatsBase

#
# - - - Clipping functions - - -
#
function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
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

function deterministic_rots(H::PauliSum{N,T}, tot_time, nmeas) where {N,T}
    # Extract Pauli strings and coefficients
    ops, coeffs = extract_hamiltonian_coeffs_and_ops(H)

    L = length(coeffs)
    if L == 0
        error("Hamiltonian has no terms.")
    end

    @printf("Hamiltonian has %d terms \n", L)
    dt = tot_time/nmeas
    tot_N_rotations = nmeas * L
    return tot_N_rotations
end


function run()
    # XXZ model parameters
    n_meas = 100 #Number of measurements
    t = 2.5
    Jx, Jy, Jz = 1.0, 1.0, 2.0
    Nqubits_lst = [4,10,20,40,60,80,100]

    for qubit in Nqubits_lst
        H = heisenberg_1D(qubit, Jx, Jy, Jz)
        NRots = deterministic_rots(H, t, n_meas)
        @printf("NQubits :  %d   NRots : %d ", qubit, NRots)
    end
end

run()
