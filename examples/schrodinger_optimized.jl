# Optimizing Schrodinger evolution functions.
using UnitaryPruning
using KrylovKit
using LinearAlgebra
using PauliOperators
using BenchmarkTools
using ExponentialUtilities  

"""
  This function precomputed the action p * ket(N,i)
  in this version, we precompute a lookup table for (i -> j, phase)
  once per operation and reuse it.   
"""
function apply_pauli_index_phase(p::Pauli{N}, i::Int) where N
    coeff, ketj = p * Ket(N, i)  
    
    return ketj.v, coeff
end

function pauli_action_map(p::Pauli{N}) where N
    map = Vector{Tuple{Int, ComplexF64}}(undef, 2^N)
    for i in 0:2^N - 1
        j, phase = apply_pauli_index_phase(p, i)
        map[i + 1] = (j, phase)
    end    
    return map
end

"""
  Updated matvec function removes the Pauli*Ket steps from the loop
  and avoids repeated allocations: Instead of allocating a new output vector every time,
  reuse one.
"""
function matvec!(σ::AbstractVector{T}, action_map::Vector{Tuple{Int,ComplexF64}}, coeff::Number, V::AbstractVector{T}) where {T<:Complex}
    fill!(σ, 0)
    for i in eachindex(V)
        j, phase = action_map[i]
        σ[j + 1] += coeff * V[i] * phase
    end
    return σ
end


function compute_schrodinger_evol(generators, parameters, ref_ket)
    nt = length(generators)
    length(parameters) == nt || throw(DimensionMismatch)

    # Promote input to ComplexF64 (if not already)
    U_psi = ComplexF64.(ref_ket)
    
    # Allocate a ComplexF64 buffer for intermediate results
    Pψ = zeros(ComplexF64, length(U_psi))

    # Precompute Pauli action maps
    action_maps = [pauli_action_map(p) for p in generators]

    for t in 1:nt
        α = parameters[t]
        matvec!(Pψ, action_maps[t], 1.0, U_psi)
        @. U_psi = cos(α/2) * U_psi - 1im * sin(α/2) * Pψ
    end

    return U_psi / norm(U_psi)
end


# === Example to check validity ===

N = 32
generators = [Pauli(N, Z=[1], X=[2]), Pauli(N, Z=[1,2]), Pauli(N, Z=[2])]
parameters = [3.14/2, 0.6, 0.3]
o = Pauli(N, Z=[2])
ket = Ket(N, 2)

vector_ket = ComplexF64.(Vector(ket))
#println("Vector ket: $ket -> ", vector_ket)

U_psi = compute_schrodinger_evol(generators, parameters, vector_ket)

# Precompute action map for observable o and use matvec!
obs_map = pauli_action_map(o)
oU = zeros(ComplexF64, length(U_psi))
matvec!(oU, obs_map, 1.0, U_psi)

expval = U_psi' * oU
println("Expval Schr (optimized) :", expval)

# Compare with Heisenberg picture using full matrix
#U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
#o_mat = Matrix(o)
#expval_heis = vector_ket' * (U' * o_mat * U) * vector_ket
#println("Expval Heis (matrix-based) :", expval_heis)
