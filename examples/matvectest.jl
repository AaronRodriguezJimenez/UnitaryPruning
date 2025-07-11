using UnitaryPruning
using LinearAlgebra
using PauliOperators
using BenchmarkTools
using ExponentialUtilities  

"""
  matvec fucntion return the effect of Operator o applied to the vector V
  which must correspond to some compatible ket    
"""
function apply_pauli_index_phase(p::Pauli{N}, i::Union{Int128, Int64}) where N
    coeff, ketj = p * Ket(N, i)  
    return ketj.v, coeff
end

function matvec(o::Pauli{N}, coeff::Number, V::Vector) where N
    σ = zeros(promote_type(typeof(coeff), eltype(V), ComplexF64), length(V))

    for i in 0:2^N - 1
        j, phase = apply_pauli_index_phase(o, i)  # returns target index and phase
        
        σ[j + 1] += coeff * V[i + 1] * phase
    end

    return σ
end

function matvec_lite(o::Pauli{N}, coeff::Number, V::Ket{N}) where N
    #σ = zeros(promote_type(typeof(coeff), eltype(V), ComplexF64), length(V))
    indx = V.v
    j, phase = apply_pauli_index_phase(o, indx)
    return Ket(N,j), phase*coeff
end

# Next we focus on compute the previous element m[1] in the Shcrodinger picture..
function compute_schrodinger_evol(generators, parameters, ref_ket)
    nt = length(generators)
    length(parameters) == nt || throw(DimensionMismatch)

    U_psi = copy(ref_ket)

    for t in 1:nt
        α = parameters[t]
        Pψ = matvec(generators[t], 1.0, U_psi)
        U_psi = cos(α/2) .* U_psi - 1im * sin(α/2) .* Pψ
        U_psi /= norm(U_psi)  # Optional: normalize at each step
    end

    return U_psi
end

## Testing Function matvec
p = Pauli("YI")  # or any multi-qubit Pauli
v = ComplexF64[1.0, 0.0, 0.0, 0.0]  # example: |00⟩
# Direct matrix multiplication
Mv = Matrix(p) * v

# Implementation
N = 128
ket = Ket(N,7)
p = Pauli(N, Z=[2])
mv, phi = matvec_lite(p, 1.0, ket)
#print(mv, phi)
#mv = phi*Vector(mv)
println(phi*expectation_value(p, mv))
# Compare
#println("M*v vs m*v - Are they equal? ", isapprox(Mv, mv))

#function compute_schrodinger_sparse_evol(generators, parameters, ref_ket::Ket{N}) where N
#    ψ = Dict{Int128, ComplexF64}()
#    ψ[ref_ket.v] = 1.0 + 0im
#
#    for t in eachindex(generators)
#        α = parameters[t]
#        Pψ = Dict{Int128, ComplexF64}()
#
#        # Apply Pauli to each basis state in ψ
#        for (i, amp) in ψ
#            j, phase = apply_pauli_index_phase(generators[t], i)
#            Pψ[j] = get(Pψ, j, 0.0) + amp * phase
#        end
#
#        # Schrödinger update
#        newψ = Dict{Int128, ComplexF64}()
#        for key in union(keys(ψ), keys(Pψ))
#            newψ[key] = cos(α/2)*get(ψ, key, 0.0) - 1im * sin(α/2)*get(Pψ, key, 0.0)
#        end
#
#        ψ = newψ
#    end
#
#    return ψ  # final wavefunction in sparse basis form
#end

function compute_schrodinger_sparse_evol(generators, parameters, ref_ket::Ket{N}) where N
    ψ = Dict{Int128, ComplexF64}()
    ψ[ref_ket.v] = 1.0 + 0im
    Pψ = Dict{Int128, ComplexF64}()
    newψ = Dict{Int128, ComplexF64}()

    for t in eachindex(generators)
        empty!(Pψ)
        α = parameters[t]
        cosα2 = cos(α / 2)
        sinα2 = sin(α / 2)

        for (i, amp) in ψ
            j, phase = apply_pauli_index_phase(generators[t], i)
            Pψ[j] = get(Pψ, j, 0.0) + amp * phase
        end

        empty!(newψ)
        for key in keys(ψ)
            ψ_val = ψ[key]
            P_val = get(Pψ, key, 0.0)
            newψ[key] = cosα2 * ψ_val - 1im * sinα2 * P_val
        end
        for key in keys(Pψ)
            if !haskey(ψ, key)
                newψ[key] = -1im * sinα2 * Pψ[key]
            end
        end

        ψ, newψ = newψ, ψ
    end

    return ψ
end


function expectation(ψ::Dict{Int128, ComplexF64}, o::Pauli{N}) where N
    result = 0.0 + 0im
    for (i, amp) in ψ
        j, phase = apply_pauli_index_phase(o, i)
        result += conj(amp) * get(ψ, j, 0.0) * phase
    end
    return result
end

function compute_schrodinger_array_evol(generators, parameters, ket::Ket{N}) where N
    inds = [ket.v]                     # List of basis indices
    amps = [1.0 + 0im]                 # Corresponding amplitudes

    nt = length(generators)
    for i in 1:nt
        α = parameters[i]
        g = generators[i]
        
        new_inds = Int128[]
        new_amps = ComplexF64[]
        seen = Dict{Int128,Int}()

        ni = length(inds)
        for j in 1:ni
            idx = inds[j]
            amp = amps[j]

            idx_p, phase = apply_pauli_index_phase(g, idx)

            # Compute the updated amplitude
            ψ0 = cos(α/2) * amp
            ψ1 = -1im * sin(α/2) * phase * amp

            # Accumulate ψ0 (original idx)
            if haskey(seen, idx)
                new_amps[seen[idx]] += ψ0
            else
                push!(new_inds, idx)
                push!(new_amps, ψ0)
                seen[idx] = length(new_inds)
            end

            # Accumulate ψ1 (transformed idx_p)
            if haskey(seen, idx_p)
                new_amps[seen[idx_p]] += ψ1
            else
                push!(new_inds, idx_p)
                push!(new_amps, ψ1)
                seen[idx_p] = length(new_inds)
            end
        end

        inds, amps = new_inds, new_amps
    end

    return inds, amps
end

function expectation_from_sparse(inds, amps, o::Pauli{N}) where N
    acc = 0.0 + 0im
    ni = length(inds)
    for i in 1:ni
        idx = inds[i]
        amp = amps[i]
        j, phase = apply_pauli_index_phase(o, idx)
        
        pos = findfirst(isequal(j), inds)
        if pos !== nothing
            acc += conj(amp) * phase * amps[pos]
        end
    end
    return acc
end

N = 128
generators = [Pauli(N, Z=[1], X=[2]), Pauli(N, Z=[1,2]), Pauli(N, Z=[2])]
parameters = [3.14/2, 0.6, 0.3]
o = Pauli(N, Z=[2])
ket = Ket(N,2)
bra = Bra(N,1)

#vector_ket = Vector(ket)
#println("Vector ket: $ket -> ", vector_ket)

#U_psi = compute_schrodinger_evol(generators, parameters, vector_ket)
#println("res :", U_psi)

#expval = U_psi' * matvec(o, 1.00, U_psi)
#println("Expval Schr :", expval)

ψ = compute_schrodinger_sparse_evol(generators, parameters, ket)

println("ψ = ", ψ)
println("sparse_expval = ", expectation(ψ, o))


inds, amps = compute_schrodinger_array_evol(generators, parameters, ket)
expval =  expectation_from_sparse(inds, amps, o)
println("array_expval = ", expectation(ψ, o))
