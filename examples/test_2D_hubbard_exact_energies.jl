using Distributed
using UnitaryPruning
using Printf
using Random
using LinearAlgebra
using PauliOperators
using SparseArrays

"""
 HERE WE ARE JUST COMPUTING THE EXACT ENERGIES UNDER THE SCHRODINGER
 PICTURE FOR THE LARGER LATTICES
"""

function jw_transform(o::Pauli{N}, site) where N
    z_string = [i for i in 1:site-1]
    # p = PauliSum(N)
    p = Pauli(N, Z = z_string, X = [site]) + im * Pauli(N, Z=z_string, Y=[site])
    return 0.5*p
end

function fermi_hubbard_2D(o::Pauli{N}; t, U, k) where N
    Nsites = Int(N/2)
    L = Int(sqrt(Nsites))

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    t_term = PauliSum(N)
    u_term = PauliSum(N)

    up(j) = 2*j - 1
    dn(j) = 2*j
    
    linear_index(x, y) = (x-1)*L + y
    for ki in 1:k
        t_term = PauliSum(N)
        for x in 1:L
            for y in 1:L
                j = linear_index(x, y)
                if x < L
                    # down coupling
                    i = linear_index(x + 1, y)
                    # α-spin c{i, α}†c{j, α} + h.c.
                    i_a = jw_transform(o, up(j))
                    j_a = jw_transform(o, up(i))
                    t_term += i_a' * j_a
                    t_term += j_a' * i_a
                    # β-spin c{i, β}†c{j, β} + h.c.
                    i_b = jw_transform(o, dn(j))
                    j_b = jw_transform(o, dn(i))
                    t_term += i_b' * j_b
                    t_term += j_b' * i_b
                end
                if y < L
                    # α-spin c{i, α}†c{j, α} + h.c.
                    i = linear_index(x, y + 1)
                    # right side coupling
                    i_a = jw_transform(o, up(j))
                    j_a = jw_transform(o, up(i))
                    t_term += i_a' * j_a
                    t_term += j_a' * i_a
                    # β-spin c{i, β}†c{j, β} + h.c.
                    i_b = jw_transform(o, dn(j))
                    j_b = jw_transform(o, dn(i))
                    t_term += i_b' * j_b
                    t_term += j_b' * i_b
                end
            end
        end
        for (pauli, coeff) in t_term
            push!(generators, Pauli(pauli))
            push!(parameters, -t*coeff)
        end
        u_term = PauliSum(N)
        for j in 1:Nsites
            # interacting term
            i_a = jw_transform(o, up(j))
            i_b = jw_transform(o, dn(j))
            u_term += i_a'*i_a*i_b'*i_b
            # println("Interaction")
            # display(u_term)
        end
        for (pauli, coeff) in u_term
            push!(generators, Pauli(pauli))
            push!(parameters, U*coeff)
        end
    end
    return generators, parameters
end

function fermi_hubbard_2D_block(o::Pauli{N}; t, U, k) where N
    Nsites = Int(N / 2)              # Total number of lattice sites
    L = Int(sqrt(Nsites))           # Lattice size L × L
    D = Nsites                      # For clarity

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    up(j) = j                       # spin-up: sites 1:D
    dn(j) = j + D                   # spin-down: sites D+1:2D

    linear_index(x, y) = (x - 1) * L + y  # 1-based indexing

    for ki in 1:k
        t_term = PauliSum(N)

        for x in 1:L
            for y in 1:L
                j = linear_index(x, y)

                # Right neighbor (y-direction)
                if y < L
                    i = linear_index(x, y + 1)

                    # α-spin hopping: c†_i,α c_j,α + h.c.
                    i_a = jw_transform(o, up(i))
                    j_a = jw_transform(o, up(j))
                    t_term += i_a' * j_a
                    t_term += j_a' * i_a

                    # β-spin hopping
                    i_b = jw_transform(o, dn(i))
                    j_b = jw_transform(o, dn(j))
                    t_term += i_b' * j_b
                    t_term += j_b' * i_b
                end

                # Down neighbor (x-direction)
                if x < L
                    i = linear_index(x + 1, y)

                    # α-spin hopping
                    i_a = jw_transform(o, up(i))
                    j_a = jw_transform(o, up(j))
                    t_term += i_a' * j_a
                    t_term += j_a' * i_a

                    # β-spin hopping
                    i_b = jw_transform(o, dn(i))
                    j_b = jw_transform(o, dn(j))
                    t_term += i_b' * j_b
                    t_term += j_b' * i_b
                end
            end
        end

        # Add hopping terms
        for (pauli, coeff) in t_term
            push!(generators, Pauli(pauli))
            push!(parameters, -t * coeff)
        end

        # Interaction terms
        u_term = PauliSum(N)
        for j in 1:D
            i_a = jw_transform(o, up(j))  # spin-up
            i_b = jw_transform(o, dn(j))  # spin-down
            u_term += i_a' * i_a * i_b' * i_b
        end

        for (pauli, coeff) in u_term
            push!(generators, Pauli(pauli))
            push!(parameters, U * coeff)
        end
    end

    return generators, parameters
end

# 
#- - - Scrodinger time evolution 
#
"""
  Fucntion to return the effect of Operator o applied to the vector V
  which must correspond to some compatible ket    
"""
function apply_pauli_index_phase(p::Pauli{N}, i::Union{Int128, Int64}) where N
    coeff, ketj = p * Ket(N, i)  
    return ketj.v, coeff
end

"""
 This function avoids the use of vector allocation and updates the result of
 the time evolution state as a dictionary.
"""
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


"""
 This function is an update to compute_schrodinger_sparse_evol which seeks to
 avoid the allocation of dictionaries.
"""
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

function run(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1)

    N = 2*Lx*Ly
    ket = Ket(N,0)
    o = Pauli(N, Z=[1])

    #Create generators and parameters for the model
    generators, parameters = fermi_hubbard_2D(o, t=t, U=U, k=k)
    #generators, parameters = fermi_hubbard_2D_block(o, t=t, U=U, k=k)

    # Exact evolution (Schrodinger picture)
    #vector_ket = Vector(ket)
    #println("Vector ket: $ket -> ", vector_ket)
    #U_psi = UnitaryPruning.compute_schrodinger_evol(generators, parameters, vector_ket)
    #expval = U_psi' * UnitaryPruning.matvec(o, 1.00, U_psi)
    
    # Schrodinger sparse-lite version
    #ψ = compute_schrodinger_sparse_evol(generators, parameters, ket)
    #expval = expectation(ψ, o)
    #println("Expval Schr :", expval)

    #Schrodinger sparse-lite version 2
    inds, amps = compute_schrodinger_array_evol(generators, parameters, ket)
    expval =  expectation_from_sparse(inds, amps, o)
    println("Expval Schr :", expval)

    # Exact evolution (Heisenberg picture)
    #U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    #o_mat = Matrix(o)
    #m = diag(U'*o_mat*U)
    #expval = m[1]
    #println("Expval Heis :", expval)

    return 0# expval
end

   
run(Lx=3, Ly=3, t=1.0, U=2.0, k=1)
