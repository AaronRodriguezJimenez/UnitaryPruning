using LinearAlgebra
using SparseArrays

function get_unitary_sequence_1D(o::Pauli{N}; α=.01, k=10) where N
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    # print("alpha", α, "\n")
    # Loop over trotter steps
    for ki in 1:k
        ## ZZ layer
        # e^{i π/2 P2} e^{i π P1 /2}|ψ>
        for i in 1:N-1
            pi = Pauli(N, Z=[i, i + 1])
            push!(generators, pi)
            push!(parameters, π/2)
        end
        #pbc 
        pi = Pauli(N, Z=[N, 1])
        push!(generators, pi)
        push!(parameters, π/2)

        ## X layer
        # e^{i αn (-X) / 2}
        for i in 1:N
            pi = Pauli(N, X=[i])
            pi = Pauli{N}((pi.θ + 2)%4, pi.pauli) # this accounts for the fact that the papers have -X and positive ZZ
            push!(generators, pi)
            push!(parameters, α)
        end
    end

    return generators, parameters
end


function get_unitary_sequence_2D(o::Pauli{N}; α=.01, k=10) where N


    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    bridges = [[1,5,7], [3,6,9]]
    sequences = [[1,3], [7,9]]

    for ki in 1:k
        # e^{i π/4 P2} e^{i π P1 /2}|ψ>
        ## ZZ layer
        for qubit in sequences
            for i in qubit[1]:qubit[2]
                pi = Pauli(N, Z=[i, i + 1])
                push!(generators, pi)
                push!(parameters, π/2)
            end
        end
        #bridges
        for link in bridges
            pi = Pauli(N, Z=[link[1], link[2]])
            push!(generators, pi)
            push!(parameters, π/2)
            
            pi = Pauli(N, Z=[link[2], link[3]])
            push!(generators, pi)
            push!(parameters, π/2)
        end
        ## X layer
        # e^{i αn Pn / 2}
        for i in 1:N
            pi = Pauli(N, X=[i])
            pi = Pauli{N}((pi.θ + 2)%4, pi.pauli) # this accounts for the fact that the papers have -X and positive ZZ
            push!(generators, pi)
            push!(parameters, α)
        end
    end
    return generators, parameters
end



function build_time_evolution_matrix(generators::Union{Vector{Pauli{N}}, Vector{PauliBasis{N}}}, angles::Vector) where N
    U = Matrix(Pauli(N))
    nt = length(generators)
    length(angles) == nt || throw(DimensionMismatch)
    for t in 1:nt
        α = angles[t]
        U = cos(α/2) .* U .- 1im*sin(α/2) .* U * Matrix(generators[t])

    end

    return U 
end

function build_time_evolution_matrix_fast!(U::AbstractMatrix{ComplexF64}, W::AbstractMatrix{ComplexF64},
    generators::Vector{Pauli{N}},angles::Vector{<:Real}) where N

    nt = length(generators)
    length(angles) == nt || throw(DimensionMismatch())

    fill!(U, 0.0)
    @inbounds for i in axes(U, 1)
        U[i, i] = 1.0
    end

    for t in 1:nt
        α = angles[t]
        Pmat = Matrix(generators[t])

        # W = U * Pmat (in-place)
        mul!(W, U, Pmat)

        # U = cos(α/2)*U - i*sin(α/2)*W (in-place)
        c, s = cos(α / 2), sin(α / 2)
        @inbounds @simd for i in eachindex(U)
            U[i] = c * U[i] - 1im * s * W[i]
        end
    end
    
    return U
end

function build_time_evolution_matrix_fast(N, generators, angles)
    dim = 2^N
    U = Matrix{ComplexF64}(undef, dim, dim)
    W = similar(U)
    return build_time_evolution_matrix_fast!(U, W, generators, angles)
end

# 
#- - - Scrodinger time evolution 
#
"""
  matvec fucntion return the effect of Operator o applied to the vector V
  which must correspond to some compatible ket    
"""
function apply_pauli_index_phase(p::Pauli{N}, i::Int) where N
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

function compute_schrodinger_evol(generators, parameters, ref_ket)
    nt = length(generators)
    length(parameters) == nt || throw(DimensionMismatch)

    U_psi = copy(ref_ket)

    for t in 1:nt
        α = parameters[t]
        Pψ = matvec(generators[t], 1.0, U_psi)
        U_psi = cos(α/2) .* U_psi - 1im * sin(α/2) .* Pψ
        U_psi /= norm(U_psi)  # normalize at each step
    end

    return U_psi
end

# # # # # # # # # # # # # # # #
#- Models
# # # # # # # # # # # # # # # #
"""
 1D linear chain version of the Heisenberg model
"""
function heisenberg_1D(o::Pauli{N}; Jx, Jy, Jz, k) where N 
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])
    # Loop over sites
    for ki in 1:k 
        for i in 1:N-1
            push!(generators, Pauli(N, X=[i, i + 1]))
            push!(parameters, Jx)
            push!(generators, Pauli(N, Y=[i, i + 1]))
            push!(parameters, Jy)
            push!(generators, Pauli(N, Z=[i, i + 1]))
            push!(parameters, Jz)
        end
    end

    return generators, parameters
end

"""
 2D Heisenberg model with periodic boundary conditions.
 Similar as the previous function, however, here we wrap around using 
 modulo arithmetic
 x_right wraps L -> 1
 y_down wraps L -> 1
 This means: site (L,y) connects to (1,y) & site(x,L) connects to (x,1.
"""
function heisenberg_2D(o::Pauli{N}; Jx, Jy, Jz, k) where N
    L = Int(sqrt(N))
    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()
    index(i, j) = (mod1(i, L) - 1) * L + mod1(j, L)
    for ki in 1:k
        for i in 1:L, j in 1:L
            a = index(i, j)
            # Right neighbor (periodic in j)
            b = index(i, j + 1)
            push!(generators, Pauli(N, X=[a, b])); push!(parameters, Jx)
            push!(generators, Pauli(N, Y=[a, b])); push!(parameters, Jy)
            push!(generators, Pauli(N, Z=[a, b])); push!(parameters, Jz)
            # Down neighbor (periodic in i)
            b = index(i + 1, j)
            push!(generators, Pauli(N, X=[a, b])); push!(parameters, Jx)
            push!(generators, Pauli(N, Y=[a, b])); push!(parameters, Jy)
            push!(generators, Pauli(N, Z=[a, b])); push!(parameters, Jz)
        end
    end
    return generators, parameters
end

"""
 The following function performs the Jordan-Wirgner mapping for fermionic 
    bilinear terms 
    N - Total number of fermionic modes
    a,b, - indices of the modes to be mapped
    returns term = c^dagger_a * c_b
"""
function JWmapping(o::Pauli{N}; i::Int, j::Int) where N
    # Compute C^dagger_i term
    ax_term = Pauli(2^(i-1)-1, 2^(i-1), N)
    ay_term = Pauli(2^(i)-1, 2^(i-1), N)
    c_dagg_a = 0.5 * (ax_term - ay_term)

    # Compute C_j term
    bx_term = Pauli(2^(j-1)-1, 2^(j-1), N)
    by_term = Pauli(2^(j)-1, 2^(j-1), N)
    c_b = 0.5 * (bx_term + by_term)

    # Build C^dagger_i*C_j
    term =  c_dagg_a*c_b

    return term
end

"""
 1D Fermi-Hubbard model
Generate a 1D Fermi-Hubbard Hamiltonian (open boundaries, no PBC)
using JW mapping into Pauli operators.

Arguments:
- o::Pauli{N} : reference Pauli object
- L::Int       : number of sites
- t::Float64   : hopping amplitude
- U::Float64   : on-site interaction
- k::Int       : number of Trotter steps (can be used later for evolution)

Returns:
- generators::Vector{Pauli{N}}
- parameters::Vector{Float64}
"""
function hubbard_model_1D(o::Pauli{N}; L::Int64, t::Float64, U::Float64, k::Int64) where N
    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    for ki in 1:k
        # Hopping terms
        for i in 1:L-1
            # spin-up
            a = 2*i - 1
            b = 2*(i + 1) - 1
            hopping_term = JWmapping(o, i=a, j=b) + JWmapping(o, i=b, j=a)
            for (pauli, coeff) in hopping_term
                push!(generators, Pauli(pauli))
                push!(parameters, -t * coeff)
            end

            # spin-down
            a = 2*i
            b = 2*(i + 1)
            hopping_term = JWmapping(o, i=a, j=b) + JWmapping(o, i=b, j=a)
            for (pauli, coeff) in hopping_term
                if coeff == 0.0
                    continue
                end
                push!(generators, Pauli(pauli))
                push!(parameters, -t * coeff)
            end
        end

        # On-site interaction terms
        for i in 1:L
            a_up = 2*i - 1   # spin-up orbital index
            a_dn = 2*i       # spin-down orbital index
            interaction_term = JWmapping(o, i=a_up, j=a_up) * JWmapping(o, i=a_dn, j=a_dn)
            for (pauli, coeff) in interaction_term
                if coeff == 0.0
                    continue
                end
                push!(generators, Pauli(pauli))
                push!(parameters, U * coeff)
            end
        end
    end

    return generators, parameters   
    
end


"""
 2D Fermi-Hubbard model

 Essentially, modeling the hopping term can be reduced to compute the adjacent
 matrix of a graph representing a 2D-lattice. Being the site lattice:
  (1)- - -(2)- - -(3)
  (4)- - -(5)- - -(6)
  (7)- - -(8)- - -(9)
 the first neighbor sites can be encoded by the adjacency matrix:
     _ _ _ _ _ _ _ _ _ _ _ _ _ _
     | 0  1  0  1  0  0  0  0  0
     | 1  0  1  0  1  0  0  0  0
     | 0  1  0  0  0  1  0  0  0
     | 1  0  0  0  1  0  1  0  0
     | 0  1  0  1  0  1  0  1  0
     | 0  0  1  0  1  0  0  0  1
     | 0  0  0  1  0  0  0  1  0
     | 0  0  0  0  1  0  1  0  1
     | 0  0  0  0  0  1  0  1  0

  Encodes the terms c^dagger_i c_j.
  Similarly, the charge-charge term, can be represented by the diagonal matrix:  
  _ _ _ _ _ _ _ _ _ _ _ _ _ _
     | U  0  0  0  0  0  0  0  0
     | 0  U  0  0  0  0  0  0  0
     | 0  0  U  0  0  0  0  0  0
     | 0  0  0  U  0  0  0  0  0
     | 0  0  0  0  U  0  0  0  0
     | 0  0  0  0  0  U  0  0  0
     | 0  0  0  0  0  0  U  0  0
     | 0  0  0  0  0  0  0  U  0
     | 0  0  0  0  0  0  0  0  U

  with each term in the diagonal encoding the terms n_{j,up}*n_{j,down}, i.e. the number
  operators n_j = c^dagg_j*c_j. For which the JW transform reads as:
  n_j = c^dagg_j*c_j = 1/2 (1 - Z_j)
  i is index a, and j is b in the following mapping
"""
function hubbard_model_2D_block(o::Pauli{N}; Lx::Int, Ly::Int, t::Float64, U::Float64, k::Int) where N
    
    D = Lx * Ly  # Number of lattice sites

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    # Linear index function (1-based)
    linear_index(x, y) = (y - 1) * Lx + x  # x in 1:Lx, y in 1:Ly

    for kl in 1:k

        H_hop = PauliSum(N)
        H_u = PauliSum(N)

        # Loop through all coordinates in 1-based indexing
        for x in 1:Lx
            for y in 1:Ly
                i = linear_index(x, y)

                # Right neighbor (x+1)
                if x < Lx
                    j = linear_index(x + 1, y)
                    for spin in 0:1
                        a = i + spin * D  # block spin: spin-up first [1:D], spin-down [D+1:2D]
                        b = j + spin * D
                        H_hop += JWmapping(o, i=a, j=b)
                        H_hop += JWmapping(o, i=b, j=a)
                    end
                end

                # Bottom neighbor (y+1)
                if y < Ly
                    j = linear_index(x, y + 1)
                    for spin in 0:1
                        a = i + spin * D
                        b = j + spin * D
                        H_hop += JWmapping(o, i=a, j=b)
                        H_hop += JWmapping(o, i=b, j=a)
                    end
                end
            end
        end

        # Add hopping terms
        for (pauli, coeff) in H_hop
            if coeff == 0.0
                continue
            end
            push!(generators, Pauli(pauli))
            push!(parameters, -t * coeff)
        end

        # On-site interaction term
        for site in 1:D
            a_up = site           # spin-up orbital
            a_dn = site + D       # spin-down orbital
            H_u += JWmapping(o, i=a_up, j=a_up) * JWmapping(o, i=a_dn, j=a_dn)
        end

        for (pauli, coeff) in H_u
            if coeff == 0.0
                continue
            end
            push!(generators, Pauli(pauli))
            push!(parameters, U * coeff)
        end
    end

    return generators, parameters
end

"""
 Same logic, interleaved version:
 Constructs the Jordan-Wigner transformed 2D Fermi-Hubbard model on an Lx × Ly square lattice
with interleaved spin ordering.

The Hamiltonian is:

    H = -t ∑⟨i,j⟩,σ (c†_{i,σ} c_{j,σ} + h.c.) + U ∑_j n_{j,↑} n_{j,↓}

where:
- `t` is the nearest-neighbor hopping amplitude,
- `U` is the on-site interaction strength,
- `k` is the number of Trotter steps (for repeated operator terms),
- `o` is a basis Pauli operator (used to construct Pauli terms),
- `N = 2 * Lx * Ly` is the total number of fermionic modes (spinful sites).

**Interleaved spin ordering**:
- Site `j`'s spin-↑ orbital is at index `2j - 1`
- Site `j`'s spin-↓ orbital is at index `2j`

Only nearest-neighbor interactions along the x and y directions are included.
Open boundary conditions (OBC) are used by default.
"""
function isidentity(p::Pauli{N}) where {N}
    pstring = string(p)
    return all(c -> c == 'I', pstring)
end

function hubbard_model_2D_interleaved(o::Pauli{N}; Lx::Int64, Ly::Int64, t::Float64, U::Float64, k::Int64) where N
    D = Lx * Ly  # number of lattice sites

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    # 1-based linear index
    linear_index(x, y) = (y - 1) * Lx + x  # returns 1 to D

    # Spin-orbital index: ↑ = 2j - 1, ↓ = 2j
    up(j) = 2*j - 1
    dn(j) = 2*j

    for kl in 1:k

        H_hop = PauliSum(N)
        H_u = PauliSum(N)

        # Loop over 1-based coordinates
        for x in 1:Lx
            for y in 1:Ly
                i = linear_index(x, y)

                # Right neighbor (x+1)
                if x < Lx
                    j = linear_index(x + 1, y)
                    for (a_fn, b_fn) in [(up, up), (dn, dn)]
                        a = a_fn(i)
                        b = b_fn(j)
                        H_hop += JWmapping(o, i=a, j=b)
                        H_hop += JWmapping(o, i=b, j=a)
                    end
                end

                # Bottom neighbor (y+1)
                if y < Ly
                    j = linear_index(x, y + 1)
                    for (a_fn, b_fn) in [(up, up), (dn, dn)]
                        a = a_fn(i)
                        b = b_fn(j)
                        H_hop += JWmapping(o, i=a, j=b)
                        H_hop += JWmapping(o, i=b, j=a)
                    end
                end
            end
        end

        # Add hopping terms
        for (pauli, coeff) in H_hop
            if coeff == 0.0
                continue
            end
            push!(generators, Pauli(pauli))
            push!(parameters, -t * coeff)
        end

        # On-site interaction terms
        for site in 1:D
            a_up = up(site)
            a_dn = dn(site)
            H_u += JWmapping(o, i=a_up, j=a_up) * JWmapping(o, i=a_dn, j=a_dn)
        end

        for (pauli, coeff) in H_u
            if coeff == 0.0
                continue
            end
            #println("C_u:", coeff)
            push!(generators, Pauli(pauli))
            push!(parameters, U * coeff)
        end
    end

    return generators, parameters
end

# # # # # # # # # # # # # # #
#- - - Hubbard model Chinmay version - - -

function jw_transform(o::Pauli{N}, site) where N
    z_string = [i for i in 1:site-1]
    # p = PauliSum(N)
    p = Pauli(N, Z = z_string, X = [site]) + 1im * Pauli(N, Z=z_string, Y=[site])
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
            if coeff == 0.0
                continue
            end
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
            if coeff == 0.0
                continue
            end
            push!(generators, Pauli(pauli))
            push!(parameters, U*coeff)
        end
    end
    return generators, parameters#, -t*t_term + U*u_term
end