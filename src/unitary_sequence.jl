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
#
# - - - 2D Transverse Field Ising Model (TFIM)
#
function get_unitary_sequence_2D_test(L; α=.01, k=10)
    N = L^2
    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])

    # Helper: convert (row, col) -> site index
    site(i, j) = (i - 1) * L + j

    # Loop over trotter steps
    for ki in 1:k
        ## ZZ layer
        # horizontal neighbors
        for i in 1:L
            for j in 1:(L-1)
                pi = Pauli(N, Z=[site(i,j), site(i,j+1)])
                push!(generators, pi)
                push!(parameters, π/2)
            end
        end

        # vertical neighbors
        for i in 1:(L-1)
            for j in 1:L
                pi = Pauli(N, Z=[site(i,j), site(i+1,j)])
                push!(generators, pi)
                push!(parameters, π/2)
            end
        end

        ## X layer
        for i in 1:N
            pi = Pauli(N, X=[i])
            pi = Pauli{N}(-pi.s, pi.z, pi.x) # flip sign for -X convention
            push!(generators, pi)
            push!(parameters, α)
        end
    end

    return generators, parameters
end

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
#function JWmapping(o::Pauli{N}; i::Int, j::Int) where N
#    # Compute C^dagger_i term
#    ax_term = Pauli(2^(i-1)-1, 2^(i-1), N)
#    ay_term = Pauli(2^(i)-1, 2^(i-1), N)
#    c_dagg_a = 0.5 * (ax_term - ay_term)
#
#    # Compute C_j term
#    bx_term = Pauli(2^(j-1)-1, 2^(j-1), N)
#    by_term = Pauli(2^(j)-1, 2^(j-1), N)
#    c_b = 0.5 * (bx_term + by_term)
#
#    # Build C^dagger_i*C_j
#    term =  c_dagg_a*c_b
#
#    return term
#end

# --- helpers for JWmapping ---
@inline ubit(i::Int) = UInt128(1) << (i-1)                         # bit at site i (1-based)
@inline umask_lt(i::Int) = i==1 ? UInt128(0) : (ubit(i) - UInt128(1))  # bits < i
@inline umask_le(i::Int) = umask_lt(i) | ubit(i)                      # bits ≤ i

# --- JW mapping (original real ± form; your coeff() supplies +i on ZX sites) ---
function JWmapping(o::Pauli{N}; i::Int, j::Int) where N
    1 <= i <= N || throw(DimensionMismatch("site i=$i out of 1:$N"))
    1 <= j <= N || throw(DimensionMismatch("site j=$j out of 1:$N"))

    # X pieces with Z-strings
    ax = Pauli{N}(1, reinterpret(Int128, umask_lt(i)), reinterpret(Int128, ubit(i)))  # Z^{<i} X_i
    bx = Pauli{N}(1, reinterpret(Int128, umask_lt(j)), reinterpret(Int128, ubit(j)))  # Z^{<j} X_j

    # "Y" pieces = Z^{≤i} X_i, Z^{≤j} X_j  (no explicit im; your coeff() turns ZX into iY)
    ay = Pauli{N}(1, reinterpret(Int128, umask_le(i)), reinterpret(Int128, ubit(i)))
    by = Pauli{N}(1, reinterpret(Int128, umask_le(j)), reinterpret(Int128, ubit(j)))

    # c†_i = (X_i - Y_i)/2,  c_j = (X_j + Y_j)/2   in your convention
    c_dagg_i = 0.5 * (ax - ay)
    c_j      = 0.5 * (bx + by)

    return c_dagg_i * c_j
end

"""
    combine_clip!(generators::Vector{Pauli{N}}, parameters::Vector{Float64};
                  atol=1e-12) where N

Absorb Pauli phases, combine identical strings, drop small terms (<atol),
and enforce real coefficients.
"""
function combine_clip!(generators::Vector{Pauli{N}},
                       parameters::Vector{Float64};
                       atol::Real=1e-12) where N
    @assert length(generators) == length(parameters)

    # accumulate by (z,x), absorbing phase via coeff(p)
    acc = Dict{Tuple{Int128,Int128}, ComplexF64}()
    for (p, a) in zip(generators, parameters)
        acc[(p.z, p.x)] = get(acc, (p.z, p.x), 0 + 0im) + a * coeff(p)
    end

    empty!(generators); empty!(parameters)

    for ((z,x), c) in acc
        # drop tiny
        if isapprox(c, 0; atol=atol); continue; end

        # enforce real
        if !isapprox(imag(c), 0; atol=atol)
            error("Non-real coefficient $c for Pauli(z=$z,x=$x). Check JWmapping/Hermitian pairing.")
        end

        # push canonical Pauli with unit scalar; keep only the real part
        push!(generators, Pauli{N}(1, z, x))
        push!(parameters, real(c))
    end
    return generators, parameters
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

    combine_clip!(generators, parameters)  # clean model

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

    combine_clip!(generators, parameters)  # clean model

    return generators, parameters
end


# # # # # # # # # # # # # # #
#- - - Hubbard model Chinmay version (improved) - - -
# Helpers: build creation operator (c^†) as Pauli/PauliSum and build bilinear c_i^† c_j + h.c.
function creation_pauli(o::Pauli{N}, mode::Int; reverse_ordering::Bool=false) where N
    # mode is the spin-orbital index in [1, 2*Nsites]
    lib = reverse_ordering ? (N - mode + 1) : mode
    z_string = lib > 1 ? collect(1:lib-1) : Int[]
    return 0.5 * (Pauli(N, Z = z_string, X = [lib]) + 1im * Pauli(N, Z = z_string, Y = [lib]))
end

function fermionic_bilinear_pauli(o::Pauli{N}, m::Int, n::Int; reverse_ordering::Bool=false) where N
    # returns PauliSum representing c_m^† c_n + c_n^† c_m
    cd_m = creation_pauli(o, m; reverse_ordering = reverse_ordering)
    cd_n = creation_pauli(o, n; reverse_ordering = reverse_ordering)
    return cd_m * PauliOperators.adjoint(cd_n) + cd_n * PauliOperators.adjoint(cd_m)
end

"""
    fermi_hubbard_2D_pauli(o::Pauli{N}; Lx, Ly, t, U, k, reverse_ordering=false)

Construct generators and parameters for the 2D spinful Hubbard model on Lx×Ly
(physical sites). Each physical site has two spin-orbitals (up, down), so
total qubits N must equal 2 * Lx * Ly.

Returns (generators::Vector{Pauli{N}}, parameters::Vector{Float64}).
"""
function fermi_hubbard_2D_pauli(o::Pauli{N}; Lx::Int, Ly::Int, t::Float64=1.0, U::Float64=2.0, k::Int=1) where N
    Nsites = Lx * Ly
    reverse_ordering = false
    if 2 * Nsites != N
        throw(ArgumentError("Total qubits N must equal 2 * Lx * Ly. Got N=$N, Lx*Ly=$Nsites"))
    end

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    up(j) = 2*j - 1
    dn(j) = 2*j
    linear_index(x,y) = (x - 1) * Ly + y   # x in 1:Lx, y in 1:Ly

    # small tolerance for dropping tiny coeffs
    eps_coeff = 1e-12

    for _ in 1:k
        # HOPPING: loop nearest-neighbour pairs once, add c_i^† c_j + c_j^† c_i (both spins)
        for x in 1:Lx, y in 1:Ly
            jsite = linear_index(x, y)

            # neighbor +x (right in x)
            if x < Lx
                isite = linear_index(x + 1, y)
                for spin in (up, dn)
                    m = spin(jsite)   # mode index for j
                    n = spin(isite)   # mode index for i
                    term = fermionic_bilinear_pauli(o, m, n; reverse_ordering = reverse_ordering)
                    for (pauli, coeff) in term
                        if abs(coeff) < eps_coeff
                            continue
                        end
                        if abs(imag(coeff)) > 1e-12
                            error("Non-real coefficient encountered in hopping term: $coeff")
                        end
                        push!(generators, Pauli(pauli))
                        push!(parameters, -t * real(coeff))
                    end
                end
            end

            # neighbor +y (right in y)
            if y < Ly
                isite = linear_index(x, y + 1)
                for spin in (up, dn)
                    m = spin(jsite)
                    n = spin(isite)
                    term = fermionic_bilinear_pauli(o, m, n; reverse_ordering = reverse_ordering)
                    for (pauli, coeff) in term
                        if abs(coeff) < eps_coeff
                            continue
                        end
                        if abs(imag(coeff)) > 1e-12
                            error("Non-real coefficient encountered in hopping term: $coeff")
                        end
                        push!(generators, Pauli(pauli))
                        push!(parameters, -t * real(coeff))
                    end
                end
            end
        end

        # ONSITE U term: n_up * n_down on each site
        for site in 1:Nsites
            m_up = up(site)
            m_dn = dn(site)
            cd_up = creation_pauli(o, m_up; reverse_ordering = reverse_ordering)
            cd_dn = creation_pauli(o, m_dn; reverse_ordering = reverse_ordering)
            n_up = cd_up * PauliOperators.adjoint(cd_up)    # c^† c
            n_dn = cd_dn * PauliOperators.adjoint(cd_dn)
            term = n_up * n_dn
            for (pauli, coeff) in term
                if abs(coeff) < eps_coeff
                    continue
                end
                if abs(imag(coeff)) > 1e-12
                    error("Non-real coefficient encountered in U term: $coeff")
                end
                push!(generators, Pauli(pauli))
                push!(parameters, U * real(coeff))
            end
        end
    end

    combine_clip!(generators, parameters)  # clean model

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
function hubbard_model_2D_interleaved(o::Pauli{N}; Lx::Int64, Ly::Int64, t::Float64, U::Float64, k::Int64) where N
    Nsites = Lx * Ly
    reverse_ordering = false
    if 2 * Nsites != N
        throw(ArgumentError("Total qubits N must equal 2 * Lx * Ly. Got N=$N, Lx*Ly=$Nsites"))
    end

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    up(j) = 2*j - 1
    dn(j) = 2*j
    linear_index(x,y) = (x - 1) * Ly + y   # x in 1:Lx, y in 1:Ly

    # small tolerance for dropping tiny coeffs
    eps_coeff = 1e-12

    for _ in 1:k
        # HOPPING: loop nearest-neighbour pairs once, add c_i^† c_j + c_j^† c_i (both spins)
        for x in 1:Lx, y in 1:Ly
            jsite = linear_index(x, y)

            # neighbor +x (right in x)
            if x < Lx
                isite = linear_index(x + 1, y)
                for spin in (up, dn)
                    m = spin(jsite)   # mode index for j
                    n = spin(isite)   # mode index for i
                    term = JWmapping(o, i=n, j=m) + JWmapping(o, i=m, j=n)
                    for (pauli, coeff) in term
                        if abs(coeff) < eps_coeff
                            continue
                        end
                        if abs(imag(coeff)) > 1e-12
                            error("Non-real coefficient encountered in hopping term: $coeff")
                        end
                        push!(generators, Pauli(pauli))
                        push!(parameters, -t * real(coeff))
                    end
                end
            end

            # neighbor +y (right in y)
            if y < Ly
                isite = linear_index(x, y + 1)
                for spin in (up, dn)
                    m = spin(jsite)
                    n = spin(isite)
                    term = JWmapping(o, i=n, j=m) + JWmapping(o, i=m, j=n)
                    for (pauli, coeff) in term
                        if abs(coeff) < eps_coeff
                            continue
                        end
                        if abs(imag(coeff)) > 1e-12
                            error("Non-real coefficient encountered in hopping term: $coeff")
                        end
                        push!(generators, Pauli(pauli))
                        push!(parameters, -t * real(coeff))
                    end
                end
            end
        end

        # ONSITE U term: n_up * n_down on each site
        for site in 1:Nsites
            m_up = up(site)
            m_dn = dn(site)
            term = JWmapping(o, i=m_up, j=m_up) * JWmapping(o, i=m_dn, j=m_dn)
            for (pauli, coeff) in term
                if abs(coeff) < eps_coeff
                    continue
                end
                if abs(imag(coeff)) > 1e-12
                    error("Non-real coefficient encountered in U term: $coeff")
                end
                push!(generators, Pauli(pauli))
                push!(parameters, U * real(coeff))
            end
        end
    end

    combine_clip!(generators, parameters)  # clean model

    return generators, parameters
end
