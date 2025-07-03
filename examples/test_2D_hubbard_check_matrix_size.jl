using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators
using SparseArrays
using BenchmarkTools

"""
 The following function performs the Jordan-Wirgner mapping for fermionic 
    bilinear terms 
    N - Total number of fermionic modes
    a,b, - indices of the modes to be mapped
"""
function JWmapping(o::Pauli{N}; i::Int, j::Int) where N
    # Compute C^dagger_i term
    println("JW mapping , $N, i: , $i, j:, $j")
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
function hubbard_model_2D(o::Pauli{N}; Lx::Int, Ly::Int, t::Float64, U::Float64, k::Int) where N
    D = Lx * Ly  # Number of lattice sites
    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    # Linear index function (1-based)
    linear_index(x, y) = (y - 1) * Lx + x  # x in 1:Lx, y in 1:Ly

    for kl in 1:k
        H_hop = PauliSum(N)
        H_u = PauliSum(N)

        # Loop through all coordinates in 1-based indexing
        for y in 1:Ly
            for x in 1:Lx
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
function hubbard_model_2D_interleaved(o::Pauli{N}; Lx::Int, Ly::Int, t::Float64, U::Float64, k::Int) where N
    D = Lx * Ly  # number of lattice sites

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    # 1-based linear index
    linear_index(x, y) = (y - 1) * Lx + x  # returns 1 to D

    # Spin-orbital index: ↑ = 2j - 1, ↓ = 2j
    up(j) = 2*j - 1
    dn(j) = 2*j

    for kl in 1:k
        H_hop = PauliSum(N)
        H_u = PauliSum(N)

        # Loop over 1-based coordinates
        for y in 1:Ly
            for x in 1:Lx
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
            push!(generators, Pauli(pauli))
            push!(parameters, U * coeff)
        end
    end

    return generators, parameters
end

# Chinmay version
function jw_transform(o::Pauli{N}, site) where N
    println(site)
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
    D = Int(N / 2)              # Total number of lattice sites
    L = Int(sqrt(Nsites))           # Lattice size L × L

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

function run_profiled(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1 , w_type = "Majorana", max_weight=1)
    N = 2*Lx*Ly
    ket = Ket(N, 0)
    o = Pauli(N, Z=[1])
    
    mem_model = @allocated generators, parameters = hubbard_model_2D(o, Lx=Lx, Ly=Ly, t=t, U=U, k=k)
    #mem_model = @allocated generators, parameters = hubbard_model_2D_interleaved(o, Lx=Lx, Ly=Ly, t=t, U=U, k=k)
    #mem_model = @allocated generators, parameters = fermi_hubbard_2D(o, t=t, U=U, k=k)
    # Method memory
    mem_bfs = @allocated ei, nops = UnitaryPruning.bfs_evolution_weight_clip(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact matrix approach
    mem_U = @allocated U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    #mem_U = @allocated U = UnitaryPruning.build_time_evolution_matrix_fast(N, generators, parameters)
    #mem_U = @allocated U = UnitaryPruning.build_time_evolution_sparse(N, generators, parameters)
    #o_mat = Matrix(o) 
    #memdiag = @allocated m = diag(U'*o_mat*U) #Memory for diagonalization

    println("Memory usage:")
    println("  Model Memory:  $(mem_model*1e-06) MB")
    println("  BFS Memory: $(mem_bfs*1e-06) MB")
    println("  Exact U:  $(mem_U*1e-06) MB")
    #println("  Diagonalization:  $(memdiag*1e-06) MB")
    
    
    # Return memory measurements in Mb
    return mem_model*1e-06, mem_bfs*1e-06, 
    
    mem_U*1e-06
end


# Testing stuff
#run(Lx = 3, Ly = 3, t = 1.0, U = 2.0, k=5 , w_type = "Majorana", max_weight=8)
mm = Float64[]
mbfs = Float64[]
mu = Float64[]
k_vals = Float64[]

for k in 1:1
    max_w = 1
    println("- - - k = $k - - -")
    mem_model, mem_bfs, mem_U = run_profiled(Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=k , w_type = "Majorana", max_weight=max_w)
    push!(mm, mem_model)
    push!(mbfs, mem_bfs)
    push!(mu, mem_U)
    push!(k_vals, k)
end

plt = plot(k_vals, mm,
        lw=2, marker=:circle,
        legend = true,
        label = "Model"
    )
    plot!(k_vals, mbfs,
    lw =2, marker =:circle,
    legend=true,
    label= "BFS"
    )
    plot!(k_vals, mu,
    lw =2, marker =:circle,
    legend=true,
    label= "Exact U"
    )

    xlabel!("k value")
ylabel!("Memory allocated (MB)")
title!(" 2x3 Lattice")

filename="2D_test_chinmay_Hubbard_MEMORY_usage.pdf"
savefig(plt, filename)

