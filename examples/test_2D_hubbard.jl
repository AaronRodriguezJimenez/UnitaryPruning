using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators
using SparseArrays

"""
 The following function performs the Jordan-Wirgner mapping for fermionic 
    bilinear terms 
    N - Total number of fermionic modes
    a,b, - indices of the modes to be mapped
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
    D = Lx * Ly # The dimension of the lattice
    #A = zeros(Int, D, D)
    #N = 2 * Lx * Ly
    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    generators = Vector{Pauli{N}}([])
    parameters = Vector{Float64}([])

    for kl in 1:k        
        #- - - Hopping term - - -
        for y in 0:(Ly - 1)
            for x in 0:(Lx - 1)
                i = y * Lx + x + 1  # Index for site i
    
                # Right neighbor
                if x < Lx - 1
                    j = y * Lx + (x + 1) + 1
                    for spin in 0:1
                        a = i + spin * D #mode index for spin
                        b = j + spin * D
                        H_hop += JWmapping(o,i=a,j=b) 
                        H_hop += JWmapping(o,i=b,j=a)
                    end
                end
    
                # Bottom neighbor
                if y < Ly - 1
                    j = (y + 1) * Lx + x + 1
                    for spin in 0:1
                        a = i + spin * D  #mode index for spin
                        b = j + spin * D
                        H_hop += JWmapping(o,i=a,j=b)
                        H_hop += JWmapping(o,i=b,j=a)
                    end
                end
            end
        end
    

        for (pauli, coeff) in H_hop
            push!(generators, Pauli(pauli))
            push!(parameters, -t*coeff)
        end

         # - - - Interaction term - - -
        for site in 1:D
            a_up = site          # spin-up orbital
            a_dn = site + D      # spin-down orbital
            H_u += JWmapping(o, i=a_up, j=a_up) * JWmapping(o, i=a_dn, j=a_dn)
        end
    
        for (pauli, coeff) in H_u
            push!(generators, Pauli(pauli))
            push!(parameters, U*coeff)
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
    D = Lx * Ly # Number of sites
    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()

    linear_index(x, y) = y * Lx + x + 1  # (0-based x, y to 1-based index)

    # Interleaved spin mapping:
    # ↑ spin for site j: index 2j - 1
    # ↓ spin for site j: index 2j
    up(j) = 2*j - 1
    dn(j) = 2*j

    for kl in 1:k
        # Hopping terms
        for y in 0:(Ly - 1)
            for x in 0:(Lx - 1)
                i = linear_index(x, y)

                # Right neighbor
                if x < Lx - 1
                    j = linear_index(x + 1, y)
                    for mode in [(up, up), (dn, dn)]
                        a = mode[1](i)
                        b = mode[2](j)
                        H_hop += JWmapping(o, i=a, j=b)
                        H_hop += JWmapping(o, i=b, j=a)
                    end
                end

                # Bottom neighbor
                if y < Ly - 1
                    j = linear_index(x, y + 1)
                    for mode in [(up, up), (dn, dn)]
                        a = mode[1](i)
                        b = mode[2](j)
                        H_hop += JWmapping(o, i=a, j=b)
                        H_hop += JWmapping(o, i=b, j=a)
                    end
                end
            end
        end

        for (pauli, coeff) in H_hop
            push!(generators, Pauli(pauli))
            push!(parameters, -t * coeff)
        end

        # Interaction terms
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


function run(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1 , w_type = "Majorana", max_weight=1)

    N = 2*Lx*Ly
    ket = Ket(N,0)
    #println("Ket: ", ket)
    o = Pauli(N, Z=[1])

    #Create generators and parameters for the model
    #generators, parameters = hubbard_model_2D(o, Lx=Lx, Ly=Ly, t=t, U=U, k=k)
    generators, parameters = hubbard_model_2D_interleaved(o, Lx=Lx, Ly=Ly, t=t, U=U, k=k)
    
    #Call to bfs bfs_evolution_test based on weight

    #ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)
    ei, nops = UnitaryPruning.bfs_evolution_weight_clip(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    o_mat = Matrix(o)
    m = diag(U'*o_mat*U)
    abs_err = abs(real(m[1])- real(ei) )
    println("Exact :", real(m[1]), " Approx :", real(ei), " Absolute Error: ", abs_err)
    return abs_err
end

function plot_abs_error_vs_weight_pdf(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1, max_weights=0:2:6)
    errors = Float64[]
    weights = Int[]

    errors_pauli = Float64[]
    weights_pauli = Int[]


    for mw in max_weights
        println("Evaluating max_weight = $mw")
        err = run(Lx = Lx, Ly = Ly, t = t, U = U, k=k, w_type="Majorana", max_weight=mw)
        push!(errors, err)
        push!(weights, mw)

        println("Evaluating max_weight = $mw")
        err = run(Lx = Lx, Ly = Ly, t = t, U = U, k=k, w_type="Pauli", max_weight=mw)
        push!(errors_pauli, err)
        push!(weights_pauli, mw)
    end

    # Plot comparison
    plt = plot(
        weights, errors,
        label = "Majorana",
        marker = :circle,
        lw = 2,
    )
    plot!(
        weights_pauli, errors_pauli,
        label = "Pauli",
        marker = :square,
        lw =2
    )

    xlabel!("Max Weight Cutoff")
    ylabel!("Absolute Error")
    title!("2D Hubbard, Lx=$Lx-Ly=$Ly-t=$t-U=$U-k=$k")

    
    filename="2D_Hubbard_test_abs_error_vs_weight_Lx=$Lx-Ly=$Ly-t=$t-U=$U-k=$k.png"
    savefig(plt, filename)
    println("Plot saved as $filename")
end

for k in 1:6
    println("Calculation for k = ", k)
    plot_abs_error_vs_weight_pdf(Lx=2, Ly=2, t=1.0, U=2.0, k=k, max_weights=1:1:8)
end

# Testing stuff
# Compute C^dagger_i term
#a = 3
#b = 1
#ax_term = Pauli(2^(a-1)-1, 2^(a-1), N)
#ay_term = Pauli(2^(a)-1, 2^(a-1), N)
#c_dagg_a = 0.5 * (ax_term - ay_term)
# Compute C_j term
#bx_term = Pauli(2^(b-1)-1, 2^(b-1), N)
#by_term = Pauli(2^(b)-1, 2^(b-1), N)
#c_b = 0.5 * (bx_term + by_term)
# Build C^dagger_i*C_j
#term =  c_dagg_a*c_b 
#result = term + adjoint(term)
#println("Build C^dagger_i*C_j:")
#println(string(result))