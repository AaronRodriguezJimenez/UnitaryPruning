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


function run(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1 , w_type = "Majorana", max_weight=1)

    N = 2*Lx*Ly
    #println(N)
    #ket = Ket(N,0) #for interleaved
    ket = Ket(N,0)
    #println("Ket: ", ket)
    #o = Pauli(N, Z=[8]) # for block
    #display(o)
    o = Pauli(N, Z=[1]) # for interleaved 

    #Create generators and parameters for the model
    #generators, parameters = hubbard_model_2D(o, Lx=Lx, Ly=Ly, t=t, U=U, k=k)
    generators, parameters = UnitaryPruning.hubbard_model_2D_interleaved(o, Lx=Lx, Ly=Ly, t=t, U=U, k=k)
    
    #Call to bfs bfs_evolution_test based on weight

    ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)
    #ei, nops = UnitaryPruning.bfs_evolution_weight_clip(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    o_mat = Matrix(o)
    m = diag(U'*o_mat*U)
    abs_err = abs(real(m[1])- real(ei) )
    println("Exact :", real(m[1]), " Approx :", real(ei), " Absolute Error: ", abs_err)
    #return real(ei),real(ei),real(ei) #Activate for expectation values analysis
    return real(m[1]), real(ei), abs_err #Activate for error analysis
end

function plot_abs_error_vs_weight_pdf(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1, max_weights=0:2:6)
    errors = Float64[]
    weights = Int[]

    errors_pauli = Float64[]
    weights_pauli = Int[]


    for mw in max_weights
        println("Evaluating max_weight = $mw")
        _, _, err = run(Lx = Lx, Ly = Ly, t = t, U = U, k=k, w_type="Majorana", max_weight=mw)
        push!(errors, err)
        push!(weights, mw)

        println("Evaluating max_weight = $mw")
        _,_, err = run(Lx = Lx, Ly = Ly, t = t, U = U, k=k, w_type="Pauli", max_weight=mw)
        push!(errors_pauli, err)
        push!(weights_pauli, mw)
    end

    # Plot comparison
    plt = plot(
        weights_pauli, errors_pauli,
        label = "Pauli",
        marker = :circle,
        lw = 2,
    )
    plot!(
        weights, errors,
        label = "Majorana",
        marker = :circle,
        lw =2
    )

    xlabel!("Max Weight Cutoff")
    ylabel!("Absolute Error")
    title!("L=$Lx, t=-$t, U=$U, k=$k")
    #title!("2D Hubbard, Lx=$Lx-Ly=$Ly-t=$t-U=$U-k=$k")

    
    #filename="2D_Hubbard_test_abs_error_vs_weight_Lx=$Lx-Ly=$Ly-t=$t-U=$U-k=$k.pdf"
    filename="2D_Hubbard_abs_error_L=$Lx-t=$t-U=$U-k=$k.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end

function properties_table(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k = 1, max_weights = 0:2:6)
    results_majo = []
    results_pauli = []

    for mw in max_weights
        println("Evaluating Majorana, max_weight = $mw")
        bench = @benchmark run(Lx = $Lx, Ly = $Ly, t = $t, U = $U, k = $k, w_type = "Majorana", max_weight = $mw)
        output = run(Lx = Lx, Ly = Ly, t = t, U = U, k = k, w_type = "Majorana", max_weight = mw)
        push!(results_majo, (k, output[1], output[2], output[3], median(bench.times) / 1e9))  # convert ns to s

        println("Evaluating Pauli, max_weight = $mw")
        bench = @benchmark run(Lx = $Lx, Ly = $Ly, t = $t, U = $U, k = $k, w_type = "Pauli", max_weight = $mw)
        output = run(Lx = Lx, Ly = Ly, t = t, U = U, k = k, w_type = "Pauli", max_weight = mw)
        push!(results_pauli, (k, output[1], output[2], output[3], median(bench.times) / 1e9))
    end

    # Save to text files
    function save_results(filename, results)
        open(filename, "w") do io
            @printf(io, "%-4s | %-12s | %-12s | %-10s | %-10s\n", "k", "E_exact", "E_circ", "Error", "Run time")
            @printf(io, "%s\n", "-"^60)
            for (k, e_exact, e_circ, err, t_run) in results
                @printf(io, "%-4d | %-12.6f | %-12.6f | %-10.2e | %-10.4f\n", k, e_exact, e_circ, err, t_run)
            end
        end
    end

    filename_majo = @sprintf("results_majorana_U%.1f_k%d.txt", U, k)
    filename_pauli = @sprintf("results_pauli_U%.1f_k%d.txt", U, k)

    save_results(filename_majo, results_majo)
    save_results(filename_pauli, results_pauli)
end

#Us = [2.0]#, 4.0, 6.0, 8.0, 10.0, 12.0]
#ks = [1,2,5,10]
#for u in Us
#    for k in ks
#    println("Calculation for k = ", k, "  U = ", u)
#    plot_abs_error_vs_weight_pdf(Lx=2, Ly=2, t=1.0, U=u, k=k, max_weights=1:1:16)
#    #properties_table(Lx=2, Ly=2, t=1.0, U=u, k=k, max_weights=1:1:8)
#    end
#end


function test_hubbard_2D()
    Lx = 1
    Ly = 64
    t = 1.0
    U = 2.0
    k = 1
    N = 2 * Lx * Ly  # Total qubits for spinful model

    o = Pauli(N)
    generators, parameters = UnitaryPruning.hubbard_model_2D_interleaved(o; Lx=Lx, Ly=Ly, t=t, U=U, k=k)

    println("2D Hubbard model generators and parameters:")
    for (gen, param) in zip(generators, parameters)
        #println("Generator: ", gen, ", Parameter: ", param)
        display(gen)
    end
    println("Total generators: ", length(generators))

end

#test_hubbard_2D()