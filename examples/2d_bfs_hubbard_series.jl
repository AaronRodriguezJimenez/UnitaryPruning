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
function JWmapping(a::Int, b::Int, N::Int) 
    # Compute C^dagger_i term
    ax_term = Pauli(2^(a-1)-1, 2^(a-1), N)
    ay_term = Pauli(2^(a)-1, 2^(a-1), N)
    c_dagg_a = 0.5 * (ax_term - ay_term)

    # Compute C_j term
    bx_term = Pauli(2^(b-1)-1, 2^(b-1), N)
    by_term = Pauli(2^(b)-1, 2^(b-1), N)
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
  
"""
function hubbard_model_2D(Lx::Int, Ly::Int, t::Float64, U::Float64)
    D = Lx * Ly # The dimension of the lattice
    #A = zeros(Int, D, D)
    N = 2 * Lx * Ly
    H_hop = PauliSum(N)
    H_u = PauliSum(N)

    generators = Vector{Pauli{N}}()
    parameters = Vector{Float64}()
    #- - - Hopping term - - -
    for y in 0:(Ly - 1)
        for x in 0:(Lx - 1)
            i = y * Lx + x + 1  # Index for site i

            # Right neighbor
            if x < Lx - 1
                j = y * Lx + (x + 1) + 1
                #A[i, j] = 1
                #A[j, i] = 1
                for spin in 0:1
                    a = i + spin * D #mode index for spin
                    b = j + spin * D
                    H_hop += JWmapping(a,b,N) 
                    H_hop += JWmapping(b,a,N)
                end
            end

            # Bottom neighbor
            if y < Ly - 1
                j = (y + 1) * Lx + x + 1
                #A[i, j] = 1 # For testing
                #A[j, i] = 1
                for spin in 0:1
                    a = i + spin * D  #mode index for spin
                    b = j + spin * D
                    H_hop += JWmapping(a,b,N)
                    H_hop += JWmapping(b,a,N)
                end
            end
        end
    end

    for (pauli, coeff) in H_hop
        push!(generators, Pauli(pauli))
        push!(parameters, -t*coeff)
    end
    # - - - Interaction term - - - 
    for j in 1:D
        for spin in 0:1
            a = j + spin * D  #mode index for spin
            H_u += JWmapping(a,a,N)
        end
    end

    for (pauli, coeff) in H_u
        push!(generators, Pauli(pauli))
        push!(parameters, U*coeff)
    end
    return generators, parameters
end

function run(; Lx = 2, Ly = 2, t = 1.0, U = 2.0 , w_type = "Majorana", max_weight=1)
    N = 2*Lx*Ly
    ket = Ket(N,0)
    println("Ket: ", ket)
    o = Pauli(N, Z=[1])

    #Create generators and parameters for the model
    generators, parameters = hubbard_model_2D(Lx, Ly, t, U)

    #Call to bfs bfs_evolution_test based on weight
    
    ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    o_mat = Matrix(o)
    m = diag(U'*o_mat*U)
    abs_err = abs(real(m[1])- real(ei) )
    println("Exact :", real(m[1]), " Approx :", real(ei), " Absolute Error: ", abs_err)
    return abs_err
end


############### Generate results functions #######################
function plot_series_and_save_results(; 
    lattice_sizes = [(2, 2), (2, 3)],
    t = 1.00,
    Us = [0.0, 2.0, 4.0, 8.0, 12.0], 
    output_dir = "2D_HUBBARD_series_results",
    txt_filename = "data_2d_bfs_heisenberg_series.txt"
)

    # Create output directory if needed
    isdir(output_dir) || mkpath(output_dir)

    txt_file_path = joinpath(output_dir, txt_filename)

    open(txt_file_path, "w") do io

        for (Lx, Ly) in lattice_sizes
            Nsites = Lx * Ly
            N = 2 * Nsites  # 2 qubits per site (↑ and ↓)

            for U in Us
                println("Running Lx = $Lx, Ly = $Ly, N = $N, U = $U")
                max_weights = 1:1:2*N
                errors_dict = Dict("Pauli" => Float64[], "Majorana" => Float64[])

                for w_type in ["Pauli", "Majorana"]
                    for mw in max_weights
                        println("  -> w_type = $w_type, max_weight = $mw")
                        err = run(Lx = Lx, Ly = Ly, t = t, U = U, w_type = w_type, max_weight = mw)
                        push!(errors_dict[w_type], err)
                        @printf(io, "Lx=%d Ly=%d N=%d U=%d w_type=%s max_weight=%d error=%.6e\n", 
                                Lx, Ly, N, U, w_type, mw, err)
                    end
                end

                # Plot comparison
                plt = plot(
                    max_weights, errors_dict["Pauli"],
                    label = "Pauli",
                    marker = :circle,
                    lw = 2
                )
                plot!(
                    max_weights, errors_dict["Majorana"],
                    label = "Majorana",
                    marker = :square,
                    lw = 2
                )
                title!("Error vs Max Weight (Lx=$Lx, Ly=$Ly, U=$U)")
                xlabel!("Max Weight Cutoff")
                ylabel!("Absolute Error")
                
                filename = "abs_error_vs_weight_Lx=$Lx-Ly=$Ly-U=$U.pdf"
                savefig(plt, joinpath(output_dir, filename))
                println("Saved plot as $filename")
            end
        end
    end

    println("All results saved in $txt_file_path")
end

# Call the function to execute the study
plot_series_and_save_results()