#
#  Here we explore the performance of the Majorana vs Pauli weight-based pruning in for the case of the
#  1D Hubbard model.
#
using HDF5

"""
 Functions for extracting the hamiltonian from HamLib data.
"""
function to_pauli_string(term::String, nqubits::Int64)
    result = fill('I', nqubits)

    isempty(strip(term)) && return join(result)

    for m in eachmatch(r"([XYZ])(\d+)", term)
        pauli = m.captures[1][1]  # Get first character as Char
        idx = parse(Int, m.captures[2])
        result[idx + 1] = pauli  # Julia uses 1-based indexing
    end

    return join(result)
end

function read_hubbard_hdf5(location::String, name_file::String, N::Int64)
    # name_file is the name of the data in the datafile
    # example: "/fh-graph-1D-grid-pbc-qubitnodes_Lx-6_U-2_enc-jw"
    # location is a string for the route to the hdf5
    # example: "FH_D-1.hdf5"
    # N - is the number of qubits known a priori

    generators = Vector{PauliBasis{N}}([])
    parameters = Vector{Float64}([])


    fid=h5open(location,"r")
    dset = fid[name_file] 
    data=read(dset)

    labels = String[]
    coeffs = ComplexF64[]
    pattern = r"\(([^)]+)\)\s*\[([^\]]*)\]"
    matches = collect(eachmatch(pattern, data))
    

    for m in matches
        coeff_str = m.captures[1]
        term_str = m.captures[2]
        #println(m)
        #println(coeff_str, term_str)
        coeff = parse(ComplexF64, strip(coeff_str))
        term = strip(term_str)
        push!(coeffs, coeff)
        push!(labels, term)
    end

    formatted = [to_pauli_string(term, N) for term in labels]
    println(formatted)
    
    for (p, c) in zip(formatted, coeffs)
        Pb = PauliBasis(p)
        push!(generators, Pb)
        push!(parameters, c)
    end
    
    close(fid)
    return generators, parameters
end


function run(; Lx=2, Ly =2, U=0, w_type = "Majorana", max_weight=1)
   
    N = Lx*Ly
    ket = Ket(N,0)
    o = Pauli(N, Z=[1, 2])
    
    loc = "/mnt/8E1C11D91C11BD61/Aaron_backup_JUN2025/Fermi-Hubbard/FH_D-2.hdf5"
    name = string("/fh-graph-1D-grid-pbc-qubitnodes_Lx-", Lx, "_U-", U,"_enc-jw")

    # Parse generators and parameters from Hamlib Data
    generators, coeffs = read_hubbard_hdf5(loc, name, N)

    
    #Call to bfs bfs_evolution_test based on weight
    
    ei, nops = UnitaryPruning.bfs_evolution_weight(generators, coeffs, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, coeffs)
    o_mat = Matrix(o)
    m = diag(U'*o_mat*U)
    abs_err = abs(real(m[1])- real(ei) )
    println("Exact :", real(m[1]), " Approx :", real(ei), " Absolute Error: ", abs_err)
    return abs_err
end


############### Generate results functions #######################
function plot_series_and_save_results(; 
    LXs = [2, 3],
    LYs = [2, 3],
    Us = [0, 2, 4, 8, 12], 
    output_dir = "2D_HUBBARD_series_results",
    txt_filename = "data_2d_bfs_heisenberg_series.txt"
)

    # Make directory if it doesn't exist
    isdir(output_dir) || mkpath(output_dir)

    txt_file_path = joinpath(output_dir, txt_filename)
    open(txt_file_path, "w") do io
        for L in LXs # Retained for calculations of square latices 2x2 and 3x3
            for U in Us
                N = 2*L #2 qubits per spin
                println("Running N = $N, U = $U")
                max_weights = 1:1:2*N
                errors_dict = Dict("Pauli" => Float64[], "Majorana" => Float64[])

                for w_type in ["Pauli", "Majorana"]
                    for mw in max_weights
                        println("  -> w_type = $w_type, max_weight = $mw")
                        err = run(Lx = L, Ly=L, U = U, w_type = w_type, max_weight = mw)
                        push!(errors_dict[w_type], err)
                        # Save result line to file
                        @printf(io, "N=%d U=%d w_type=%s max_weight=%d error=%.6e\n", N, U, w_type, mw, err)
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
                title!("Error vs Max Weight (N=$N, U=$U)")
                xlabel!("Max Weight Cutoff")
                ylabel!("Absolute Error")
                

                filename = "abs_error_vs_weight_N=$N-U=$U.pdf"
                savefig(plt, joinpath(output_dir, filename))
                println("Saved plot as $filename")
            end
        end
    end

    println("All results saved in $txt_file_path")
end

# Call the function to execute the study
plot_series_and_save_results()