using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators

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

#
# Run simulation for 2D TFIM
#
function run_2D_TFIM(; L=3, k=6, w_type="Majorana", max_weight=1)
    N = L^2
    ket = Ket(N, 0) 
    o = Pauli(N, Z=[1])   # observable: Z on first site
    α = π / 2  # coupling strength

    # Generators and parameters for 2D lattice
    generators, parameters = get_unitary_sequence_2D_test(L; α=α, k=k)

    # Approximate evolution
    ei, nops = UnitaryPruning.bfs_evolution_weight(
        generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight
    )

    # Exact evolution
    U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    o_mat = Matrix(o)
    m = diag(U' * o_mat * U)
    abs_err = abs(real(m[1]) - real(ei))

    println("Exact :", real(m[1]), 
            " Approx :", real(ei), 
            " Absolute Error: ", abs_err)

    return abs_err
end

function plot_abs_error_vs_weight_pdf_2D_compare(; L=3, k=10, max_weights=0:2:6)
    errors_majorana = Float64[]
    errors_pauli = Float64[]
    weights = Int[]

    for mw in max_weights
        println("Evaluating max_weight = $mw (Majorana)")
        err_maj = run_2D_TFIM(L=L, k=k, w_type="Majorana", max_weight=mw)
        push!(errors_majorana, err_maj)

        println("Evaluating max_weight = $mw (Pauli)")
        err_pau = run_2D_TFIM(L=L, k=k, w_type="Pauli", max_weight=mw)
        push!(errors_pauli, err_pau)

        push!(weights, mw)
    end

    plt = plot(
        weights, errors_majorana,
        xlabel = "Max Weight Cutoff",
        ylabel = "Absolute Error",
        title = "2D TFIM (N = $(L^2), k = $k)",
        marker = :circle,
        lw = 2,
        label = "Majorana",
        grid = true
    )

    plot!(weights, errors_pauli, marker=:square, lw=2, label="Pauli")

    filename = "2D_TFIM_abs_error_vs_weight-L=$L-k=$k-h_large.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end

# Example run
plot_abs_error_vs_weight_pdf_2D_compare(L=2, k=10, max_weights=1:1:8)
