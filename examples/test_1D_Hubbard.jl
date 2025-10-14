using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators

#
#- - - 1D Hubbard model 
#
# Model from Hamlib: /fh-graph-1D-grid-pbc-qubitnodes_Lx-2_U-2_enc-jw
# model with periodic boundary conditions 
#OPENFERMION FORMAT
#(1+0j) [] +
#(-0.5+0j) [X0 X1] +
#(-0.5+0j) [Y0 Y1] +
#(-0.5+0j) [Z0] +
#(0.5+0j) [Z0 Z2] +
#(-0.5+0j) [Z1] +
#(0.5+0j) [Z1 Z3] +
#(-0.5+0j) [X2 X3] +
#(-0.5+0j) [Y2 Y3] +
#(-0.5+0j) [Z2] +
#(-0.5+0j) [Z3]

function hubbard_1D_test()
    generators = Vector{Pauli{4}}([])
    parameters = Vector{Float64}([])

    push!(generators, Pauli(4))
    push!(parameters, 1.000)
    push!(generators, Pauli(4, X=[1, 2]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Y=[1, 2]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Z=[1]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Z=[1, 3]))
    push!(parameters, 0.5000)
    push!(generators, Pauli(4, Z=[2]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Z=[2, 4]))
    push!(parameters, 0.5000)
    push!(generators, Pauli(4, X=[3, 4]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Y=[3, 4]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Z=[3]))
    push!(parameters, -0.5000)
    push!(generators, Pauli(4, Z=[4]))
    push!(parameters, -0.5000)

    return generators, parameters
end

"""
Data from Hamlib: /fh-graph-1D-grid-pbc-qubitnodes_Lx-6_U-2_enc-jw

 SparsePauliOp(['IIIIIIIIIIII', 'XXIIIIIIIIII', 'XZZZZXIIIIII', 'YYIIIIIIIIII', 'YZZZZYIIIIII', 'ZIIIIIIIIIII', 'ZIIIIIZIIIII', 'IXXIIIIIIIII', 'IYYIIIIIIIII', 'IZIIIIIIIIII', 'IZIIIIIZIIII', 'IIXXIIIIIIII', 'IIYYIIIIIIII', 'IIZIIIIIIIII', 'IIZIIIIIZIII', 'IIIXXIIIIIII', 'IIIYYIIIIIII', 'IIIZIIIIIIII', 'IIIZIIIIIZII', 'IIIIXXIIIIII', 'IIIIYYIIIIII', 'IIIIZIIIIIII', 'IIIIZIIIIIZI', 'IIIIIZIIIIII', 'IIIIIZIIIIIZ', 'IIIIIIXXIIII', 'IIIIIIXZZZZX', 'IIIIIIYYIIII', 'IIIIIIYZZZZY', 'IIIIIIZIIIII', 'IIIIIIIXXIII', 'IIIIIIIYYIII', 'IIIIIIIZIIII', 'IIIIIIIIXXII', 'IIIIIIIIYYII', 'IIIIIIIIZIII', 'IIIIIIIIIXXI', 'IIIIIIIIIYYI', 'IIIIIIIIIZII', 'IIIIIIIIIIXX', 'IIIIIIIIIIYY', 'IIIIIIIIIIZI', 'IIIIIIIIIIIZ'],
              coeffs=[ 3. +0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5+0.j,
 -0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,
  0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j,
 -0.5+0.j,  0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,
 -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,
 -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j, -0.5+0.j,
 -0.5+0.j])
"""
function hubbard_1D_test_6_sites()
    generators = Vector{Pauli{12}}([])
    parameters = Vector{Float64}([])

    function parse_pauli(s::String)
        X = Int[]
        Y = Int[]
        Z = Int[]
        for (i, c) in enumerate(s)
            if c == 'X'
                push!(X, i)
            elseif c == 'Y'
                push!(Y, i)
            elseif c == 'Z'
                push!(Z, i)
            end
        end
        return Pauli(12, X=X, Y=Y, Z=Z)
    end

    pauli_strings = [
        "IIIIIIIIIIII", "XXIIIIIIIIII", "XZZZZXIIIIII", "YYIIIIIIIIII", "YZZZZYIIIIII",
        "ZIIIIIIIIIII", "ZIIIIIZIIIII", "IXXIIIIIIIII", "IYYIIIIIIIII", "IZIIIIIIIIII",
        "IZIIIIIZIIII", "IIXXIIIIIIII", "IIYYIIIIIIII", "IIZIIIIIIIII", "IIZIIIIIZIII",
        "IIIXXIIIIIII", "IIIYYIIIIIII", "IIIZIIIIIIII", "IIIZIIIIIZII", "IIIIXXIIIIII",
        "IIIIYYIIIIII", "IIIIZIIIIIII", "IIIIZIIIIIZI", "IIIIIZIIIIII", "IIIIIZIIIIIZ",
        "IIIIIIXXIIII", "IIIIIIXZZZZX", "IIIIIIYYIIII", "IIIIIIYZZZZY", "IIIIIZIIIII",
        "IIIIIIIXXIII", "IIIIIIIYYIII", "IIIIIIIZIIII", "IIIIIIIIXXII", "IIIIIIIIYYII",
        "IIIIIIIIZIII", "IIIIIIIIIXXI", "IIIIIIIIIYYI", "IIIIIIIIIZII", "IIIIIIIIIIXX",
        "IIIIIIIIIIYY", "IIIIIIIIIIZI", "IIIIIIIIIIIZ"
    ]

    coeffs = [
        3.0, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5,
        -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
        0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5
    ]

    for (p, c) in zip(pauli_strings, coeffs)
        push!(generators, parse_pauli(p))
        push!(parameters, c)
    end

    return generators, parameters
end


function run(; N = 4, w_type = "Majorana", max_weight=1)

    ket = Ket(N,0)
    println("Ket: ", ket)
    o = Pauli(N, Z=[1])

    #Create generators and parameters for the model
    generators, parameters = hubbard_1D_test_6_sites()
    #generators, parameters = hubbard_1D_test()
    
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

function plot_abs_error_vs_weight_pdf(; N=4, w_type="Pauli", max_weights=0:2:6)
    errors = Float64[]
    weights = Int[]
    
    for mw in max_weights
        println("Evaluating max_weight = $mw")
        err = run(N=N, w_type=w_type, max_weight=mw)
        push!(errors, err)
        push!(weights, mw)
    end

    plt = plot(
        weights, errors,
        xlabel = "Max Weight Cutoff",
        ylabel = "Absolute Error",
        title = "Error vs Max Weight Cutoff (N = $N, 6 sites)",
        marker = :circle,
        lw = 2,
        legend = false,
        grid = true
    )

    filename="1D_Hubbard_test_abs_error_vs_weight_$w_type-N=$N-6sites.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end
    
#plot_abs_error_vs_weight_pdf(N=12, w_type="Pauli", max_weights=1:2:12)

# Test 1D Hubbard model
#
function test_hubbard_1D()
    L = 20
    t = 1.0
    U = 2.0
    k = 1
    N = 2 * L  # Total qubits for spinful model

    o = Pauli(N)
    generators, parameters = UnitaryPruning.hubbard_model_1D(o; L=L, t=t, U=U, k=k)

    println("1D Hubbard model generators and parameters:")
    for (gen, param) in zip(generators, parameters)
        #println(" Parameter: ", param)
        display(gen)
    end
    println("Total generators: ", length(generators))
    println("Parameters: ", parameters)

end

test_hubbard_1D()
