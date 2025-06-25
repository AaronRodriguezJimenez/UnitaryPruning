using Distributed
@everywhere begin
    using UnitaryPruning
    using Plots
    using Statistics
    using Printf
    using Random
    using LinearAlgebra
    using SharedArrays
    using PauliOperators
    using JSON
end


function H2_model_test()

    # For H2 molecue
    # pauli_ops = ["IIII", "IIIZ", "IIZI", "IZII", "ZIII", "IIZZ", "IZIZ", "ZIIZ",
    #             "YYYY", "XXYY", "YYXX", "XXXX", "IZZI", "ZIZI", "ZZII"]

    # coeffs=[-0.81217061,  0.17141283, -0.22343154,  0.17141283,
    #         -0.22343154,  0.12062523,  0.16868898,  0.16592785,
    #         0.04530262,  0.04530262,  0.04530262,  0.04530262,
    #         0.16592785,  0.17441288,  0.12062523]

    data = JSON.parsefile("examples/h2/H2_hamiltonian.json")
    pauli_ops = data["paulis"]
    coeffs = ComplexF64[]
    for (re, im) in zip(data["coeffs_real"], data["coeffs_imag"])
        push!(coeffs, complex(re, im))
    end

    N = length(pauli_ops[1])
    ket = Ket(N, 3) 
    display(ket)
    H = PauliSum(N)
    for i in eachindex(pauli_ops)
        sum!(H, Pauli(pauli_ops[i])*coeffs[i])
    end
    display(H)
    return
end

H2_model_test()
