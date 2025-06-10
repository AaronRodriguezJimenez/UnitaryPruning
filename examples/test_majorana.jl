# Example usage PauliOperators
# Based on examples > plot_1d_n6_reference.jl
using Distributed
@everywhere begin
    using UnitaryPruning
    using Plots
    using Statistics
    using Printf
    using Random
    using LinearAlgebra
    using PauliOperators
end

println(" ")
println("* * * * * Testing PauliOperator functions * * * * * *")
println(" ")

# Create two pauli basis for comparisons and basic operations
Pbasis = PauliBasis("ZZZ")
Qbasis = PauliBasis("III")
ket = Ket(3, 0)

println("P =", Pbasis, "Q =", Qbasis, "Ket =", ket)
println("P and Q commute?", PauliOperators.commute(Pbasis, Qbasis))
println("Expectation value test: ", expectation_value(Qbasis, ket))

# Additional tests for operations.jl in PauliOperators
# Z on qubit 1
Z1 = PauliBasis("ZII")
# X on qubit 1 — non-diagonal
X1 = PauliBasis("XII")

# |000⟩
ket0 = Ket(3, 0b000)
# |010⟩ — qubit 2 is 1
ket1 = Ket(3, 0b010)

println("Z1, ket0: ", expectation_value(Z1, ket0))  # should be +1
println("Z1, ket1: ", expectation_value(Z1, ket1))  # still +1 (Z on qubit 1, which is 0)
println("Z1, Ket(0b100): ", expectation_value(Z1, Ket(3, 0b100)))  # qubit 1 is 1 ⇒ should be -1

# Non-diagonal
println("X1, ket0 (non-diagonal): ", expectation_value(X1, ket0))  # should be 0


println(" ")
println("* * * * * Testing Majorana weights * * * * * *")
println(" ")
# Basic usage of Majorana weight estimation
w = PauliOperators.simple_majorana_weight(Pbasis)
println("Simple Majorana weight is: ", w)

w_two = PauliOperators.pauli_to_majorana_occupation(Pbasis)
println("Majorna weight ver 2 is: ", w_two)
println("")

# Generate all possible Pauli strings for a given Number

N = 2
all_paulis, nops = PauliOperators.generate_all_pauli_strings(N)

@printf("Total operators generated: %i\n", nops)
println("- - Simple weight // Pauli 2 Majorana Weight - - - ")

for (i,op) in enumerate(all_paulis)
    w_simple = PauliOperators.simple_majorana_weight(op)

    w_p2m = PauliOperators.pauli_to_majorana_occupation(op)

    p_string = PauliOperators.string(op)

    println("Pauli $i - $p_string:    ", w_simple, "   ", w_p2m[1])    

    println("  ")
end
