using LinearAlgebra
using UnitaryPruning
using PauliOperators

"""
 In this example, we check the correctness of the bfs_evolution_weight function
 by comparing the estimation of the expectation value of <Z1> after evolving
 under a 2D Hubbard model Hamiltonian with the exact evolution.

 NOTE: For the standard (number-conserving) Hubbard model, starting in the all-zero 
 computational state (the fermionic vacuum), the expectation value 
⟨𝑍1(𝑡)⟩ stays constant at its initial value—i.e. +1
"""

#- - - Check the expectation value o Z1
u = 0.3  # On-site interaction strength
t = 0.2  # hopping parameter

Lx = 2 # Number of sites in x direction
Ly = 2 # Number of sites in y direction
N = 2*Lx*Ly # Number of qubits

o = Pauli(N, Z=[1])  # Observable Z1
ket = Ket(N, 0)  # 4 qubits, all in state |0>

exp_val_exact = PauliOperators.expectation_value(o, ket)  # Initial expectation value of Z1

# Define the generators and angles for the 2D Hubbard model
generators, parameters = UnitaryPruning.hubbard_model_2D_interleaved(o, Lx=Lx, Ly=Ly, t=t, U=u, k=1)
#generators, parameters = UnitaryPruning.fermi_hubbard_2D(o, t=t, U=u, k=1)

println("* * * * Hamiltonian Generators and Parameters * * * *")
for g in generators
    display(g)
end

for p in parameters
    println("Parameter: ", p)
end

println("==== DEBUGGING HUBBARD MODEL ====")

function isidentity(p::Pauli{N}) where {N}
    pstring = string(p)
    return all(c -> c == 'I', pstring)
end

# Check identity terms
for (g, p) in zip(generators, parameters)
    #println(isidentity(g))
    if isidentity(g)
        println("* * * Found identity term with coeff=", p)
    end
end

# Check hermiticity
non_hermitian_terms = []
for (g, p) in zip(generators, parameters)
    if g != g'
        #println("* * * Non-Hermitian term: ", g, " coeff=", p)
        push!(non_hermitian_terms, (g, p))
    end
end

# Check commutators
non_commuting_pairs = []
for i in 1:length(generators)
    for j in i+1:length(generators)
        if !PauliOperators.commute(generators[i], generators[j])
            #println("* * * Non-commuting terms: ", i, " vs ", j)
            push!(non_commuting_pairs, (i, j))
        end
    end
end

# Test action on vacuum state
vac = Ket(N, 0)
terms = Vector{Pauli{N}}()

println("Transformation of the Initial vacuum state:")
for (g, p) in zip(generators, parameters)
    v0 = Vector(vac)    
    new_vac = Matrix(g) * v0

    if new_vac != v0
        #display(g*vac)
        push!(terms, g)
    end
end

println("Total number of generators: ", length(generators))
println("Number of terms that move vacuum: ", length(terms))
for t in terms
    display(t)
end

# Tests within matrix evaluation
# Build exact fermionic operators as matrices
function fermionic_creation(N, i)
    # Build 2^N x 2^N matrix
    dim = 2^N
    mat = zeros(ComplexF64, dim, dim)
    for state in 0:dim-1
        # check if site i is occupied
        if iszero((state >> (i-1)) & 0x1)
            new_state = state | (1 << (i-1))
            # fermionic sign = number of 1's below i
            sign = (-1)^(count_ones(state & ((1 << (i-1))-1)))
            mat[new_state+1, state+1] = sign
        end
    end
    return mat
end

function fermionic_annihilation(N, i)
    # c_i = (c_i^\dagger)†
    return fermionic_creation(N, i)'
end

function fermionic_bilinear(N, i, j)
    return fermionic_creation(N, i) * fermionic_annihilation(N, j)
end

function hubbard_2D_fermionic_matrix(o::Pauli{N}; Lx::Int64, Ly::Int64, t::Float64, U::Float64) where N
    dim = 2^N
    H_hop = zeros(ComplexF64, dim, dim)
    H_onsite = zeros(ComplexF64, dim, dim)

    # Hopping terms
    a_fn(i) = 2*i - 1  # spin-up
    b_fn(i) = 2*i      # spin-down

    for x in 1:Lx
        for y in 1:Ly
            i = (x-1)*Ly + y
            # Hopping in x direction
            if x < Lx
                j = x*Ly + y
                i_up, j_up = a_fn(i), a_fn(j)
                H_hop += t * fermionic_bilinear(N, i_up, j_up)
                i_down, j_down = b_fn(i), b_fn(j)
                H_hop += t * fermionic_bilinear(N, i_down, j_down)
            end
            # Hopping in y direction
            if y < Ly
                j = (x-1)*Ly + (y+1)
                i_up, j_up = a_fn(i), a_fn(j)
                H_hop += t * fermionic_bilinear(N, i_up, j_up)
                i_down, j_down = b_fn(i), b_fn(j)
                H_hop += t * fermionic_bilinear(N, i_down,j_down)
            end
        end
    end

    # On-site interaction terms
    for site in 1:(Lx*Ly)
        i_up, j_up = a_fn(site), a_fn(site)
        n_up = fermionic_bilinear(N, i_up, j_up)
        i_down, j_down = b_fn(site), b_fn(site)
        n_down = fermionic_bilinear(N, i_down,j_down)
        H_onsite += U * n_up * n_down
    end

    H_total = H_hop + H_onsite
    return Matrix(H_total)
end

H_mat = hubbard_2D_fermionic_matrix(o; Lx=Lx, Ly=Ly, t=t, U=u)

# Compare the expectation value before and after evolution
println("Initial expectation value <Z1>: ", exp_val_exact)

# Evolve the operator and compute exact evolution
U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
o_mat = Matrix(o)
m = diag(U'*o_mat*U)
println("Exact expectation value <Z1> after bfs evolution: ", m[1])

# Perform computation with this matrix
U_exact = exp(-1im * H_mat)
m2 = diag(U_exact'*o_mat*U_exact)
println("Exact expectation value <Z1> after evolution (matrix method): ", m2[1])

# Further tests with the Hamiltonian from unitary sequences
H = zeros(ComplexF64, 2^N, 2^N)
for (g, p) in zip(generators, parameters)
    H .+= p * Matrix(g)
end

println("Is H Hermitian? ", ishermitian(H))

Vacuum = zeros(ComplexF64, 2^N); Vacuum[1] = 1
println("Vacuum energy ", Vacuum' * H * Vacuum)

Z1_op = Matrix(Pauli(N, Z=[1]))
println("Vacuum <Z1> ", Vacuum' * Z1_op * Vacuum)

# Evolve the operator with bfs_evolution
exp_val_bfs, n_ops = UnitaryPruning.bfs_evolution(generators, parameters, PauliSum(o), ket; thresh=1e-6)
#ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

println("BFS evolution expectation value <Z1>: ", exp_val_bfs)
println("Number of operators during evolution: ", n_ops)   


# Evolve the operator with bfs_evolution_weight
max_weight = 2
w_type = "Pauli"  # "Majorana" or "Pauli"
exp_val_bfs_w, n_ops_w = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type; max_weight=max_weight)
#ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight) 
println("BFS (weight) evolution expectation value <Z1>: ", exp_val_bfs_w)
println("Number of operators during (weight) evolution: ", n_ops_w)
