using Distributed
using UnitaryPruning
using Plots
using Statistics
using Printf
using Random
using LinearAlgebra
using PauliOperators
using SparseArrays

function jw_transform(o::Pauli{N}, site) where N
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
    Nsites = Int(N / 2)              # Total number of lattice sites
    L = Int(sqrt(Nsites))           # Lattice size L × L
    D = Nsites                      # For clarity

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

function run(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1 , w_type = "Majorana", max_weight=1)

    N = 2*Lx*Ly
    ket = Ket(N,0)
    o = Pauli(N, Z=[1])

    #Create generators and parameters for the model
    generators, parameters = fermi_hubbard_2D(o, t=t, U=U, k=k)
    #generators, parameters = fermi_hubbard_2D_block(o, t=t, U=U, k=k)
    

    #Call to bfs bfs_evolution_test based on weight
    ei, nops = UnitaryPruning.bfs_evolution_weight(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)
    #ei, nops = UnitaryPruning.bfs_evolution_weight_clip(generators, parameters, PauliSum(o), ket, w_type, max_weight=max_weight)

    # Exact evolution (Schrodinger picture)
    vector_ket = Vector(ket)
    #println("Vector ket: $ket -> ", vector_ket)
    U_psi = compute_schrodinger_evol(generators, parameters, vector_ket)
    expval = U_psi' * UnitaryPruning.matvec(o, 1.00, U_psi)
    println("Expval Schr :", expval)

    
    # Exact evolution (Heisenberg picture)
    #U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
    #o_mat = Matrix(o)
    #m = diag(U'*o_mat*U)
    #expval = m[1]

    abs_err = abs(real(expval)- real(ei) )
    println("Exact :", real(expval), " Approx :", real(ei), " Absolute Error: ", abs_err)
    return abs_err
end

function plot_abs_error_vs_weight_pdf(; Lx = 2, Ly = 2, t = 1.0, U = 2.0, k=1, max_weights=0:2:6)
    errors = Float64[]
    weights = Int[]

    errors_pauli = Float64[]
    weights_pauli = Int[]


    for mw in max_weights
        #println("Evaluating max_weight = $mw")
        err = run(Lx = Lx, Ly = Ly, t = t, U = U, k=k, w_type="Majorana", max_weight=mw)
        push!(errors, err)
        push!(weights, mw)
        println("Majorana : maxw : $mw  k : $k ei : $err")
    end

    for mw in max_weights
        #println("Evaluating max_weight = $mw")
        err = run(Lx = Lx, Ly = Ly, t = t, U = U, k=k, w_type="Pauli", max_weight=mw)
        push!(errors_pauli, err)
        push!(weights_pauli, mw)
        println("Pauli : maxw : $mw  k : $k ei : $err")

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
    ylabel!("Estimated energy")
    #ylabel!("Absolute Error")
    title!("Hubbard, U=$U, t=-$t, k=$k")
    title!("Error vs Max Weight Cutoff (Hubbard, U=$U, t=-$t)")

    #filename="2D_Hubbard_test_energy_Lx=$Lx-Ly=$Ly-k=$k-CH.pdf"
    filename="TEST-2D_Hubbard_test_abs_error_vs_weight_Lx=$Lx-Ly=$Ly-k=$k-CH.pdf"
    savefig(plt, filename)
    println("Plot saved as $filename")
end
    
plot_abs_error_vs_weight_pdf(Lx=2, Ly=2, t=1.0, U=2.0, k=1, max_weights=1:1:8)

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