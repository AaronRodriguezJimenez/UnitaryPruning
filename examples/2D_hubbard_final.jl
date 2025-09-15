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
end

function run(; N=10, k=5, thresh=1e-3, w_type = 0, w = 2)
   
    ket = Ket(N, 0) 
    o = Pauli(N, Z=[1])

    # generators, parameters = UnitaryPruning.fermi_hubbard_1D(o, t = 1, U = 6, k=k)
    generators, parameters, hammy = UnitaryPruning.fermi_hubbard_2D(o, t = 1, U = 2, k=k)

    ei , nops, c_norm2 = UnitaryPruning.bfs_evolution_thresh_weight(generators, parameters, PauliSum(o), ket, thresh=thresh, w_type = w_type, w = w)
    
    α = π/4

    @printf(" α: %6.4f e: %12.8f+%12.8fi nops: %6i norm2: %3.8f threshold: %3.10f\n", α, real(ei), imag(ei), nops[end], c_norm2, thresh)

    # U = UnitaryPruning.build_time_evolution_matrix_fast(generators, parameters)
    # o_mat = Matrix(o)
    # m = diag(U'*o_mat*U)
    # error = real(m[1]) -  real(ei) 
    # println(real(m[1]), " ", real(ei), " Error: ", error)


    # ψ = zeros(ComplexF64, 2^N); ψ[1] = 1.0
    # ψp = U * ψ
    # e = dot(ψp, Matrix(o) * ψp)
    # error = real(e) -  real(ei) 
    # println(real(e), " ", real(ei), " Error: ", error)

    return real(ei)
end

# Here need to change the t, U and o manually in the run_weight() and run() function.
function run_weights()
    L = 2
      
    N = 2*L*L
    o = Pauli(N, Z=[1])
    set_k = [1, 2, 3, 4, 5, 10]
    new_set_k = [1, 2, 5, 10]
    # thresholds = [1e-4, 1e-3]
    thresholds = [-1]
    for k in new_set_k
        println("k: ", k)
        
        # generators, parameters = UnitaryPruning.fermi_hubbard_1D(o, t = 1 , U = 6, k=k)
        #generators, parameters = UnitaryPruning.fermi_hubbard_2D(o, t = 1 , U = 2, k=k)


        #U = UnitaryPruning.build_time_evolution_matrix(generators, parameters)
        #o_mat = Matrix(o)
        #m = diag(U'*o_mat*U)
        #m = [0.9646848803317583 + 0.0im]
        m = [0]
        println("Exact Expectaition Value: ", m[1])

        weights = [i for i in 1:N]

        
        plt = plot(xlabel = "Weight Cutoff", ylabel = "Expectation Value Error", title = "L = $L, N = $N, k=$k", grid = true, dpi = 200, legend =:bottomright)

        for thresh in thresholds
            w_type = 0

            p_errors = []

            for i in weights
                println("Weight Threshold: ", i)
                e_v = run(N = N, k = k, thresh = thresh, w_type = w_type, w = i)
                push!(p_errors, abs(real(m[1]) - real(e_v)))
                # push!(p_errors, real(e_v))

            end
            display(p_errors)

            w_type = 1
            m_errors = []

            for i in weights
                println("Weight Threshold: ", i)
                e_v = run(N = N, k = k, thresh = thresh, w_type = w_type, w = i)
                push!(m_errors, abs(real(m[1]) - real(e_v)))
                # push!(m_errors, real(e_v))

            end

            display(m_errors)

            plot!(weights, [p_errors, m_errors],
                label = ["Pauli $thresh" "Majorana $thresh"],
                lw = 2,
                marker = :circle,
                # legend = :topright,
                grid = true)

        end
  
        #savefig("test/heisenberg_1d_weights_L$L-k$k.png")
        savefig("HUBBARD_2D_t=1_U=2-k$k.png")
    end
end

function run_ops(Lx; N=10, k=5, thresh=1e-3, w_type = 0, w = 2)
   
    ket = Ket(N, 0) 
    o = Pauli(N, Z=[1])

    # generators, parameters = UnitaryPruning.fermi_hubbard_1D(o, t = 1, U = 6, k=k)
    #generators, parameters = UnitaryPruning.fermi_hubbard_2D(o, t = 1.0, U = 2.0, k=k)
    #generators, parameters = UnitaryPruning.hubbard_model_2D_interleaved(o; Lx=Lx, Ly=Lx, t=1.0, U=2.0, k=k)
    generators, parameters = UnitaryPruning.fermi_hubbard_2D_pauli(o; Lx=Lx, Ly=Lx, t=1.0, U=2.0, k=k)

    ei , nops, c_norm2 = UnitaryPruning.bfs_evolution_thresh_weight(generators, parameters, PauliSum(o), ket, thresh=thresh, w_type = w_type, w = w)
    
    α = π/4

    @printf(" α: %6.4f e: %12.8f+%12.8fi nops: %6i norm2: %3.8f threshold: %3.10f\n", α, real(ei), imag(ei), maximum(nops), c_norm2, thresh)

    return real(ei), nops[end], length(generators)
end

function run_weights_and_ops(run_weights_plot::Bool = true, run_ops_plot::Bool = true)
    Lx = 2 # linear modes (for hubbard model 2_interleaved version)
    L = 4   # Total fermionic modes 4 for a 2x2 lattice, 9 for a 3x3 lattice and so on...
    N = 2*L  # Total Qubits 
    o = Pauli(N, Z=[1])
    new_set_k = [2,4,8,10]

    thresholds = [1e-3, 1e-4]
    #thresholds = [-1]

    weights = [i for i in 1:15]#2*N]

    for k in new_set_k
        println("k: ", k)
        
        # return
        # Placeholder for "exact" expectation value, set to 0
        m = [0.0]

        if run_weights_plot
            plt1 = plot(xlabel = "Weight Cutoff", ylabel = "Expectation Value ",
                        title = "L = $L, N = $N, k=$k", grid = true, dpi = 200, legend = :bottomright)
        end

        if run_ops_plot
            plt2 = plot(xlabel = "Weight Cutoff", ylabel = "Active Operators ",
                        title = "L = $L, N = $N, k=$k", grid = true, dpi = 200)
        end

        for thresh in thresholds
            for w_type in (0, 1)  # 0: Pauli, 1: Majorana
                #p_errors = Float64[]
                P_expval = Float64[]
                op_counts = Float64[]

                for w in weights
                    println("Weight Threshold: ", w)
                    # ev = run(N=N, k=k, thresh=thresh, w_type=w_type, w=w)

                    ev, nops, _ = run_ops(Lx, N=N, k=k, thresh=thresh, w_type=w_type, w=w)
                    push!(op_counts, nops)
                    push!(P_expval, ev)
                    #push!(p_errors, abs(real(m[1]) - real(ev)))

                end

                label = w_type == 0 ? "Pauli $thresh" : "Majorana $thresh"

                if run_weights_plot
                    plot!(plt1, weights, P_expval, label=label, lw=2, marker=:circle)
                end

                if run_ops_plot
                    plot!(plt2, weights, op_counts, label=label, lw=2, marker=:circle)
                end
            end
        end

        if run_weights_plot
            savefig(plt1, "Hubbard_2D_weights_L$L-k$k.png")
        end
        if run_ops_plot
            savefig(plt2, "Hubbard_2D_nops_L$L-k$k.png")
        end
    end
end


#run_weights() 
run_weights_and_ops()
