using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Plots
"""
 Here, we explore the OTOC estimation for the 1D XXZ Heisenberg
 model.
"""
function clip_coeff!(p::Dict{PauliBasis{N}, T}; thresh=1e-3) where {N, T}
    filter!(p-> abs(p.second) ≥ thresh, p)
end

"""
 The following funciton computes the OTOC expectation value between two operators
 and an initial state (ket).
 """
function otoc_expval(o_1::PauliSum{N}, o_2::PauliSum{N}, ket) where {N}
    output = 0

    for (oi_1, coeff_1) in o_1
        for (oi_2, coeff_2) in o_2
            output += expectation_value(oi_1*oi_2, ket)*coeff_1*coeff_2
        end
    end

    return output
end

function evolution_otoc(generators::Vector{Pauli{N}}, angles, o::PauliSum{N}, ket ; thresh=1e-3) where {N}

    #
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    vcos = cos.(angles)
    vsin = sin.(angles)

    # Data collection
    step_evo = Int64(nt/(3*(N-1)))
    otoc = Vector{Float64}(undef, step_evo+1)
    temp_o = PauliSum(N)
    time = 1+1
    o_transformed = deepcopy(o)
    dicts = Vector(undef, nt)    
    sin_branch = PauliSum(N)

     otoc[1] = real(expectation_value(o, ket))
    
    for t in 1:nt

        g = generators[t]
        pb = PauliBasis(g)
        sin_branch = PauliSum(N)

        for (oi,coeff) in o_transformed
           
            abs(coeff) > thresh || continue


            if !PauliOperators.commute(oi, pb)
                
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's
                sum!(sin_branch, oj * vsin[t] * coeff * 1im)
  
            end
        end
        sum!(o_transformed, sin_branch) 
        clip_coeff!(o_transformed, thresh=thresh)

        dicts[t] = deepcopy(o_transformed)

        if t%(3*(N-1)) == 0
            println("Time: ", time)
            os = o*o_transformed
            otoc[time] = real(otoc_expval(os, os, ket))
            time += 1
        end

    end

    return otoc, dicts
end


function run(; N=10,threshold = 1e-3, dt=0.1, T=10)
   
    ket = Ket(N, 0) 
    o = Pauli(N, Z=[1])
    k = Int64(floor(T/dt))
    x = 0.9
    y = 0.9
    z = 0.5

    generators, parameters = UnitaryPruning.heisenberg_1D(o, Jx = x*dt, Jy = y*dt,Jz = z*dt, k=k)
    otoc, dict = evolution_otoc(generators, parameters, PauliSum(o), ket, thresh = threshold)

    # @printf("α: %6.4f e: %12.8f+%12.8fi nops: %6i norm2: %3.8f threshold: %3.10f\n", α, real(ei_n), imag(ei_n), maximum(nops_n), c_norm2_n, threshold)

    #println(otoc)
    evol = []
    for data in dict
        temp_pauli = []
        temp_coeff = []
        for (oi, coeff) in data
            push!(temp_pauli, string(oi))
            push!(temp_coeff, coeff)
        end
        push!(evol, temp_pauli)
        push!(evol, temp_coeff)
    end
    time_steps = [ dt*i for i in 1:k+1]

    plot(time_steps, otoc, titlefontsize = 5)
    xlabel!("Time")
    ylabel!("OTOC")
    title!("N=$N; Jx = $x, Jy = $y, Jz = $z, thresh = $threshold; dt = $dt")
    savefig("test_otoc_N=$N-thresh=$threshold-dt=$dt-T=$T.pdf")

    # writedlm("test/tree_evolN$N-$threshold.dat", evol)
    return
end

run(N = 60, threshold = 1e-3, dt = 1.0, T = 100)