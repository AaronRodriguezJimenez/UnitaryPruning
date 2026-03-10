using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random
using StatsBase

function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
end

# return a PauliOperators Ket equivalent to a given bitstring
function string_to_ket(bits::String)
    b = collect(bits)
    v = parse.(Int128, b)
    N = length(v)
    out = 0
    count = 0

    for bit in v
        if bit%2 == 1
            out += 2^count
        end
        count +=1
    end
    ket = Ket{N}(out)
    return ket, out
end

"""
 Evolve function from DBF code
"""
function evolve!(O::PauliSum{N, T}, G::PauliBasis{N}, θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im*sin(θ)
    sin_branch = PauliSum(N)
    for (p,c) in O
        if PauliOperators.commute(p,G) == false
            # replace sum! with more efficient version
            # sum!(sin_branch, c*_sin*G*p)
            tmp = c*_sin*G*p
            curr = get(sin_branch, PauliBasis(tmp), 0.0) + PauliOperators.coeff(tmp)
            sin_branch[PauliBasis(tmp)] = curr 
            O[p] *= _cos
        end
    end
    sum!(O, sin_branch)
    return O 
end

#
#- - - Hamiltonian - - -
#
function heisenberg_1D(N, Jx, Jy, Jz; x=0, y=0, z=0)
    H = PauliSum(N, Float64)
    for i in 0:N-1
        H += -Jx * Pauli(N, X=[i+1,(i+1)%(N)+1])
        H += -Jy * Pauli(N, Y=[i+1,(i+1)%(N)+1])
        H += -Jz * Pauli(N, Z=[i+1,(i+1)%(N)+1])
    end 
    for i in 1:N
        H += x * Pauli(N, X=[i])
        H += y * Pauli(N, Y=[i])
        H += z * Pauli(N, Z=[i])
    end 
    coeff_clip!(H)
    return H
end

# Helper: extract PauliBasis operators and coefficients from PauliSum
function extract_hamiltonian_coeffs_and_ops(H::PauliSum{N,T}) where {N,T}
    ops = PauliBasis{N}[]
    coeffs = Float64[]
    for (p, c) in H
        push!(ops, p)
        push!(coeffs, float(c))
    end
    return ops, coeffs
end

"""
    qdrift_propagator(H::PauliSum, t::Real, eps::Real;
                      seed::Union{Int,Nothing}=nothing,
                      plot::Bool=true, print_selection::Bool=true)

Perform a qDRIFT approximation for e^{i H t} where H is a PauliSum.

Returns a NamedTuple with:
  V         = simulated qDRIFT unitary (matrix)
  U_exact   = exact unitary exp(i * t * H) (matrix)
  sample    = vector of sampled indices (1-based)
  probs     = sampling distribution p_j
  λ         = sum(abs(h_j))
  τ         = per-step angle t*λ/N
  N         = number of samples used
  ops, coeffs = original lists of PauliBasis and coefficients
"""
function qdrift_propagator(ket, o::PauliSum{N,T}, H::PauliSum{N,T}, 
                           thresh::Real, tot_measurements::Int,
                           t::Real, eps::Real;
                           seed::Union{Int,Nothing}=nothing,
                           plot::Bool=true, print_selection::Bool=true) where {N,T}

    # Extract Pauli strings and coefficients
    ops, coeffs = extract_hamiltonian_coeffs_and_ops(H)
    @printf("Hamiltonian has %.2f terms \n", length(coeffs))
    L = length(coeffs)
    if L == 0
        error("Hamiltonian has no terms.")
    end

    # λ = sum_j |h_j| and probabilities p_j = |h_j| / λ
    λ = sum(abs.(coeffs))
    if λ == 0.0
        error("Sum of absolute coefficients is zero (λ=0).")
    end
    probs = abs.(coeffs) ./ λ

    # Number of samples (N); 
    Nsamples = ceil(Int, 2 * (λ * t)^2 / eps)
    if Nsamples < 1
        Nsamples = 1
    end

    # Set RNG seed if provided
    if seed !== nothing
        Random.seed!(seed)
    end

    # i.i.d. sampling of indices 1..L with replacement according to probs
    sample_list = sample(1:L, Weights(probs), Nsamples; replace=true)

    # Per-step angle tau = t * λ / N
    τ = t * λ / Nsamples

    #
    #* * * QSP setup * * *
    #
    Wt = deepcopy(o)        # evolve W ≡ U*OU
    W  = deepcopy(o)        # initial operator 
    rCtvals = Vector{Float64}([])# vector for C(t) values storing
    iCtvals = Vector{Float64}([])

    # t = 0 
    WW = W*W
    expval0 = expectation_value(WW, ket)

    C0real = real(expval0)
    C0imag = imag(expval0)

    #push!(rCtvals, C0real)
    #push!(iCtvals, C0imag)

    # Number of sampling steps
    N_tau = length(sample_list)

    # Desired number of measurement points (coarse-grain the timeline for plotting)
    N_meas = min(tot_measurements, N_tau)   # do not request more measurements than steps

    # Build measurement indices evenly spaced between 1 and N_tau (inclusive)
    # Use LinRange then round to Int to avoid zero/step problems.
    meas_lst = unique(round.(Int, LinRange(1, N_tau, N_meas)))

    # Corresponding time grid:
    time_grid =  collect(range(0.0, stop=t, length=N_meas))
    #time_grid =  collect(range(0.0, stop=N_tau * dt/lamb, length=N_meas))

    sort!(meas_lst)             # ensure ascending order
    # optional: show how many actual measurement points we will have
    @printf("Will take %d measurements at indices (first 20): %s\n", length(meas_lst),
            string(meas_lst[1:min(end,20)]))

    # prepare counters
    meas_idx = 1
    next_meas = meas_lst[meas_idx]
    sample_idx = 1

    # Perform C(t) estimation using qDRIFT
    for idx in sample_list
        Pi = ops[idx]
        s = sign(coeffs[idx])              # sign(h_j)
        theta = 2 * τ * s                  #Angles -- keep if evolve! expects this convention
        pb = PauliBasis(Pi)
        evolve!(Wt, pb, theta)             # in-place evolve the operator Wt
        coeff_clip!(Wt, thresh=thresh)
        WWt = W * Wt                       # OTOC-like product

        # Check whether to measure at this step
        if sample_idx == next_meas
            # Measurements:
            expval = expectation_value(WWt, ket) # Contraction with reference ket
            println("C(T) MEAS : ", expval)
            Ctreal = real(expval)
            Ctimag = imag(expval)
            push!(rCtvals, Ctreal)
            push!(iCtvals, Ctimag)

            # advance to next measurement index (if any)
            meas_idx += 1
            if meas_idx <= length(meas_lst)
                next_meas = meas_lst[meas_idx]
            else
                # no more measurements;
                next_meas = typemax(UInt128) # sentinel that will never be reached
            end
        end

        sample_idx += 1
    end

    # Print selection if requested
    if print_selection
        println("=== qDRIFT Selection (first 200 shown) ===")
        println(sample_list[1:min(end,200)])
        println("... (total samples = $Nsamples)")
    end

    # Plot distribution if requested
    if plot
        plt_bar = bar!(probs, legend=false, xlabel="Term index j", ylabel="p_j",
            title = @sprintf("qDRIFT sampling distribution (λ = %.6g)", λ))
        savefig(plt_bar, "qdrift_sampling_distribution.pdf")
    end

    return (sample=sample_list, probs=probs, time_grid=time_grid,
            λ=λ, τ=τ, N=Nsamples, RCt=rCtvals, ICt=iCtvals)
end

# Wrapper to average over multiple runs
function averaged_qdrift(n_runs, ket, o::PauliSum{N,T}, H::PauliSum{N,T}, 
                           thresh::Real, tot_measurements::Int,
                           t::Real, eps::Real) where {N,T}
    sum_RCt = zeros(tot_measurements)
    sum_ICt = zeros(tot_measurements)
    t_grid = zeros(tot_measurements)
    
    for r in 1:n_runs
        res = qdrift_propagator(ket, o, H, thresh, tot_measurements, t, eps)
        sum_RCt .+= res.RCt
        sum_ICt .+= res.ICt
        t_grid = res.time_grid
    end
    
    return sum_RCt ./ n_runs, sum_ICt ./ n_runs, t_grid
end

# demo parameters
Jx, Jy, Jz = 1.0, 1.0, 2.0
Nqubits = 4
ket = Ket(Nqubits, 1)
#ket, _ = string_to_ket("1000")
#ket, _ = string_to_ket("10000000000000000000")
H = heisenberg_1D(Nqubits, Jx, Jy, Jz)
t = 2.5
eps = 0.1
reps = 1
thresh = 1e-4 # Evolution threshold for evolve
n_meas = 100 #Number of measurements

o = Pauli(Nqubits, X=[1])
o = PauliSum(o)
# 
#res = qdrift_propagator(ket, o, H, thresh, n_meas, t, eps; seed=666, plot=false, print_selection=true)
#rRES = res.RCt
#iRES = res.ICt
#n_intervals = length(res.sample)
#dt = res.τ
#lamb = res.λ
#tgrid = collect(range(0.0, stop=n_intervals * dt/lamb, length=n_meas))
#tgrid = res.time_grid
t1 = time()
rRES, iRES, time_grid = averaged_qdrift(reps, ket, o, H, thresh, n_meas, t, eps)

elapsed_time = time() - t1
println("Elapsed time: ", elapsed_time, " seconds")

println("TIME GRID : ", tgrid)
# Number of snapshots actually returned
nsnap = length(rRES)
println("* * * * Number of snapshots collected: $nsnap")

# Print C(t) results
plt = plot(tgrid, rRES, lw=2, seriestype=:scatter,
           label="Re(C(t), th=$thresh")
plt = plot!(tgrid, iRES, lw=2, seriestype=:scatter,
           label="Im(C(t), th=$thresh")

# Read file exact result for comparison
lines = readlines("/Users/admin/VSCProjects/UnitaryPruning/QSP_XXZ_1D/exact_4Q_XXZ.txt")
lines = lines[2:end]

parsed = [parse.(Float64, split(line)) for line in lines]

t_exact   = getindex.(parsed, 1)
ReC_exact = getindex.(parsed, 2)
ImC_exact = getindex.(parsed, 3)
# Plot exact curve
plot!(t_exact, ReC_exact, label="Re C exact", lw=2)
plot!(t_exact, ImC_exact, label="Im C exact", lw=2)

xlabel!(plt, "Time"); ylabel!(plt, "< O(0)O(t) >")
title!(plt, "N=$Nqubits, J=$Jx, Jz=$Jz")

savefig(plt, "qdrift_QSP_XXZ_4Q_1D_eps=$eps.pdf") 