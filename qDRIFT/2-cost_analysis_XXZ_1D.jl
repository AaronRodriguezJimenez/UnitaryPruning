"""
 Here we compare estimates for qDRIFT without performing calculations
    given an epsilon, compute the number of Meas/run, lambda and tau
"""
#
#
using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random
using StatsBase

#
# - - - Clipping functions - - -
#
function coeff_clip!(ps::KetSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter!(p->abs(p.second) > thresh, ps)
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
    return filter(p->abs(p.second) > thresh, ps)
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
# qdrift_propagator: use an explicit rng (or seed) for all random choices
function qdrift_estimations(ket, o::PauliSum{N,T}, H::PauliSum{N,T},
                           thresh::Real, tot_measurements::Int,
                           t::Real, eps::Real;
                           rng::AbstractRNG = Random.default_rng(),
                           seed::Union{Integer,Nothing}=nothing,
                           plot::Bool=true, print_selection::Bool=true) where {N,T}

    # If caller passed an explicit seed, override rng with a seeded MersenneTwister.
    # This makes reproducing a run trivial by passing the same seed later.
    if seed !== nothing
        rng = MersenneTwister(Int(seed))
    end

    # Extract Pauli strings and coefficients
    ops, coeffs = extract_hamiltonian_coeffs_and_ops(H)
    #@printf("Hamiltonian has %d terms \n", length(coeffs))
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
    Nsamples = max(1, ceil(Int, 2 * (λ * t)^2 / eps))
    #println("Number of Samples: ", Nsamples)
    # i.i.d. sampling of indices 1..L with replacement according to probs using provided rng
    sample_list = sample(rng, 1:L, Weights(probs), Nsamples; replace=true)

    # Per-step angle tau = t * λ / N
    τ = t * λ / Nsamples

    # Measurement scheduling
    N_tau = length(sample_list)
    N_meas = min(tot_measurements, N_tau)
    meas_lst = unique(round.(Int, LinRange(1, N_tau, N_meas)))
    sort!(meas_lst)
    #@printf("Will take %d measurements at indices (first 20): %s\n",
    #        length(meas_lst), string(meas_lst[1:min(end,20)]))

    if plot
        plt_bar = bar!(probs, legend=false, xlabel="Term index j", ylabel="p_j",
            title = @sprintf("qDRIFT sampling distribution (λ = %.6g)", λ))
        savefig(plt_bar, "qdrift_sampling_distribution.pdf")
    end

    # If the provider RNG was a MersenneTwister seeded from a seed variable it will be reproducible.
    # Return the seed if it is a MersenneTwister, otherwise return nothing.
    used_seed = isa(rng, MersenneTwister) ? copy(rng.seed) : nothing

    return (sample=sample_list, nsamples=Nsamples, λ=λ, τ=τ, seed=used_seed)
end

# demo parameters
Jx, Jy, Jz = 1.0, 1.0, 2.0
Nqubits = 100
ket = Ket(Nqubits, 1)
#ket, _ = string_to_ket("1000")
#ket, _ = string_to_ket("10000000000000000000")
H = heisenberg_1D(Nqubits, Jx, Jy, Jz)
t = 2.5
reps = 1 
thresh = 1e-4 # Evolution threshold for evolve
n_meas = 100 #Number of measurements

o = Pauli(Nqubits, X=[1])
o = PauliSum(o)

s = rand(RandomDevice(), UInt32)      # RandomDevice() uses OS entropy
seed = Int(s)
rng = MersenneTwister(seed)


eps_lst = [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.30, 
           0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 
           0.85, 0.9, 0.95, 1.0, 1.5, 2.0, 2.5, 3.0]
println("Eps   Meas/run   lambda   tau")
for eps in eps_lst
    # Perform qDRIF estimation
    res = qdrift_estimations(ket, o, H, thresh, n_meas, t, eps;
                                rng=rng, plot=false, print_selection=false)
    @printf("%.4f   %d    %.6f    %.6f\n", eps,  res.nsamples, res.λ, res.τ)
    
end
