# Takens + DMD comparison benchmark
#
# Goals:
#   - Compare DMD+Takens behavior on structured vs random signals.
#   - Use the same analysis pipeline across all cases.
#   - Check whether PP operator-weight dynamics behave more like structured
#     dynamics or like random / shuffled controls.
#
# Assumes the PP functions/types already exist in the session:
#   - evolution_op
#   - heisenberg_1D
#   - fit_dmd
#   - weight_stats
#   - plot_weight_heatmap
#   - plot_dmd_modes_abs
#
# This script is intentionally independent from the production PP code.

using LinearAlgebra
using Statistics
using Printf
using Random
using Plots

# ------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------

mean_weight(w::AbstractVector) = sum((0:length(w)-1) .* w)

function weight_entropy(w::AbstractVector; ϵ::Float64=1e-15)
    p = clamp.(w, ϵ, 1.0)
    return -sum(p .* log.(p))
end

function dominant_rank(s::AbstractVector{<:Real}; energy::Float64=0.95)
    tot = sum(abs2, s)
    tot <= 0 && return 0
    acc = 0.0
    for (i, σ) in enumerate(s)
        acc += abs2(σ)
        if acc / tot >= energy
            return i
        end
    end
    return length(s)
end

function l2_relative(a::AbstractVector, b::AbstractVector)
    return norm(a .- b) / max(norm(b), eps())
end

function time_series_metrics(snapshots::Vector{<:AbstractVector})
    nt = length(snapshots)
    μ = zeros(Float64, nt)
    σ2 = zeros(Float64, nt)
    ent = zeros(Float64, nt)

    for t in 1:nt
        μ[t], σ2[t] = weight_stats(snapshots[t])
        ent[t] = weight_entropy(snapshots[t])
    end

    return (mean_weight = μ, variance = σ2, entropy = ent)
end

# ------------------------------------------------------------
# Delay embedding
# ------------------------------------------------------------

"""
Takens-style delay embedding.
Each column is [x_t; x_{t+1}; ...; x_{t+q-1}].
"""
function delay_embed(snapshots::Vector{<:AbstractVector}, q::Int)
    m = length(snapshots)
    d = length(snapshots[1])
    @assert q >= 1 "q must be at least 1"
    @assert m >= q "need at least q snapshots"

    ncols = m - q + 1
    X = zeros(Float64, d * q, ncols)

    for k in 1:ncols
        for j in 1:q
            X[(j-1)*d + 1 : j*d, k] .= snapshots[k + j - 1]
        end
    end

    return X
end

embed_snapshots(snapshots::Vector{<:AbstractVector}; q::Int=1) = q == 1 ? hcat(snapshots...) : delay_embed(snapshots, q)

function embed_fit_summary(snapshots::Vector{<:AbstractVector}; q::Int=1, r::Union{Nothing,Int}=2)
    X = embed_snapshots(snapshots; q=q)
    res = fit_dmd(X; r=r)
    X1 = X[:, 1:end-1]
    X2 = X[:, 2:end]
    rel_fit = norm(X2 - res.A_ls * X1) / max(norm(X2), eps())
    rank95 = dominant_rank(res.singular_values; energy=0.95)
    rank99 = dominant_rank(res.singular_values; energy=0.99)
    return (
        X = X,
        res = res,
        rel_fit = rel_fit,
        rank95 = rank95,
        rank99 = rank99,
    )
end

# ------------------------------------------------------------
# Signal generators
# ------------------------------------------------------------

function synthetic_signal(m::Int; ω1::Float64=0.12, ω2::Float64=0.31)
    t = collect(0:m-1)
    x1 = sin.(ω1 .* t) .+ 0.3 .* cos.(ω2 .* t)
    x2 = cos.(ω1 .* t) .- 0.2 .* sin.(ω2 .* t)
    x3 = 0.5 .* sin.(0.5 .* ω1 .* t .+ 0.1)
    return [Float64[x1[i], x2[i], x3[i]] for i in 1:m]
end

function white_noise_signal(m::Int, d::Int; seed::Int=1)
    rng = MersenneTwister(seed)
    return [randn(rng, d) for _ in 1:m]
end

"""
 Correlated random signal (AR process)

Pure white noise is almost too easy. 
A more realistic null model is correlated stochastic dynamics
AR(1) formula: x_t = α * x_{t-1} + noise
"""
function ar1_signal(m::Int, d::Int; α::Float64=0.9, seed::Int=1)
    rng = MersenneTwister(seed)
    x = zeros(Float64, d)
    out = Vector{Vector{Float64}}(undef, m)
    for t in 1:m
        x = α .* x .+ 0.1 .* randn(rng, d)
        out[t] = copy(x)
    end
    return out
end

"""
 Take PP snapshots and shuffle time order
"""
function shuffled_snapshots(snapshots::Vector{<:AbstractVector}; seed::Int=1)
    rng = MersenneTwister(seed)
    idx = randperm(rng, length(snapshots))
    return snapshots[idx]
end

# ------------------------------------------------------------
# Benchmark container
# ------------------------------------------------------------

struct BenchmarkResult
    name::String
    q_values::Vector{Int}
    fit_error::Vector{Float64}
    rank95::Vector{Int}
    rank99::Vector{Int}
    sv_first::Vector{Float64}
    sv_second::Vector{Float64}
end

function benchmark_signal(name::String, snapshots::Vector{<:AbstractVector}; q_values=1:10, r::Union{Nothing,Int}=2)
    qv = collect(q_values)
    fit_error = Float64[]
    rank95 = Int[]
    rank99 = Int[]
    sv_first = Float64[]
    sv_second = Float64[]

    for q in qv
        if length(snapshots) < q + 1
            @printf("Skipping q=%d for %s (not enough snapshots)\n", q, name)
            push!(fit_error, NaN)
            push!(rank95, 0)
            push!(rank99, 0)
            push!(sv_first, NaN)
            push!(sv_second, NaN)
            continue
        end

        out = embed_fit_summary(snapshots; q=q, r=r)
        push!(fit_error, out.rel_fit)
        push!(rank95, out.rank95)
        push!(rank99, out.rank99)
        push!(sv_first, isempty(out.res.singular_values) ? NaN : out.res.singular_values[1])
        push!(sv_second, length(out.res.singular_values) >= 2 ? out.res.singular_values[2] : NaN)
    end

    return BenchmarkResult(name, qv, fit_error, rank95, rank99, sv_first, sv_second)
end

# ------------------------------------------------------------
# PP-specific wrappers
# ------------------------------------------------------------

function run_pp_snapshots(ket, o, H, n_intervals::Int, dt::Real; threshold::Float64=1e-10)
    rRES, iRES, tgrid, w_snapshots = evolution_op(ket, o, H, n_intervals, dt; thresh=threshold)
    return (
        rRES = rRES,
        iRES = iRES,
        tgrid = tgrid,
        w_snapshots = w_snapshots,
        metrics = time_series_metrics(w_snapshots),
    )
end

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

function plot_benchmark_metric(results::Vector{BenchmarkResult}; metric::Symbol=:fit_error)
    p = plot(
        xlabel = "embedding q",
        ylabel = string(metric),
        title = "Takens + DMD benchmark",
        legend = :right,
        xscale = :linear,
    )

    for r in results
        vals = getproperty(r, metric)
        plot!(p, r.q_values, vals, marker=:circle, lw=2, label=r.name)
    end

    return p
end

function plot_benchmark_ranks(results::Vector{BenchmarkResult})
    p1 = plot(
        xlabel = "embedding q",
        ylabel = "rank for 95% energy",
        title = "Effective rank (95%)",
        legend = :right,
    )
    p2 = plot(
        xlabel = "embedding q",
        ylabel = "rank for 99% energy",
        title = "Effective rank (99%)",
        legend = :right,
    )

    for r in results
        plot!(p1, r.q_values, r.rank95, marker=:circle, lw=2, label=r.name)
        plot!(p2, r.q_values, r.rank99, marker=:circle, lw=2, label=r.name)
    end

    return p1, p2
end

function plot_singular_values_vs_q(results::Vector{BenchmarkResult})
    p = plot(
        xlabel = "embedding q",
        ylabel = "leading singular value",
        title = "Leading singular values vs embedding dimension",
        legend = :right,
    )

    for r in results
        plot!(p, r.q_values, r.sv_first, marker=:circle, lw=2, label="$(r.name): σ₁")
        plot!(p, r.q_values, r.sv_second, marker=:square, lw=2, label="$(r.name): σ₂")
    end

    return p
end

# ------------------------------------------------------------
# Main benchmark
# ------------------------------------------------------------

function main_benchmark()
    println("========================================")
    println("Takens + DMD comparison benchmark")
    println("========================================")

    q_values = 1:12
    r = 2

    # Structured synthetic control
    structured = synthetic_signal(250)

    # Null controls
    white = white_noise_signal(250, 3; seed=2)
    ar1 = ar1_signal(250, 3; α=0.9, seed=3)

    # Shuffled version of the structured signal
    shuffled = shuffled_snapshots(structured; seed=4)

    # PP data
    # Replace these with your current setup if needed.
    N = 6
    ket = Ket(N, 1)
    o = PauliSum(Pauli(N, X=[3], Z=[1]))
    H = heisenberg_1D(N, 1.0, 1.0, 1.0; x=0.0, y=0.0, z=0.0)
    pp = run_pp_snapshots(ket, o, H, 100, 0.5; threshold=1e-10)
    shuffled_pp = shuffled = shuffled_snapshots(pp.w_snapshots; seed=4)

    # Benchmark all cases
    cases = BenchmarkResult[]
    push!(cases, benchmark_signal("synthetic", structured; q_values=q_values, r=r))
    push!(cases, benchmark_signal("white noise", white; q_values=q_values, r=r))
    push!(cases, benchmark_signal("AR(1)", ar1; q_values=q_values, r=r))
    push!(cases, benchmark_signal("shuffled synthetic", shuffled; q_values=q_values, r=r))
    push!(cases, benchmark_signal("PP weight snapshots", pp.w_snapshots; q_values=1:min(12, length(pp.w_snapshots)-1), r=r))
    push!(cases, benchmark_signal("PP shuffled snapshots",shuffled_pp; q_values=1:min(12, length(pp.w_snapshots)-1), r=r))

    println("\n--- Summary ---")
    for c in cases
        println("Case: ", c.name)
        println("  q values    : ", c.q_values)
        println("  fit error   : ", c.fit_error)
        println("  rank95      : ", c.rank95)
        println("  rank99      : ", c.rank99)
    end

    # Plots
    p_fit = plot_benchmark_metric(cases; metric=:fit_error)
    p_rank95, p_rank99 = plot_benchmark_ranks(cases)
    p_sv = plot_singular_values_vs_q(cases)

    display(p_fit)
    display(p_rank95)
    display(p_rank99)
    display(p_sv)

    # PP-specific diagnostics
    display(plot_weight_heatmap(pp.w_snapshots; tgrid=pp.tgrid))

    return (; cases, pp)
end

function signals()
    println("========================================")
    println("PLOT SIGNALS")
    println("========================================")
    
    m = 250
    labels = ["x1" "x2" "x3"] # Horizontal matrix for series labeling

    # 1. Generate Signals (Vector of 3-element Vectors)
    structured_raw = synthetic_signal(m)
    white_raw      = white_noise_signal(m, 3; seed=2)
    ar1_raw        = ar1_signal(m, 3; α=0.9, seed=3)
    shuffled_raw   = shuffled_snapshots(structured_raw; seed=4)

    # 2. Conversion helper 
    # Transforms Vector{Vector{Float64}} into a Matrix{Float64} of size (m, 3)
    prepare(s) = reduce(hcat, s)'

    # 3. Plotting
    # We plot all 3 components on the same subplot for each signal type
    p1 = plot(prepare(structured_raw), title="Structured Signal", label=labels)
    p2 = plot(prepare(white_raw),      title="White Noise",       label=false)
    p3 = plot(prepare(ar1_raw),        title="AR(1) Process",     label=false)
    p4 = plot(prepare(shuffled_raw),   title="Shuffled Snapshots", label=false)

    # Combine into a 4-row stack
    combined_plot = plot(p1, p2, p3, p4, 
        layout=(4, 1), 
        size=(900, 1100), 
        link=:x,           # Links the x-axis for easier scrolling/comparison
        margin=5Plots.mm,
        ylabel="Value")

    display(combined_plot)
    return combined_plot

end

# run 
results = main_benchmark()
signals()