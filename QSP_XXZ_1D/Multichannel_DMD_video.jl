using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots
using Printf
using Random

# Here we attempt to track the dynamics of PP weight profiles using multichannel DMD / MODMD.
# The idea is to treat the weight profile at each time as a multichannel snapshot, and apply DMD to this multichannel time series.
# This is a proof-of-concept to see if DMD can extract meaningful modes from the weight dynamics,
# which could potentially be used for pruning or understanding operator growth.
# The code is structured as follows:
# 1. Multichannel DMD implementation (fit_multichannel_dmd and related functions)
# 2. A toy example with synthetic multichannel data (demo_multichannel_dmd)
# 3. Helpers for converting PP weight snapshots into multichannel format and running DMD on them (run_pp_multichannel_dmd)
# 4. Simple plotting functions for visualizing the multichannel snapshots and DMD modes.
# 
# Each channel, tracks the coefficient of a given pauli weight in the operator expansion. 
#The DMD modes then represent coherent patterns of how these weights evolve together over time.
#

# ============================================================
# Helpers
# ============================================================
coeff_clip!(ps; thresh=1e-16) = filter!(p -> abs(p.second) > thresh, ps)
weight(p::PauliBasis) = count_ones(p.x | p.z)
neighbor(site::Int, N::Int) = site == N ? 1 : site + 1

# ------------------------------------------------------------
# Hamiltonian
# ------------------------------------------------------------
function rectanglebricktopology(nx::Integer, ny::Integer)
    # LinearIndices automatically maps (x, y) coordinates to flat 1D indices
    LI = LinearIndices((nx, ny))

    # Layer A: Horizontal edges starting at odd columns (1, 3, 5...)
    layer_A = [(LI[x, y], LI[x+1, y]) for x in 1:2:(nx-1) for y in 1:ny]

    # Layer B: Horizontal edges starting at even columns (2, 4, 6...)
    layer_B = [(LI[x, y], LI[x+1, y]) for x in 2:2:(nx-1) for y in 1:ny]

    # Layer C: Vertical edges starting at odd rows (1, 3, 5...)
    layer_C = [(LI[x, y], LI[x, y+1]) for x in 1:nx for y in 1:2:(ny-1)]

    # Layer D: Vertical edges starting at even rows (2, 4, 6...)
    layer_D = [(LI[x, y], LI[x, y+1]) for x in 1:nx for y in 2:2:(ny-1)]

    # Concatenate the sublayers into a single continuous Vector
    return vcat(layer_A, layer_B, layer_C, layer_D)
end

function get_ordered_generators(Nx, Ny, J, h)
    N = Nx * Ny
    generators = []
    angles = []

    topology = rectanglebricktopology(Nx, Ny)

    # 1. ZZ Interactions (ordered strictly by brick topology layers)
    for (i, j) in topology
        push!(generators, PauliBasis(Pauli(N, Z=[i, j])))
        push!(angles, J) # Keeping the J/4 scaling from your previous convention
    end

    # 2. Transverse Field (X on all sites)
    for i in 1:N
        push!(generators, PauliBasis(Pauli(N, X=[i])))
        push!(angles, h) # Keeping the h/2 scaling
    end

    return generators, angles
end

function Ising_rectangle(Nx, Ny, J, h)
    N = Nx * Ny
    H = PauliSum(N, Float64)

    topology = rectanglebricktopology(Nx, Ny)

    # 1. ZZ Interactions (ordered strictly by brick topology layers)
    for (i, j) in topology        
        H += -J * Pauli(N, Z=[i, j])
    end

    # 2. Transverse Field (X on all sites)
    for i in 1:N
        H += h * Pauli(N, X=[i])
    end

    return H
end


function heisenberg_1D(N, Jx, Jy, Jz; x=0.0, y=0.0, z=0.0)
    H = PauliSum(N, Float64)

    for site in 1:N
        nxt = neighbor(site, N)
        H += -Jx * Pauli(N, X=[site, nxt])
        H += -Jy * Pauli(N, Y=[site, nxt])
        H += -Jz * Pauli(N, Z=[site, nxt])
    end

    for site in 1:N
        if x != 0.0
            H += x * Pauli(N, X=[site])
        end
        if y != 0.0
            H += y * Pauli(N, Y=[site])
        end
        if z != 0.0
            H += z * Pauli(N, Z=[site])
        end
    end

    return coeff_clip!(H)
end


# ============================================================
# Hamiltonian coefficients
# ============================================================
struct HamiltonianCache{N}
    ops::Vector{PauliBasis{N}}
    coeffs::Vector{Float64}
end

function extract_coeffs_and_ops(H::PauliSum{N, T}) where {N, T}
    ops = PauliBasis{N}[]
    coeffs = Float64[]
    sizehint!(ops, length(H))
    sizehint!(coeffs, length(H))

    for (p, c) in H
        push!(ops, p)
        push!(coeffs, float(c))
    end

    return HamiltonianCache{N}(ops, coeffs)
end

"""
 weight_profile!
Compute the weight profile of a PauliSum O and store it in the provided vector prof.
:l2 is the default and tracks the normalized squared coefficient norm in each weight sector.
: real is the real part of the coefficients,
 :imag is the imaginary part, 
 :abs is the absolute value, 
 :complex is the raw complex coefficient sum in each weight sector.
"""
function weight_profile!(prof::AbstractVector{Float64}, O::PauliSum{N, T};
    kind::Symbol = :l2, normalize::Bool = true) where {N, T}

    fill!(prof, 0.0)

    for (p, c) in O
        w = weight(p) + 1
        if kind == :l2
            prof[w] += abs2(c)
        elseif kind == :abs
            prof[w] += abs(c)
        elseif kind == :real
            prof[w] += real(c)
        elseif kind == :imag
            prof[w] += imag(c)
        elseif kind == :complex
            prof[w] += c
        else
            error("Unknown kind = $kind")
        end
    end

    if normalize && kind == :l2
        s = sum(prof)
        s > 0 && (prof ./= s)
    end

    return prof
end

# ============================================================
# Pauli propagation step
# ============================================================

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


# ============================================================
# Evolution driver
# ============================================================
"""
Evolve an operator under the Trotterized Hamiltonian and store:

1. full weight-grouped operator snapshots
2. scalar weight channels for DMD
   - real
   - imag
   - energy
   - abs

Returns a NamedTuple with all containers.
"""
function channel_evolution(ket, o::PauliSum{N, T}, Hcache::HamiltonianCache{N}, n_intervals::Int,
                           dt::Real; thresh::Float64 = 1e-3, kind::Symbol = :l2, normalize::Bool = false,
                           track_corr::Bool = false,
                           ) where {N, T}

    O0 = deepcopy(o)
    Ot = deepcopy(o)

    ops = Hcache.ops
    coeffs = Hcache.coeffs
    nt = length(coeffs)

    @printf("Total Pauli rotations: %d\n", nt * n_intervals)

    # Preallocate snapshot matrix: rows = channels, cols = time
    snapshots = Matrix{Float64}(undef, N + 1, n_intervals + 1)
    tmp_prof = zeros(Float64, N + 1)

    # Initial snapshot
    weight_profile!(tmp_prof, Ot; kind=kind, normalize=(kind == :l2))
    snapshots[:, 1] .= tmp_prof

    corr_real = track_corr ? Vector{Float64}(undef, n_intervals + 1) : Float64[]
    corr_imag = track_corr ? Vector{Float64}(undef, n_intervals + 1) : Float64[]

    if track_corr
        c0 = expectation_value(O0 * Ot, ket)
        corr_real[1] = real(c0)
        corr_imag[1] = imag(c0)
    end

    for step in 1:n_intervals
        #accumulated_error = 0.0 + 0.0im

        for j in 1:nt
            θ = 2 * dt * coeffs[j]
            evolve!(Ot, ops[j], θ)

            coeff_clip!(Ot; thresh=1e-12)
            #before = expectation_value(O0 * Ot, ket)

            coeff_clip!(Ot; thresh=thresh)
            #after = expectation_value(O0 * Ot, ket)

            #accumulated_error += after - before

        end

        #display(Ot)

        if kind == :l2
             weight_profile!(tmp_prof, Ot; kind=kind, normalize=(kind == :l2))
        else
            weight_profile!(tmp_prof, Ot; kind=kind, normalize=false)
        end

        snapshots[:, step + 1] .= tmp_prof

        if track_corr
            c = expectation_value(O0 * Ot, ket) + accumulated_error
            corr_real[step + 1] = real(c)
            corr_imag[step + 1] = imag(c)
        end
    end

    tgrid = collect(0.0:dt:n_intervals * dt)

    return (
        tgrid = tgrid,
        snapshots = snapshots,
        corr_real = corr_real,
        corr_imag = corr_imag,
    )
end

# ============================================================
# Multichannel DMD / MODMD
# ============================================================
struct MultiChannelDMDResult
    A_ls::Matrix{ComplexF64}
    A_tilde::Matrix{ComplexF64}
    modes::Matrix{ComplexF64}
    evals::Vector{ComplexF64}
    amplitudes::Vector{ComplexF64}
    singular_values::Vector{Float64}
    residual_rel::Float64
    delay::Int
    dt::Float64
    channel_names::Vector{String}
end

"""
Convert channel-wise time series into a matrix X ∈ R^{I×T},
where each row is one channel and each column is one time point.
"""
function snapshots_to_matrix(snapshots::AbstractVector)
    @assert !isempty(snapshots) "snapshots cannot be empty"

    T = length(snapshots[1])
    @assert all(length(s) == T for s in snapshots) "all channels must have same time length"

    X = reduce(hcat, snapshots)'   # rows = channels, cols = time
    return X
end

"""
Build Takens-embedded multichannel Hankel matrices.

Input:
    X[:, t] = x_t ∈ R^I, with T total time samples

For delay = d, build:
    z_t = [x_t; x_{t+1}; ...; x_{t+d-1}] ∈ R^(dI)

Return:
    X1 = [z_1 z_2 ... z_{T-d}]
    X2 = [z_2 z_3 ... z_{T-d+1}]

So X1 and X2 have size (dI, T-d).
"""
function build_multichannel_hankel(X::AbstractMatrix, delay::Int)
    I, T = size(X)
    @assert delay >= 1 "delay must be >= 1"
    @assert T > delay "need at least delay+1 snapshots"

    ncols = T - delay
    X1 = zeros(eltype(X), I * delay, ncols)
    X2 = zeros(eltype(X), I * delay, ncols)

    for col in 1:ncols
        @views X1[:, col] .= vec(X[:, col:col + delay - 1])
        @views X2[:, col] .= vec(X[:, col + 1:col + delay])
    end

    return X1, X2
end

"""
Fit multichannel DMD / MODMD.

Arguments
---------
snapshots : Vector of length T, each entry a vector of length I
delay     : Hankel embedding depth
r         : optional truncation rank
tol       : relative SVD cutoff if r is not supplied
dt        : time step between snapshots
"""
function fit_multichannel_dmd(
    snapshots::Vector{<:AbstractVector};
    delay::Int = 2,
    r::Union{Nothing,Int} = nothing,
    tol::Real = 1e-12,
    dt::Real = 1.0,
    channel_names::Vector{String} = String[],
)
    X = snapshots_to_matrix(snapshots)
    rhs, lhs = build_multichannel_hankel(X, delay)

    # Full least-squares operator in embedded space
    A_ls = Matrix(lhs * pinv(rhs))
    residual_rel = norm(lhs - A_ls * rhs) / max(norm(lhs), eps())

    # SVD truncation
    F = svd(rhs; full=false)
    U, s, V = F.U, F.S, F.V

    @assert !isempty(s) "No singular values were found"

    if r === nothing
        keep = findall(>(tol * s[1]), s)
        @assert !isempty(keep) "All singular values were truncated; reduce tol."
        r = keep[end]
    else
        r = min(r, length(s))
    end

    Ur = U[:, 1:r]
    sr = s[1:r]
    Vr = V[:, 1:r]

    Sinv = Diagonal(1.0 ./ sr)

    # Reduced operator
    A_tilde = Matrix(Ur' * lhs * Vr * Sinv)

    eig = eigen(A_tilde)
    λ = eig.values
    W = eig.vectors

    # DMD modes in the embedded space
    Φ = Matrix(lhs * Vr * Sinv * W)

    # Initial amplitudes
    b = Φ \ complex.(rhs[:, 1])

    if isempty(channel_names)
        channel_names = ["ch$(i)" for i in 1:I]
    end

    return MultiChannelDMDResult(
        A_ls,
        A_tilde,
        Φ,
        λ,
        b,
        s,
        residual_rel,
        delay,
        float(dt),
        channel_names,
    )
end

"""
Compact summary sorted by |amplitude|.
"""
function print_multichannel_dmd_summary(res::MultiChannelDMDResult; topk::Int = 6)
    λ = res.evals
    amp = res.amplitudes
    dt = res.dt

    growth = log.(abs.(λ)) ./ dt
    freq = angle.(λ) ./ dt
    idx = sortperm(abs.(amp), rev=true)

    println("---- Multichannel DMD summary ----")
    println("delay embedding q = ", res.delay)
    println("relative LS residual = ", res.residual_rel)
    println("top singular values = ", res.singular_values[1:min(topk, length(res.singular_values))])

    println("\nDominant modes by |amplitude|:")
    for j in 1:min(topk, length(idx))
        i = idx[j]
        @printf("  mode %d\n", i)
        @printf("    λ         = %s\n", string(λ[i]))
        @printf("    |λ|       = %.6f\n", abs(λ[i]))
        @printf("    growth    = %.6f\n", growth[i])
        @printf("    frequency = %.6f\n", freq[i])
        @printf("    amplitude = %s\n", string(amp[i]))
    end
end

function print_dmd_summary(
    res::MultiChannelDMDResult;
    topk::Int = 6,
    nsnapshots::Int          # number of time snapshots used in the reconstruction
)
    λ = res.evals
    b = res.amplitudes
    dt = res.dt

    growth = log.(abs.(λ)) ./ dt
    freq   = angle.(λ) ./ dt

    modes = res.modes
    nsnapshots

    mode_norms = map(norm, eachcol(modes))

    scores = zeros(Float64, length(λ))
    for i in eachindex(λ)
        r = abs(λ[i])^2
        geom_sum = isapprox(r, 1.0; atol=1e-12) ? nsnapshots :
                   (1 - r^nsnapshots) / (1 - r)
        scores[i] = abs2(b[i]) * mode_norms[i]^2 * geom_sum
    end

    idx = sortperm(scores, rev=true)

    println("---- Multichannel DMD summary ----")
    println("delay embedding q = ", res.delay)
    println("relative LS residual = ", res.residual_rel)
    println("top singular values = ", res.singular_values[1:min(topk, length(res.singular_values))])

    println("\nDominant modes by contribution score:")
    for j in 1:min(topk, length(idx))
        i = idx[j]
        @printf("  mode %d\n", i)
        @printf("    score     = %.6e\n", scores[i])
        @printf("    λ         = %s\n", string(λ[i]))
        @printf("    |λ|       = %.6f\n", abs(λ[i]))
        @printf("    growth    = %.6f\n", growth[i])
        @printf("    frequency = %.6f\n", freq[i])
        @printf("    amplitude = %s\n", string(b[i]))
        @printf("    ||phi||   = %.6e\n", mode_norms[i])
    end
end

# ============================================================
# Simple plots
# ============================================================
function plot_multichannel_heatmap(
    snapshots;
    channel_labels=nothing,
    cmap=:haline,
)

    X = snapshots_to_matrix(snapshots)

    nchannels, nt = size(X)

    # Default labels
    if channel_labels === nothing
        channel_labels = ["w=$i" for i in 0:nchannels-1]
    end

    heatmap(
        0:nt-1,
        0:nchannels-1,
        X,

        xlabel = "time index",
        ylabel = "channel",
        title = "Multichannel Weight Dynamics",

        yticks = (0:nchannels-1, channel_labels),

        color = cmap,
        colorbar_title = "Coefficient values",

        framestyle = :box,
        aspect_ratio = :auto,
        dpi = 300,
    )
end

function plot_dmd_mode_shapes(res::MultiChannelDMDResult; nmodes::Int = 4)
    r = min(nmodes, size(res.modes, 2))
    x = 1:size(res.modes, 1)

    p = plot(
        xlabel = "embedded coordinate",
        ylabel = "|mode amplitude|",
        title = "DMD mode shapes",
        legend = :right,
    )

    for j in 1:r
        plot!(p, x, abs.(res.modes[:, j]), label = "mode $j", lw=2)
    end

    return p
end

# ===========================
# Check ups
# ===========================
"""
Visualize the learned Takens / MODMD operator A.

The matrix is shown together with block boundaries corresponding
to the Takens embedding structure.

Arguments
---------
A :
    Learned operator matrix of size (dI, dI)

I :
    Number of channels

d :
    Takens embedding dimension

Options
-------
title :
    Plot title

cmap :
    Colormap

show_blocks :
    Draw block boundaries

show_values :
    Overlay numerical values (only recommended for small matrices)

logscale :
    Plot log10(abs(A)+eps()) instead of abs(A)

companion_overlay :
    Highlight the expected companion-shift blocks.
"""
function plot_operator_matrix(
    A::AbstractMatrix,
    I::Int,
    d::Int;
    title::String = "Takens / MODMD Operator",
    cmap = :balance,
    show_blocks::Bool = true,
    show_values::Bool = false,
    logscale::Bool = false,
    companion_overlay::Bool = true,
)

    n1, n2 = size(A)

    @assert n1 == n2 "A must be square"
    @assert n1 == I*d "A size inconsistent with I*d"

    # What to visualize
    M = logscale ? log10.(abs.(A) .+ eps()) : abs.(A)

    p = heatmap(
        1:n2,
        1:n1,
        M,

        yflip = true,

        xlabel = "column index",
        ylabel = "row index",

        title = title,

        color = cmap,
        aspect_ratio = :equal,

        framestyle = :box,
        dpi = 300,

        colorbar_title = logscale ? "log10|A|" : "|A|",
    )

    # --------------------------------------------------------
    # Draw Takens block boundaries
    # --------------------------------------------------------

    if show_blocks
        for k in 1:d-1
            pos = k * I + 0.5

            vline!(p, [pos], color=:white, lw=2, alpha=0.8)
            hline!(p, [pos], color=:white, lw=2, alpha=0.8)
        end
    end

    # --------------------------------------------------------
    # Companion-structure overlay
    # --------------------------------------------------------

    if companion_overlay && d > 1

        # Expected identity-shift blocks:
        #
        # [ * * * ]
        # [ I 0 0 ]
        # [ 0 I 0 ]
        #
        for block in 2:d

            row0 = (block - 1) * I
            col0 = (block - 2) * I

            # rectangle around expected identity block
            xs = [col0 + 1, col0 + I, col0 + I, col0 + 1, col0 + 1]
            ys = [row0 + 1, row0 + 1, row0 + I, row0 + I, row0 + 1]

            plot!(
                p,
                xs,
                ys,
                color = :yellow,
                lw = 3,
                label = false,
            )
        end
    end

    # --------------------------------------------------------
    # Optional numerical annotations
    # --------------------------------------------------------

    if show_values && n1 <= 20
        for i in 1:n1
            for j in 1:n2
                annotate!(
                    p,
                    j,
                    i,
                    text(round(A[i,j], digits=2), 7, :black)
                )
            end
        end
    end

    return p
end

# MODES VISUALIZATION
########################################
"""
Reshape one embedded DMD mode into an I × d block matrix.

Assumes the embedding vector was built as:
    [x_t; x_{t+1}; ...; x_{t+d-1}]

So reshape(mode, I, d) returns:
    columns = delay blocks
    rows    = channels / weight sectors
"""
function reshape_mode_blocks(mode::AbstractVector, I::Int, d::Int)
    @assert length(mode) == I * d "mode length must equal I*d"
    return reshape(mode, I, d)
end

"""
Plot a single DMD mode reshaped into I × d blocks.

Arguments
---------
mode :
    One DMD mode vector of length I*d
I :
    Number of channels / weight sectors
d :
    Takens delay dimension

Options
-------
kind = :abs, :real, :imag, :phase
    What to visualize

channel_labels :
    Labels for the y-axis (weight sectors)

delay_labels :
    Labels for the x-axis (delay blocks)
"""
function plot_mode_blocks(
    mode::AbstractVector,
    I::Int,
    d::Int;
    kind::Symbol = :abs,
    channel_labels = nothing,
    delay_labels = nothing,
    title::String = "DMD mode in block form",
    cmap = :magma,
)

    M = reshape_mode_blocks(mode, I, d)

    Z = if kind == :abs
        abs.(M)
    elseif kind == :real
        real.(M)
    elseif kind == :imag
        imag.(M)
    elseif kind == :phase
        angle.(M)
    else
        error("Unknown kind = $kind. Use :abs, :real, :imag, or :phase.")
    end

    if channel_labels === nothing
        channel_labels = ["w$(i-1)" for i in 1:I]
    end
    if delay_labels === nothing
        delay_labels = ["τ$(j-1)" for j in 1:d]
    end

    heatmap(
        1:d,
        1:I,
        Z,
        yflip = true,
        xlabel = "delay block",
        ylabel = "weight sector",
        title = title,
        color = cmap,
        yticks = (1:I, channel_labels),
        xticks = (1:d, delay_labels),
        colorbar_title = string(kind),
        framestyle = :box,
        aspect_ratio = :auto,
        dpi = 300,
    )
end

function print_weight_channels(res, N; digits=6)
    channels = res.snapshots[1:N, :]
    tgrid = res.tgrid

    # Header
    @printf("%12s", "time")
    for w in 0:N-1
        @printf("%12s", "w$w")
    end
    println()

    # Rows
    for t in eachindex(tgrid)
        @printf("%12.*f", digits, tgrid[t])

        for w in 1:N
            @printf("%12.*f", digits, channels[w, t])
        end

        println()
    end
end

function plot_weight_channels(res, N)
    channels = res.snapshots[1:N, :]' # Extract first 7 channels and transpose
    labels = reshape(["Weight $i" for i in 0:N], 1, :) # Create labels dynamically

    plot(
        res.tgrid,
        channels,
        label=labels,
        xlabel="time",
        ylabel="L2-norm",
        title="PP weight channels over time",
        lw=2.5,
        xtickfontsize=20,
        ytickfontsize=20,
        guidefontsize=20,
        left_margin = 15Plots.mm,
        top_margin = 10Plots.mm,
        dpi = 300,
        legend=:best,
        size=(800, 800),

    )
end

using Plots

function plot_dmd_eigs_on_unit_circle(λ; scores=nothing, topk=6, title_str="DMD Eigenvalues")
    θ = range(0, 2π; length=400)

    p = scatter(
        real.(λ),
        imag.(λ),
        aspect_ratio = :equal,
        xlabel = "Re(λ)",
        ylabel = "Im(λ)",
        title = title_str,
        label = "DMD eigenvalues",
        markerstrokewidth = 0,
        markersize = 6,
        zcolor = scores,
        colorbar = scores === nothing ? false : true,
    )

    plot!(p, cos.(θ), sin.(θ), label = "unit circle", linewidth = 2)

    if scores !== nothing
        idx = sortperm(scores, rev=true)[1:min(topk, length(λ))]
        for i in idx
            annotate!(p, real(λ[i]), imag(λ[i]), text("$(i)", 8))
        end
    end

    return p
end

# = = =. = = = = =. = === = = == = =. = = = = =. = = = = = = =
#Hamiltonian and initial operator for testing
#N = 6
#Jx = 0.10
#Jy = 0.10
#Jz = 1.0
#ket = Ket(N, 1)
#o = PauliSum(Pauli(N, X=[1]))
#H = heisenberg_1D(N, Jx, Jy, Jz)


Nx = 3
Ny = 3
N= Nx * Ny
J = 1.0
h = 0.25
H = Ising_rectangle(Nx, Ny, J, h)

c_ind = (Nx ÷ 2 + 1) + (Ny ÷ 2) * Nx
o = PauliSum(Pauli(N, Z = [c_ind])) #Z_init
ket = Ket(N, 0)

Hcache = extract_coeffs_and_ops(H)

total_time = 50.0
dt = 0.1 #total_time / n_intervals
n_intervals = total_time / dt |> Int
thresh = 1e-4

# kind keys: :l2, :abs, :real, :imag, :complex determine what we track in the channels. 
res = channel_evolution(ket, o, Hcache, n_intervals, dt; thresh=thresh, kind=:l2,
                        normalize=false, track_corr=false)


# Matrix of snapshots: rows = channels (weights), cols = time steps
println("snapshots size = ", size(res.snapshots))
display(res.snapshots)

S = res.snapshots
t = res.tgrid

n_weights, n_times = size(S)
weights = 0:(n_weights - 1)

ymax = maximum(S) * 1.05

anim = @animate for k in 1:n_times
    bar(
        weights,
        S[:, k],
        xlabel = "Pauli weight",
        ylabel = "channel value",
        title = @sprintf("Weight profile at t = %.4f", t[k]),
        legend = false,
        ylim = (0, ymax),
        xlim = (minimum(weights) - 0.5, maximum(weights) + 0.5),
        size = (800, 500),
    )
end

gif(anim, "channel_weights.gif", fps = 20)