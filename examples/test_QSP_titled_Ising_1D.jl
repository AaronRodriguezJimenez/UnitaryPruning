using LinearAlgebra
using UnitaryPruning
using PauliOperators
using Statistics
using Plots

"""
 This example is based on the model expossed at the results of Fig.6 in
  https://arxiv.org/abs/2505.21606

   In this case we use the so-called tilted-field Ising Hamiltonian consisting of Z, X and ZZ terms.
 Results from the paper use a time step dt=0.05 and simulate up to 30 Trotter layers, the results
 are intended to show the different <Z_{1} Z_{1}(t)> values obtained with PP.
"""
function clip_coeff!(p::Dict{PauliBasis{N}, T}; thresh=1e-3) where {N, T}
    filter!(p-> abs(p.second) ≥ thresh, p)
end

"HS norm in Pauli basis: sum_P |c_P|^2."
function hs_norm2(ps::PauliSum)
    s = 0.0
    @inbounds for (_, c) in ps
        s += abs2(c)
    end
    return s
end

"Post-merge pruning in one place."
function prune_post!(ps::PauliSum; scheme::Symbol=:none, mag_thresh::Float64=0.0, wmax::Union{Nothing,Int}=nothing)
    if scheme == :mag || scheme == :combined
        (mag_thresh > 0) && filter!(kv -> abs(kv.second) ≥ mag_thresh, ps)
    end
    if scheme == :weight || scheme == :combined
        (wmax !== nothing) && UnitaryPruning.clip_pauli_weight!(ps, max_w = wmax)
    end
    return ps
end

"""
Pauli-propagation time series with pruning controls.

Arguments:
- generators, angles  : from UnitaryPruning.heisenberg_1D(...)
- o                   : PauliSum (used as both V and W by default)
- ket                 : state for estimator (your current approach)
- scheme              : :none | :mag | :weight | :combined
- thresh              : magnitude threshold (|coeff|)
- wmax                : max Pauli string length (Int or nothing)
- collect_snapshots   : store PauliSum at each sampled time

Returns:
(C::Vector{Float64}, dicts::Vector{PauliSum}, n_terms::Vector{Int}, loss::Vector{Float64})
where C is your C(t)-type, dicts are per-layer snapshots only if requested,
n_terms is term count per sample, loss = 1 - HS_norm2 per sample.
"""
function evolution_op(generators::Vector{Pauli{N}}, angles,
                        o::PauliSum{N}, ket;
                        scheme::Symbol=:mag,
                        thresh::Float64=1e-3,
                        wmax::Union{Nothing,Int}=nothing,
                        collect_snapshots::Bool=false) where {N}

    nt = length(angles)
    @assert length(generators) == nt "generators/angles length mismatch"
    vcos = cos.(angles); vsin = sin.(angles)

    step = 1 #3*(N-1)                      # layers per physical time step
    nsamp = Int(nt ÷ step) + 1          # include t=0
    Cvals  = Vector{Float64}(undef, nsamp)
    nterms = Vector{Int}(undef, nsamp)
    loss   = Vector{Float64}(undef, nsamp)

    Wt = deepcopy(o)                    # evolve W ≡ o
    V  = deepcopy(o)                              # V ≡ o by default

    # optional layer-by-layer dict dump 
    dicts = collect_snapshots ? Vector{Any}(undef, nt) : Any[]

    # t = 0 sample
    C0  = real(expectation_value(V, ket))  # C(0)=1 for Pauli V=W on any ket
    Cvals[1]  = C0
    nterms[1] = length(Wt)
    n2 = hs_norm2(Wt); loss[1] = 1 - n2

    sample_idx = 2

    for t in 1:nt
        g  = generators[t]
        pb = PauliBasis(g)
        next_W = PauliSum(N)

        @inbounds for (oi, ci) in Wt
            if PauliOperators.commute(oi, pb)
                sum!(next_W, oi * ci)
            else
                # cos branch
                sum!(next_W, oi * (ci * vcos[t]))
                # sin branch 
                oj = g * oi
                sum!(next_W, oj * (ci * vsin[t] * 1im))
                end
            end


        # --- POST pruning on the MERGED sum (recommended default) ---
        prune_post!(next_W; scheme=scheme, mag_thresh=thresh, wmax=wmax)
        # -----------------------------------------------------------

        Wt = next_W
        if collect_snapshots
            dicts[t] = deepcopy(Wt)
        end

        os = V * Wt
        Ct  = real(expectation_value(os, ket))
        Cvals[sample_idx]  = Ct
        nterms[sample_idx] = length(Wt)
        n2 = hs_norm2(Wt); loss[sample_idx] = 1 - n2
        sample_idx += 1

    end

    return Cvals, dicts, nterms, loss
end

"""
tilted_field_ising_hamiltonian(N, J, B, theta; periodic=false)
from:
https://journals.aps.org/pra/pdf/10.1103/PhysRevA.75.022304

Return (generators, coeffs) that represent the Hamiltonian

  H = J * sum_{n=1}^{N-1 (or N if periodic)} σ^z_n σ^z_{n+1}
      + B * sum_{n=1}^N [ sin(theta) * σ^x_n + cos(theta) * σ^z_n ]

- N: number of spins (Int)
- J: ZZ coupling (Real)
- B: field magnitude (Real)
- theta: tilt angle (Real)
- periodic: if true, include bond (N,1); otherwise open chain
"""
function unitary_sequence_tilted_field_ising_1D(N::Int, J::Real, B::Real, theta::Real; k::Int=2)
    generators = Vector{Pauli{N}}([])          # array of Pauli operator descriptors
    coeffs     = Vector{Float64}([])           # matching coefficients
    for _ in 1:k
        # ZZ coupling terms
        for i in 1:N-1
           push!(generators, Pauli(N, Z=[i, i+1]))
           push!(coeffs, -J)
        end

        # local field terms: B * ( sin(theta) * X + cos(theta) * Z )
        bx = B * sin(theta)
        bz = B * cos(theta)
        for i in 1:N
            push!(generators, Pauli(N, X=[i]))
            push!(coeffs, bx)
            push!(generators, Pauli(N, Z=[i]))
            push!(coeffs, bz)        
        end
            
    end

    return generators, coeffs

end



"""
 Arguments:
 - N            : number of qubits
 - threshold    : pruning threshold (magnitude-based)
 - dt           : time step for Trotterization
 - T            : total time to simulate
 - scheme       : pruning scheme (:none | :mag | :weight | :combined)
 - wmax         : max Pauli weight (Int or nothing)
 - keep_layers  : whether to store the full operator at each layer (for debugging)
 - saveprefix   : prefix for saved figure files

 Returns:
 A named tuple with fields:
 - t     : time grid
 - F     : EXPECTATION VALUE OF THE OPERATOR BEING EVOLVED
 - n     : number of Pauli terms over time
 - loss  : truncation loss over time
"""
function run1D(; N=6, threshold=1e-3, dt=0.1, T=10.0, scheme=:mag,
             wmax::Union{Nothing,Int}=nothing, keep_layers=false, saveprefix="test")


    # - - - Global parameters - - - #
    ket = Ket(N, 0)
    J = 1.0
    B = 1.0
    theta = π/4
    k = Int(floor(T/dt))

    #- - - 1D tilted field Ising model - - -#
    o   = Pauli(N, Z=[1])
    gens, params = unitary_sequence_tilted_field_ising_1D(N, J, B, theta, k=k)

    # Call evolution_op. Forward k if present (and hope evolution_op accepts it).
    F, dict, nterms, loss = evolution_op(gens, params, PauliSum(o), ket;
                                         scheme=scheme, thresh=threshold, wmax=wmax,
                                         collect_snapshots=keep_layers)

    # ------------------------------------------------------------------------------
    # Ensure 1D vectors
    F = vec(F)
    nterms = vec(nterms)
    loss = vec(loss)

    # Number of snapshots actually returned
    nsnap = length(F)

    if nsnap == 0
        error("evolution_op returned zero snapshots (length(F) == 0). Aborting.")
    end

    # Build time grid by mapping snapshots evenly across [0, T].
    # This is robust to whether snapshots were per-dt, per-Trotter-layer, etc.
    # If you *know* snapshots correspond to exact dt spacing, you can instead use 0:dt:...
    println("* * * * Number of snapshots collected: $nsnap")
    tgrid = collect(range(0.0, stop=T, length=nsnap))
    #println("tgrid = $tgrid")
    
    # Align nterms and loss to F's length
    function align_to_ref(vec, ref_len, name)
        if length(vec) == ref_len
            return vec
        elseif length(vec) > ref_len
            @warn("$name longer than F; truncating to match snapshots: $(length(vec)) -> $ref_len")
            return vec[1:ref_len]
        else
            @warn("$name shorter than F; padding with last value to match snapshots: $(length(vec)) -> $ref_len")
            return vcat(vec, fill(vec[end], ref_len - length(vec)))
        end
    end

    nterms = align_to_ref(nterms, nsnap, "nterms")
    loss   = align_to_ref(loss, nsnap, "loss")

    # main plot
    plt1 = plot(tgrid, real(F), lw=2,
                label="$(scheme), th=$(threshold), w=$(wmax === nothing ? "All" : string(wmax))")
    xlabel!(plt1, "Time"); ylabel!(plt1, "Expectation Value")
    title!(plt1, "N=$N; dt=$dt; k=$(k === missing ? "auto" : string(k))")

    # complexity & loss
    plt2 = plot(tgrid, nterms, lw=2, label="n_terms")
    xlabel!(plt2, "Time"); ylabel!(plt2, "# Pauli terms"); title!(plt2, "Complexity growth")

    plt3 = plot(tgrid, loss, lw=2, label="1 - HS_norm2")
    xlabel!(plt3, "Time"); ylabel!(plt3, "lost HS weight"); title!(plt3, "Truncation loss")

    fname_w = (wmax === nothing) ? "All" : string(wmax)
    savefig(plt1, "$(saveprefix)_F_N=$(N)_$(scheme)_th=$(threshold)_w=$(fname_w).pdf")
    savefig(plt2, "$(saveprefix)_nterms_N=$(N)_$(scheme)_th=$(threshold)_w=$(fname_w).pdf")
    savefig(plt3, "$(saveprefix)_loss_N=$(N)_$(scheme)_th=$(threshold)_w=$(fname_w).pdf")

    return (t=tgrid, F=F, n=nterms, loss=loss, dict=dict)
end


# 1D tests
#- - - Check Hamiltonian properties and spectrum - - -#
function tilted_field_ising_hamiltonian(N::Int, J::Real, B::Real, theta::Real)
    H = PauliSum(N, Float64)                          # Hamiltonian PauliSum 
    # ZZ coupling terms
    for i in 1:N-1
        H += Pauli(N, Z=[i, i+1]) * (-J)
    end

    # local field terms: B * ( sin(theta) * X + cos(theta) * Z )
    bx = B * sin(theta)
    bz = B * cos(theta)
    for i in 1:N
        H += Pauli(N, X=[i]) * bx
        H += Pauli(N, Z=[i]) * bz        
    end
            
    return H
end


# - - - Global parameters - - - #
NQubits = 4
J = 1.0
B = 1.0
theta = 0.0# π/4
#- - - 1D tilted field Ising model - - -#
o   = Pauli(NQubits, Z=[1])
H = tilted_field_ising_hamiltonian(NQubits, J, B, theta)
e, v = eigen(Matrix(H))

println("Eigenspectrum of tilted-field 1D Ising Hamiltonian (N=$NQubits):")
for eig in e
    println(eig)
end

for veig in eachcol(v)
    println("eigenvector:")
    for comp in veig
        println(comp)
    end
end




# # - - - Threshold+Weight Based Pruning Comparisons - - - # 
thresholds = [1e-3]
curves = [] 
total_time = 0.50 
wmax = 6
for th in thresholds
    #push!(curves, run1D(N=NQubits, threshold=th, dt=0.01, T=total_time, scheme=:combined, wmax=wmax, saveprefix="test_QSP_Ising_1D"))
    push!(curves, run1D(N=NQubits, threshold=th, dt=0.1, T=total_time, scheme=:mag, saveprefix="test_QSP_Ising_1D"))
end 

# overlay C(t) curves in one figure 
plt = plot() 
for (i,th) in enumerate(thresholds) 
    plot!(plt, curves[i].t, curves[i].F, lw=2, label="th=$(th)")
end
xlabel!("Time"); ylabel!("<Z>"); title!("Max weight = $wmax")
savefig(plt, "N=$(NQubits)_titled_Ising_1D.pdf")
