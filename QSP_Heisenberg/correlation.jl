#=
Correlation Function Simulation
-----------------------------------------------
Dynamic Correlator for an Ising/Heisenberg model
on a lattice defined by an adjacency matrix.
It uses the Heisenberg picture evolution
of operators (Pauli strings) via Trotterization.
=#

using PauliOperators
using LinearAlgebra
using Printf
using NPZ
using DelimitedFiles
using DBF


function ising_hyperbolic_pauli(adjacency_matrix, Jzz, Jx, m)
    am = adjacency_matrix
    N = size(am, 1)
    H_sum = PauliSum(N)

    for i in 1:N
        # transverse field term
        H_sum += (-Jx) * Pauli(N, X=[i])
        # symmetry breaking term
        H_sum += (-m) * Pauli(N, Z=[i])
        for j in 1:N
            if am[i, j] == 1 && i < j
                # interaction term
                H_sum += (-Jzz) * Pauli(N, Z=[i, j])
            end
        end
    end
    return H_sum
end



"""
build_H_terms(H_total_sum; order_type::String = "lexical", rng_seed::Int = 1234)
Given a PauliSum representing the total Hamiltonian, extract its terms and order them according to the specified order_type.
order_type can be:
- "lexical": Sort by Z then X bits (default)
- "by_site": Sort by the sites involved (e.g., Z1, Z2
- "by_kind": Sort by type (Z-only, X-only, mixed) then by sites
- "random": Randomize the order using the provided rng_seed
Returns a vector of pairs (PauliBasis, coefficient) representing the Hamiltonian terms in the desired order.

"""
function build_H_terms(H_total_sum; order_type::String = "lexical", rng_seed::Int = 1234)
	H_vec = collect(H_total_sum)

	if order_type == "lexical"
		H_terms = sort(H_vec, by = p -> (p.first.z, p.first.x))

	elseif order_type == "by_site"
		site_pair(basis) = begin
			zbits = basis.z
			xbits = basis.x
			mask = zbits | xbits
			sites = Int[]
			i = 1
			while mask != 0
				if (mask & 0x1) == 1
					push!(sites, i)
				end
				mask >>= 1
				i += 1
			end
			if isempty(sites)
				(typemax(Int), typemax(Int))
			elseif length(sites) == 1
				(sites[1], sites[1])
			else
				(minimum(sites), maximum(sites))
			end
		end
		H_terms = sort(H_vec, by = p -> site_pair(p.first))

	elseif order_type == "by_kind"
		function kind_code(basis)
			zbits = basis.z
			xbits = basis.x
			mask = zbits | xbits
			has_x = (xbits != 0)
			has_z = (zbits != 0)
			if has_z && !has_x   
				return 1
			elseif has_x && !has_z  
				return 2
			elseif has_x && has_z   
				return 3
			else                    
				return 4
			end
		end
		site_pair(basis) = begin
			zbits = basis.z
			xbits = basis.x
			mask = zbits | xbits
			sites = Int[]
			i = 1
			while mask != 0
				if (mask & 0x1) == 1
					push!(sites, i)
				end
				mask >>= 1
				i += 1
			end
			if isempty(sites)
				(typemax(Int), typemax(Int))
			elseif length(sites) == 1
				(sites[1], sites[1])
			else
				(minimum(sites), maximum(sites))
			end
		end
		H_terms = sort(H_vec, by = p -> (kind_code(p.first), site_pair(p.first)))

	elseif order_type == "random"
		rng = MersenneTwister(rng_seed)
		H_terms = shuffle(rng, H_vec)

	else
		error("Unknown order_type = $order_type")
	end

	return H_terms
end
function anti_commutes(p1::Pauli, p2::Pauli)
	return !PauliOperators.commute(PauliBasis(p1), PauliBasis(p2))
end

function trotter_step_1st_order_DBf!(W::PauliSum{N, T},
	H_terms::Vector{Pair{PauliBasis{N}, T}},
	dt::Float64,
    V_ops::Vector{Pauli{N}};
	cutoff = 1e-8,
	error_accumulation = false,
) where {N, T}
	
    step_errors = zeros(Float64, length(V_ops))
	corr_values = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = 2 * h_real * dt

		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		
        if error_accumulation==true
			vals_pre = measure_all_corrs(W)
		end
		
        DBF.coeff_clip!(W, thresh = cutoff)
		vals_post = measure_all_corrs(W)
		
        if error_accumulation==true
			step_errors .+= (vals_pre .- vals_post)
		end
		corr_values .+= vals_post
	end
	return W, corr_values, step_errors
end

function trotter_step_2nd_order_DBf!(
						W::PauliSum{N, T},
						H_terms::Vector{Pair{PauliBasis{N}, T}},
						dt::Float64,
						V_ops::Vector{Pauli{N}};
						cutoff = 1e-8,
						error_accumulation = false,
) where {N, T}

	step_errors = zeros(Float64, length(V_ops))
	corr_values = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	# --- 1. Forward half-step ---
	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		
        if error_accumulation==true
			vals_pre = measure_all_corrs(W)
		end
		
        DBF.coeff_clip!(W, thresh = cutoff)
		vals_post = measure_all_corrs(W)
		
        if error_accumulation==true
			step_errors .+= (vals_pre .- vals_post)
		end
		corr_values .+= vals_post
	end

	# --- 2. Backward half-step ---
	for (H_basis, h_coeff) in Iterators.reverse(H_terms)
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)

		DBF.coeff_clip!(W, thresh = 1e-12)
		
        if error_accumulation==true
			vals_pre = measure_all_corrs(W)
		end
		
        DBF.coeff_clip!(W, thresh = cutoff)
		vals_post = measure_all_corrs(W)
		
        if error_accumulation==true
			step_errors .+= (vals_pre .- vals_post)
		end
		corr_values .+= vals_post
	end

	return W, corr_values, step_errors
end
function two_point_correlator_new(W_sum::PauliSum{N,T}, V_op::Pauli{N};
	Z0::Float64 = 1.0) where {N,T}

	# wrap V_op into a PauliSum
	V_sum = PauliSum(N, T)
	V_sum[PauliBasis(V_op)] = one(T)

	# raw two-point <Z1(t) V(0)> ∝ Tr[W_sum * V]
	corr = 0.0
	for (v_basis, v_coeff) in V_sum
		w_coeff = get(W_sum, v_basis, zero(T))
		corr += real(w_coeff * conj(v_coeff))
	end

	# <Z1(t)> ∝ Tr[W_sum]
	expZ1 = tr(W_sum)

	return corr - expZ1 * Z0
end

"""
    two_point_correlator(W_sum, V_sum)
Computes C(t) = <W(t) V(0)>_{∞T}.
"""
function two_point_correlator(W_sum::PauliSum{N, T},  V_op::Pauli{N}) where {N, T}
    corr = 0.0
	V_sum = PauliSum(N, ComplexF64)
	V_sum[PauliBasis(V_op)] = 1.0
    for (v_basis, v_coeff) in V_sum
        w_coeff = get(W_sum, v_basis, zero(T))
        corr += real(w_coeff * conj(v_coeff))
        
    end
    # corr = corr/tr(W_sum * V_sum) # Normalize by the trace of W*V to get the correct infinite-temperature value
    return corr
end

function as_PauliSum(p::Pauli{N}) where {N}
    ps = PauliSum(N, ComplexF64)
    ps[PauliBasis(p)] = 1.0 + 0im
    return ps
end
function trotter_step_2nd_order_DBf_commutationwise!(
						W::PauliSum{N, T},
						H_terms::Vector{Pair{PauliBasis{N}, T}},
						dt::Float64,
						V_ops::Vector{Pauli{N}};
						cutoff = 1e-8,
						error_accumulation = false,
) where {N, T}

	step_errors = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	# --- 1. Forward half-step ---
	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		W, clipped_W = coeff_clip(W, thresh = cutoff)
		
        if error_accumulation==true
			errors_val = measure_all_corrs(clipped_W)
			step_errors .+= errors_val
		end
	end

	# --- 2. Backward half-step ---
	for (H_basis, h_coeff) in Iterators.reverse(H_terms)
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)

		DBF.coeff_clip!(W, thresh = 1e-12)
		W, clipped_W = coeff_clip(W, thresh = cutoff)
		
        if error_accumulation==true
			errors_val = measure_all_corrs(clipped_W)
			step_errors .+= errors_val
		end
	end

	return W, step_errors
end

function trotter_step_2nd_order_DBf_commutationwise_weight!(
						W::PauliSum{N, T},
						H_terms::Vector{Pair{PauliBasis{N}, T}},
						dt::Float64,
						V_ops::Vector{Pauli{N}};
						cutoff = 1e-8,
						max_pauli_weight::Int = 0,
						error_accumulation = false,
) where {N, T}

	step_errors = zeros(Float64, length(V_ops))

	function measure_all_corrs(W_curr)
		return [correlator_from_W(W_curr, v) for v in V_ops]
	end

	# --- 1. Forward half-step ---
	for (H_basis, h_coeff) in H_terms
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)
		DBF.coeff_clip!(W, thresh = 1e-12)
		val_pre=measure_all_corrs(W)
		weight_clip!(W, max_pauli_weight)
		val_post=measure_all_corrs(W)
		step_errors .+= val_post .- val_pre
	end

	# --- 2. Backward half-step ---
	for (H_basis, h_coeff) in Iterators.reverse(H_terms)
		h_real = real(h_coeff)
		θ = h_real * dt
		W = DBF.evolve(W, H_basis, θ)

		DBF.coeff_clip!(W, thresh = 1e-12)
		val_pre=measure_all_corrs(W)
		weight_clip!(W, max_pauli_weight)
		val_post=measure_all_corrs(W)
		step_errors .+= val_post .- val_pre
	end

	return W, step_errors
end

function coeff_clip(ps::PauliSum{N}; thresh=1e-16) where {N}
	ps_=deepcopy(ps)
    return filter(p->abs(p.second) > thresh, ps), filter(p->abs(p.second) <= thresh, ps_)
end

"""
	correlator_from_W(W_sum, V_op)

Compute the infinite-temperature correlation function C(t) = <W(t) V(0)>_{∞T}.
In the Pauli basis, this is simply the coefficient of V_op in W(t).
"""
function correlator_from_W(W_sum::PauliSum{N, T}, V_op::Pauli{N}) where {N, T}
	V_basis = PauliBasis(V_op)
	# Extract the coefficient of V_basis, default to 0.0 if it doesn't exist
	c_val = get(W_sum, V_basis, zero(T))
	return real(c_val) 
end


function save_corr_csv(filename::String,
	time_data::Vector{Float64},
	corr_data::Dict{Int, Vector{Float64}},
	V_sites::Vector{Int})

	nT = length(time_data)
	nJ = length(V_sites)

	M = Array{Float64}(undef, nT, nJ + 1)

	for t_idx in 1:nT
		M[t_idx, 1] = time_data[t_idx]
	end

	for (col_idx, j) in enumerate(V_sites)
		vals = corr_data[j]
		@assert length(vals) == nT
		for t_idx in 1:nT
			M[t_idx, col_idx+1] = vals[t_idx]
		end
	end

	writedlm(filename, M, ',')
end


# ==========================================
# 4. Main Simulation 
# ==========================================

function load_and_run_corr(npy_path::String;
							Jzz = 1.0, Jx = 1.5, m = 0.1,
							dt = 0.05, t_max = 2.0,
							cutoff = 1e-6,
							max_maj_weight::Int = 0,
							max_pauli_weight::Int = 0,
							trotter_order::Int = 2,
							order_type::String = "lexical",
							errorsor_check::Bool = false,
							old_clipping::Bool = false)

    println("="^60)
    println("Correlation Simulation (Trotter Order: $trotter_order)")
    println("="^60)

    # --- 1. Load Data ---
    println("Loading adjacency matrix from: $npy_path")
    adj_matrix = readdlm(npy_path, Int)
    if !all(x -> x == floor(x), adj_matrix)
        adj_matrix = Int.(round.(adj_matrix))
    end
    N = size(adj_matrix, 1)
   
    println("System Size: N = $N")

    # --- 2. Build Hamiltonian ---
    println("\nBuilding Hamiltonian...")
    H_total_sum = ising_hyperbolic_pauli(adj_matrix, Jzz, Jx, m)
    H_terms = build_H_terms(H_total_sum; order_type = "lexical")

    # --- 3. Define Operators ---
    println("\nInitializing W(0) = Z_1 and V sites...")
    i_site = 1
    V_sites = [2] # You can expand this array to include more sites like [2,3,4,5]

    W_op = Pauli(N, Z = [i_site])
    W_sum = PauliSum(N, ComplexF64)
    W_sum[PauliBasis(W_op)] = 1.0

    V_ops_vec = [Pauli(N, Z = [j]) for j in V_sites]

    # --- 4. Simulation Constants ---
    steps = floor(Int, t_max / dt)

    time_data = Float64[]
    corr_data = Dict{Int, Vector{Float64}}()
    for j in V_sites
        corr_data[j] = Float64[]
    end

    total_accumulated_error = zeros(Float64, length(V_sites))

    # --- 5. Print Headers ---
    println("\n" * "-"^160)

    header_parts = String[]
    for j in V_sites
        s = @sprintf("C_z%dz%-2d  Corr       Err     ", i_site, j)
        push!(header_parts, s)
    end
    data_header = join(header_parts, "    ")

    println(@sprintf("%8s  %10s  %10s    %s", "Time", "N_terms", "Calc_Time", data_header))
    println("-"^160)

    # --- 6. Time Evolution Loop ---
    for step in 0:steps
        t = step * dt

        step_errors = zeros(Float64, length(V_sites))

        t_start = time()
		
        # 1. DBF Second Order with Tracking
        if trotter_order == 2 && !old_clipping
            W_sum, step_errors = trotter_step_2nd_order_DBf_commutationwise!(
                W_sum,
                H_terms,
                dt,
                V_ops_vec;
                cutoff = cutoff,
                error_accumulation = errorsor_check,
            )
		elseif trotter_order == 2 && old_clipping
				W_sum, corr_val_arr, step_errors = trotter_step_2nd_order_DBf!(
					W_sum,
					H_terms,
					dt,
					V_ops_vec;
					cutoff = cutoff,
					error_accumulation = errorsor_check,
				)
        # 2. DBF First Order with Tracking
        elseif trotter_order == 1
            W_sum, corr_val_arr, step_errors = trotter_step_1st_order_DBf!(
                W_sum, 
                H_terms, 
                dt, 
                V_ops_vec;
                cutoff = cutoff,
                error_accumulation = errorsor_check
            )
		elseif max_pauli_weight > 0
			W_sum, 
			corr_val_arr, 
			step_errors=trotter_step_2nd_order_DBf_commutationwise_weight!,# W_sum, 
			H_terms, 
			dt, 
			V_ops_vec;
			cutoff = cutoff,
			max_pauli_weight = max_pauli_weight,
			error_accumulation = errorsor_check
		else
            error("Invalid configuration")
        end

        if errorsor_check
            total_accumulated_error .+= step_errors
        end

        print_parts = String[]
        for (idx, j) in enumerate(V_sites)
			# Measure the correlation coefficient
			corr_val = correlator_from_W(W_sum, V_ops_vec[idx])
      
            push!(corr_data[j], corr_val)
            err_val = total_accumulated_error[idx]
			corrected_val = corr_val + err_val
           
			str = @sprintf("%8.5f  %8.5f  %8.2e", corr_val, corrected_val, err_val)
            push!(print_parts, str)
        end
        push!(time_data, t)
		
        calc_time = time() - t_start
        row_str = join(print_parts, "    ")
        @sprintf("%8.3f  %10d  %9.4fs    %s\n", t, length(W_sum), calc_time, row_str) |> print
        flush(stdout)

        # --- C. Extra Clipping ---
        if max_maj_weight > 0
            majorana_weight_clip!(W_sum, max_maj_weight)
        end
        if max_pauli_weight > 0
            weight_clip!(W_sum, max_pauli_weight)
        end
        DBF.coeff_clip!(W_sum, thresh = 1e-15)
    end

    return time_data, corr_data, V_sites
end