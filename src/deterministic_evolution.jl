using Random
using BenchmarkTools
"""
    deterministic_pauli_rotations(generators::Vector{P}, angles, o::P, ket; nsamples=1000) where {N, P<:Pauli}


"""
function deterministic_pauli_rotations(generators::Vector{Pauli{N}}, angles, o::Pauli{N}, ket ; thres=1e-3) where {N}

    #
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    expval = zero(ComplexF64)


    # opers = PauliSum{N}(Dict(o.pauli=>1.0*(1im)^o.θ))#Dict{Tuple{Int128, Int128}, Complex{Float64}}((o.pauli.z,o.pauli.x)=>1.0*(1im)^o.θ)
    opers = PauliSum(o)
  
    n_ops = zeros(Int,nt)
    
    for t in 1:nt

        g = generators[t]
        branch_opers = PauliSum(N)

        sizehint!(branch_opers.ops, 1000)
        for (key,value) in opers.ops
            
            oi = key
            if commute(oi, g.pauli)
                if haskey(branch_opers, oi)
                    branch_opers[oi] += value
                else
                    branch_opers[oi] = value
                end
                continue
            end
            if abs(value) > thres #If greater than threshold then split the branches
                # cos branch
                coeff = vcos[t] * value
                           
                if haskey(branch_opers, oi) # Add operator to dictionary if the key doesn't exist
                    branch_opers[oi] += coeff
                else
                    branch_opers[oi] = coeff #Modify the coeff if the key exists already
                end

                # sin branch
                coeff = vsin[t] * 1im * value
                oi = Pauli{N}(0, oi)
                oi = g * oi    # multiply the pauli's

                if haskey(branch_opers, oi.pauli) # Add operator to dictionary if the key doesn't exist
                    branch_opers[oi.pauli] += coeff * (1im)^oi.θ
                else
                    branch_opers[oi.pauli] = coeff * (1im)^oi.θ #Modify the coeff if the key exists already
                end
            end
        end
        n_ops[t] = length(branch_opers)
        opers = deepcopy(branch_opers) # Change the list of operators to the next row

    end

    for (key,value) in opers.ops
        oper = Pauli(UInt8(0), key)
        expval += value*PauliOperators.expectation_value(oper, ket)
    end
   
    return expval, n_ops
end

"""
    clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    Clip the PauliSum in place, removing terms with abs(coeff) < thresh
"""
function clip!(ps::PauliSum{N}; thresh=1e-16) where {N}
    filter!(p->abs(p.second) ≥ thresh , ps)
end

"""
    bfs_evolution(generators::Vector{Pauli{N}}, angles, o::Pauli{N}, ket ; thres=1e-3) where {N}


"""
function bfs_evolution(generators::Union{Vector{Pauli{N}},Vector{PauliBasis{N}}}, angles, o::PauliSum{N}, ket ; thresh=1e-3) where {N}

    #
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    expval = zero(ComplexF64)


    o_transformed = deepcopy(o)
  
    n_ops = zeros(Int,nt)
    
    for t in 1:nt

        g = generators[t]

        sin_branch = PauliSum(N)

        for (oi,coeff) in o_transformed
           
            abs(coeff) > thresh || continue


            if PauliOperators.commute(oi, PauliBasis(g)) == false
                
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's
                sum!(sin_branch, oj * vsin[t] * coeff * 1im)

            end
        end
        sum!(o_transformed, sin_branch) 
        clip!(o_transformed, thresh=thresh)
        n_ops[t] = length(o_transformed)
    end

    for (oi,coeff) in o_transformed
        expval += coeff*PauliOperators.expectation_value(oi, ket)
    end
   
    return expval, n_ops
end


"""
    bfs_evolution(generators::Vector{Pauli{N}}, angles, o::Pauli{N}, ket ; thres=1e-3) where {N}
    Based on max_weight. This function avoids the use of a clipping function.
"""
function bfs_evolution_weight(generators::Union{Vector{Pauli{N}},Vector{PauliBasis{N}}}, angles, o::PauliSum{N}, ket , w_type::String ; max_weight=4) where {N}
#
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    expval = zero(ComplexF64)

    o_transformed = deepcopy(o)
    #sin_branch = PauliSum(N)
  
    n_ops = zeros(Int,nt)

    for t in 1:nt

        g = generators[t]
        
        sin_branch = PauliSum(N)

        for (oi, coeff) in o_transformed#.ops
            #println("coeff ", coeff, " oi: ", oi, "generator ", PauliBasis(g))
            

            if PauliOperators.commute(oi, PauliBasis(g)) == false
            
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's

                if w_type == "Majorana"
                    if PauliOperators.simple_majorana_weight(oj) > max_weight
                        #println("Majorana weight larger than max...")
                        continue
                    end
                elseif w_type == "Pauli"
                        if PauliOperators.pauli_weight(oj) > max_weight
                          #  println("Pauli weight larger than max...")
                            continue
                        end
                    else
                        error("Unknown weight type: $w_type")
                end

                sum!(sin_branch, oj * vsin[t] * coeff * 1im)

            end
        end
        sum!(o_transformed, sin_branch) 
        #clip_thresh!(o_transformed, thresh=thresh)
        # clip_weight!(o_transformed, weight=majo_thresh)
        n_ops[t] = length(o_transformed)
    end

    for (oi,coeff) in o_transformed#.ops
        expval += coeff*expectation_value(oi, ket)
    end
   
    return expval, n_ops
end

function clip_majorana_weight!(p::Dict{PauliBasis{N}, T}; max_w=4) where {N, T}
    # Accessing the keys of the sum works
    #for k in collect(keys(p))
    #    print("k ", k, "p[k] ", p[k])
    #    println("Test weight :", PauliOperators.simple_majorana_weight(k))
    #end
    filter!(q -> PauliOperators.simple_majorana_weight(q.first) ≤ max_w, p)
end

function clip_pauli_weight!(p::Dict{PauliBasis{N}, T}; max_w=4) where {N, T}
    filter!(q -> PauliOperators.pauli_weight(q.first) ≤ max_w, p)
end

function clip_thresh_weight!(ps::PauliSum{N}; thresh=1e-16, lc = 0, w_type = 0) where {N}
    if w_type == 0 
        filter!(p->(PauliOperators.pauli_weight(p.first) ≤ lc) && (abs(p.second) ≥ thresh) , ps)
    else
        filter!(p->(PauliOperators.simple_majorana_weight(p.first) ≤ lc) && (abs(p.second) ≥ thresh) , ps)
    end     
end


"""
    bfs_evolution(generators::Vector{Pauli{N}}, angles, o::Pauli{N}, ket ; thres=1e-3) where {N}
    Based on max_weight. This function uses clipping functions.
"""
function bfs_evolution_weight_clip(generators::Union{Vector{Pauli{N}},Vector{PauliBasis{N}}}, angles, o::PauliSum{N}, ket , w_type::String ; max_weight=4) where {N}
#
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    expval = zero(ComplexF64)

    o_transformed = deepcopy(o)
  
    n_ops = zeros(Int,nt)

    for t in 1:nt

        g = generators[t]
        
        sin_branch = PauliSum(N)

        for (oi, coeff) in o_transformed#.ops
            #println("coeff ", coeff, " oi: ", oi, "generator ", PauliBasis(g))

            if PauliOperators.commute(oi, PauliBasis(g)) == false
            
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's

                sum!(sin_branch, oj * vsin[t] * coeff * 1im)

            end
        end
        sum!(o_transformed, sin_branch) 

        if w_type == "Majorana"
            clip_majorana_weight!(o_transformed, max_w = max_weight)
        
        elseif w_type == "Pauli"
            clip_pauli_weight!(o_transformed, max_w = max_weight)
        end
            
        n_ops[t] = length(o_transformed)
    end

    for (oi,coeff) in o_transformed#.ops
        expval += coeff*expectation_value(oi, ket)
    end
   
    return expval, n_ops
end

"""
 The following function performs the bfs evolution performing the coefficient thresholding in combination with the weight cutoff pruning.
    w_type = 0 : Pauli weight
    w_type = 1 : Majorana weight
    w : weight cutoff
"""
function bfs_evolution_thresh_weight(generators::Vector{Pauli{N}}, angles, o::PauliSum{N}, ket ; thresh=1e-3, w_type = 0, w = 2) where {N}

    #
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ) O + i sin(θ) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    vcos = cos.(angles)
    vsin = sin.(angles)

    # collect our results here...
    expval = zero(ComplexF64)


    o_transformed = deepcopy(o)
    sin_branch = PauliSum(N)
 
    n_ops = zeros(Int,nt)
    
    for t in 1:nt

        g = generators[t]

        sin_branch = PauliSum(N)

        for (oi, coeff) in o_transformed
           
            abs(coeff) > thresh || continue

            if PauliOperators.commute(oi, PauliBasis(g)) == false
                
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's
                sum!(sin_branch, oj * vsin[t] * coeff * 1im)
  
            end
        end
        sum!(o_transformed, sin_branch) 
        # clip!(o_transformed, thresh=thresh)
        clip_thresh_weight!(o_transformed, thresh=thresh, lc = w, w_type = w_type)
        # if w_type == 0

        #     weightclip!(o_transformed, lc = w)

        # elseif w_type == 1 

        #     majorana_clip!(o_transformed, lc = w)

        # end

        n_ops[t] = length(o_transformed)
    end

    coeff_norm2 = 0

    for (oi,coeff) in o_transformed
        expval += coeff*PauliOperators.expectation_value(oi, ket)
        coeff_norm2+= abs(coeff)^2      # final list of operators
    end
    coeff_norm2 = sqrt(coeff_norm2)

    return expval, n_ops, coeff_norm2
end

"""
    bfs_evolution_vqe(generators::Vector{Pauli{N}}, angles, o::Pauli{N}, ket ; thres=1e-3) where {N}
    This function is used for VQE type calculations where we evolve the target operator under the action
    of a given operator pool, this initial function uses a thresholding pruning scheme to limit the number of operators.
    It Returns the evolved operator for further processing, and works with one operator at a time.

"""
function bfs_evolution_vqe(generators::Union{Vector{Pauli{N}},Vector{PauliBasis{N}}}, angles, o::PauliSum{N}; thresh=1e-3) where {N}

    #
    # for a single pauli Unitary, U = exp(-i θn Pn/2)
    # U' O U = cos(θ/2) O + i sin(θ/2) OP
    nt = length(angles)
    length(angles) == nt || throw(DimensionMismatch)
    
    vcos = cos.(angles)
    vsin = sin.(angles)

    o_transformed = deepcopy(o)
  
    n_ops = zeros(Int,nt)
    
    for t in 1:nt

        g = generators[t]

        sin_branch = PauliSum(N)

        for (oi,coeff) in o_transformed
           
           # abs(coeff) > thresh || continue


            if PauliOperators.commute(oi, PauliBasis(g)) == false
                
                # cos branch
                o_transformed[oi] = coeff * vcos[t]

                # sin branch
                oj = g * oi    # multiply the pauli's
                sum!(sin_branch, oj * vsin[t] * coeff * 1im)

            end
        end
        sum!(o_transformed, sin_branch) 
        clip!(o_transformed, thresh=thresh)
        n_ops[t] = length(o_transformed)
    end

      
    return o_transformed
end