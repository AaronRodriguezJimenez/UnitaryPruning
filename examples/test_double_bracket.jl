using PauliOperators
using LinearAlgebra

"""
 COMMENT: that's very close. But you'll want to take the diagonal of H 
 (i.e., compute D) at each time step! So the generator G, 
 is taken as the largest coefficient from the diagonal of the time dependent H. 
"""
function commutator(H::PauliSum{N,T}, D::PauliSum{N,T}) where {N,T}
    return H * D - D * H
end

function diag_op(O::PauliSum{N,T}) where {N,T}

    D = PauliSum(N)

    for (p, c) in O
        if p.x == 0
            sum!(D, c*p)
        end
    end

    return D
end

function largest_term(ps::PauliSum)
    best = nothing
    maxval = 0.0
    for (p, c) in ps
        v = abs(c)
        if v > maxval
            maxval = v
            best = p
        end
    end
    return maxval*best
end


function evolve(P::PauliSum{N,T}, G::Pauli{N}, dt) where {N,T}

    _cos = cos(dt*coeff(G))
    _sin = -1im*sin(dt*coeff(G))

    out = deepcopy(P) 

    sin_branch = PauliSum(N)

    for (p,c) in P
        if PauliOperators.commute(p,PauliBasis(G)) == false
            out[p] *= _cos
            sum!(sin_branch, c*_sin*p*PauliBasis(G))
        end
    end

    sum!(out, sin_branch)

    return out
end

function run()
    N = 5
    H = rand(PauliSum{N}, n_paulis = 100)
    max = largest_term(H)

    eigval, _ = eigen(Matrix(H))
    # display(eigval)


    t_steps = 10
    dt = 0.1

    for i in 1:t_steps
        H_temp = evolve(H, max, dt)

        D = diag_op(H_temp)
        com = commutator(H_temp, D)
        #display(com)
        max = largest_term(D)
        
        dHdt = commutator(H_temp, com)

        err = eigval - diag(Matrix(dHdt))
        display(norm(err))

        H = H_temp
    end

    return
end

run()