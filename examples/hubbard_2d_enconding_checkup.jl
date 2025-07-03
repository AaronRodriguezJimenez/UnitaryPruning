function print_qubit_mapping(D::Int; encoding::String = "interleaved")
    N = 2*D #D = Lx*Ly, the dimension of the site lattice.
    println("Encoding: $encoding")
    println("Qubit Index | Site | Spin")
    println("------------|------|------")
    for q in 1:N
        if encoding == "interleaved"
            site = div(q - 1, 2) + 1
            spin = isodd(q) ? "↑" : "↓"
        elseif encoding == "block"
            if q <= D
                site = q
                spin = "↑"
            else
                site = q - D
                spin = "↓"
            end
        else
            error("Unknown encoding: $encoding")
        end
        println(rpad("$q", 12), "|", rpad(" $site", 6), "|  $spin")
    end
end

# Build a Z observable for a given site and spin
function logical_Z(site::Int, spin::Symbol; D::Int, encoding = "interleaved")
    N = 2D
    qubit_index = 0
    if encoding == "interleaved"
        qubit_index = 2*(site - 1) + (spin == :up ? 1 : 2)
    elseif encoding == "block"
        qubit_index = site + (spin == :up ? 0 : D)
    else
        error("Unknown encoding: $encoding")
    end

    ops = fill('I', N)
    ops[qubit_index] = 'Z'
    return Pauli(String(ops))
end

print_qubit_mapping(4; encoding ="interleaved")
print_qubit_mapping(4; encoding ="block")

Z_block = logical_Z(1, :up; D=4, encoding="block")
Z_inter = logical_Z(1, :up; D=4, encoding="interleaved")

println("Z_block      => ")
display(Z_block)
println("Z_interleaved => ")
display(Z_inter)