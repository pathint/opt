###########################################################
#
# Collection of Optimization/Search Algorithms
# Simplex Algorithm for Linear Programming
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Pivotal Transformation
# Input: tableau matrix (a) and pivotal position (p,q)
# Output: tableau w.r.t new basis
#..........................................................

function PivotalTransform(a, p, q)
    m, n = size(a)
    b = similar(a, Float64)
    b[p,:] = a[p,:] ./ a[p,q]
    for i = 1:p-1
        b[i,:] = a[i,:] .- b[p,:] .* a[i,q] # vector .* scalar
    end
    for i = p+1:m
        b[i,:] = a[i,:] .- b[p,:] .* a[i,q] # vector .* scalar
    end
    return b
end

## function PivotalTransform(a, p, q)
##     m, n = size(a)
##     b = similar(a, Float64)
##     for i = 1:m
##         for j = 1:n
##             if i == p
##                 b[i,j] = a[i,j]/a[p,q]
##             else
##                 b[i,j] = a[i,j] - a[i,q]*a[p,j]/a[p,q]
##             end
##         end
##     end
##     return b
## end

#...................................
# Simple test cases
#...................................

a = [1 5 1 0 0 40; 
     2 1 0 1 0 20; 
     1 1 0 0 1 12;
     -3 -5 0 0 0 0];
PivotalTransform(a, 1, 2)

function SelectQ(r)
    n = length(r)
    q = 0
    qv= 0
    for j = 1:n
        if r[j] < qv
            q =  j
            qv=r[j]
        end
    end
    return q
end

function SelectP(a, m, n, q)
    p = 0
    pv= Inf
    for i = 1:m
        if a[i, q] > 0 
            temp =a[i, n]/a[i, q] 
            if temp < pv
                p = i
                pv= temp
            end
        end
    end
    return p
end

#..........................................................
# Simplex Algorithm for Linear Programming given an initial tableau 
# Input: tableau matrix (a) for the initial basic feasible solution
# Output: Minimum solution the objective function
#..........................................................
function Simplex(a, # tableau matrix for Ax=b with object function, c'*x 
                 bases_original; # index vector for column basis vectors
                 iterations::Int64 = 128)
    m, n = size(a)
    b = a
    bases = bases_original
    for k = 1:iterations
        r = b[m,1:n-1]
        q = SelectQ(r)
        if q == 0 # Optimal solution is found
            return b, bases
        end
        p = SelectP(b, m, n, q)
        if p == 0 # Not-bounded
            println("Not bounded!")
            return b, bases
        end
        b = PivotalTransform(b, p, q)
        bases[p]=q
    end
    println("WARNING: ", iterations ," iterations have been exceeded.")
    return b, bases
end

#...................................
# Simple test cases
#...................................

a = [1 5 1 0 0 40; 
     2 1 0 1 0 20; 
     1 1 0 0 1 12;
     -3 -5 0 0 0 0];
Simplex(a, [3, 4, 5])

a = [  2  1 1 0 3;
       1  4 0 1 4;
      -7 -6 0 0 0];
Simplex(a, [3, 4])

a = [  1  0 1 0 0 4;
       0  1 0 1 0 6;
       1  1 0 0 1 8;
      -2 -5 0 0 0 0];
Simplex(a, [3, 4, 5])


#..........................................................
# Two-Phase Simplex Algorithm for Linear Programming  
# Phase 1: Looking for the initial basic feasible solution,
#           through auxiliary vectors
# Phase 2: Searching for optimal solution 
#..........................................................
function SimplexTwo(a, # A matrix in Ax = b 
                    b, # b vector in Ax = b,
                    c, # c vector in object function, c'*x
                   )
    m, n = size(a)
    t = vcat(hcat(a, eye(m), b), 
             hcat(zeros(Float64, n)', ones(Float64, m)', [0.0]))
    t = vcat(hcat(eye(m), zeros(Float64, m)),
             hcat(-ones(Float64, m)', [1.0])) * t # tableau
    t, bases = Simplex(t, ones(Int64, m))
    bases = vcat(bases, setdiff(1:n,bases), m+n+1) # Delete auxiliary vectors
    t = vcat(t[1:m,bases], hcat(c', [0.0])) # Shift basis vectors to left `m` columns
    t = vcat(hcat(eye(m), zeros(Float64, m)),
             hcat(-c[1:m]', [1.0]))*t # Start tableau for 2nd phase
    Simplex(t, ones(Int64, m))
end

#...................................
# Simple test cases
#...................................

a = [4 2 -1   0;
     1 4  0  -1];
b = [12, 6];
c = [2, 3, 0, 0];
SimplexTwo(a, b, c)

