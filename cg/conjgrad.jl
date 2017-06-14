###########################################################
#
# Collection of Optimization/Search Algorithms
# Conjugate Gradient-based Algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Conjugate Gradient Algorithm, for quadratic functions or
# systems of linear equations
# Input: function f, gradients and Hessian matrix (constants)
#        start points m0
# Output: minimmum point and value
#..........................................................

function ConjGradQuad(f,  # objective function
                      fd, # Gradient (1st derivative), vector-valued functions
                      hessian, # Hessian matrix, constants
                      start = 0; # Guessed parameters, default = zero vectors  
                      accuracy::Float64 = eps(),
                      verbose::Bool = false)
    # check hessian
    if ndims(hessian) != 2
        return false
    end
    nr, nc = size(hessian)
    if nr != nc 
        return false
    end
    h = Symmetric(hessian) # Force a symmetric Hessian matrix from the upper triangle of `hessian`
    # check gussed starting point
    x0 = start
    if x0 == 0
        x0 = zeros(nc)
    elseif length(x0) != nc
        return false 
    end
    r0 = -fd(x0...)
    rs0= r0'*r0
    d0 = r0
    for i = 1:nc
        a = r0'*r0/(d0'*h*d0)
        x = x0 .+ a*d0
        r = r0 .- a*h*d0
        rs= r'*r 
        if rs < accuracy
            return x, f(x...)
        end
        b = rs/rs0
        d = r + b*d0
        # Verbose output
        if verbose 
            println(i, "\tx=", x, "\tf(x)=", f(x...), "\n\tr=", r, "\tr^2=", rs, "\td=", d)
        end
        x0 = x
        r0 = r
        rs0= rs
        d0 = d
    end
    println(nc, " iterations have been calculated.")
    println("Final step rs = ", rs0)
    return x0, f(x0...)
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
ConjGradQuad((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
             (x1, x2) -> [5.0x1+2x2-3
                          x2+2x1-1],
              [5.0 2.0;
               2.0 1.0],
              verbose=true)

# Mininum Point: x=[1, 0, 0], f(x)=-3/2 
ConjGradQuad((x1, x2, x3) -> (3/2)*x1^2 + 2x2^2 +(3/2)*x3^2 + x1*x3 + 2x2*x3 -3x1 - x3,
             (x1, x2, x3) -> [3x1+x3-3,
                              4x2+2x3,
                              x1+2x2+3x3-1],
              [3 0 1;
               0 4 2;
               1 2 3],
              verbose=true)

#..........................................................
# Conjugate Gradient Algorithm, for general functions
# the so-called nonlinear conjugate gradient algorithm
# Input: function f, gradients
#        start points m0
# Output: minimmum point and value
#..........................................................

function ConjugateGradient(f,  # objective function
                      fd, # Gradient (1st derivative), vector-valued functions
                      start; # Guessed parameters, default = zero vectors  
                      beta_method = "FR", #FR, PR, HS, or DY
                      alpha_method = "Wolfe", #Wolfe, Goldstein, or Armijo
                      alpha::Float64 = 4.0, # Upper alpha for line search 
                      accuracy::Float64 = 0.001,
                      accuracy_1::Float64 = 0.001, # accuracy for line search for optimal alpha
                      accuracy_x::Float64 = accuracy, # arguments | geometry
                      accuracy_f::Float64 = accuracy, # objective | energy
                      accuracy_g::Float64 = accuracy, # gradient  | force
                      convergence_rule = prod, # `prod` means `and (&&)`, `sum` means `or (||)`
                      convergence_abs = true, # absolute convergence threshold or relative threshold
                      iterations::Int64 = 128,
                      verbose::Bool = false)
    # initial iteration
    n = length(start)
    # check gussed starting point
    x0 = start
    # WARNINING: how to check if the vector-valued gradient function matchs with `start`
    # ?? try...catch...
    #if x0 == 0
    #    x0 = zeros(n)
    #elseif length(x0) != n
    #    return false 
    #end
    r0 = -fd(x0...)
    rs0=r0'*r0
    #NOTE: macro is needed!
    # search for alpha0, fa0 is discarded
    if alpha_method == "Armijo"
        a0, fa0 = Armijo(f, x0, r0, f(x0...), -r0)
    elseif alpha_method == "Goldstein"
        a0, fa0 = Goldstein(f, x0, r0, f(x0...), -r0)
    elseif alpha_method == "Wolfe"
        a0, fa0 = Wolfe(f, fd, x0, r0, f(x0...), -r0)
    else
        a0, fa0 = Wolfe(f, fd, x0, r0, f(x0...), -r0)
    end
    d0= r0 .+ a0*r0
    x = x0 .+ a0*d0
    a = a0
    d = d0
    for i = 1:iterations
        r = - fd(x...)
        rs=r'*r
        if mod(i, n) == 0
            b =0
        elseif beta_method == "FR"
            #Fletcher-Reeves
            b = rs/rs0
        elseif beta_method == "PR"
            #Polak-Ribere
            b = max(0, -a0*r'*d0/rs0)
        elseif beta_method == "HS"
            #Hestenes-Stiefel
            b = - r'*d0/(d0'*d0)
        elseif beta_method == "DY"
            #Dai-Yuan
            b = - rs /(a0*d0'*d0)
        else
            b = max(0, -a0*r'*d0/rs0)
        end
        d0 = d
        d = r .+ b*d0
        a0= a
        #Line search for a; fa is discarded
        #a, fa = Armijo(f, x, d, f(x...), -r) 
        #a, fa = Goldstein(f, x, d, f(x...), -r) 
        #a, fa = Wolfe(f, fd, x, d, f(x...), -r) 
        if alpha_method == "Armijo"
            a, fa = Armijo(f, x, d, f(x...), -r) 
        elseif alpha_method == "Goldstein"
            a, fa = Goldstein(f, x, d, f(x...), -r) 
        elseif alpha_method == "Wolfe"
            a, fa = Wolfe(f, fd, x, d, f(x...), -r) 
        else
            a, fa = Wolfe(f, fd, x, d, f(x...), -r) 
        end
        x0= x
        x = x + a*d
        r0=r
        rs0=rs
        # Verbose output
        if verbose 
            println(i, "\tx=", x, "\tf(x)=", f(x...), "\n\tr=", r, "\tr^2=", rs, "\n\ta=", a, "\tb=", b, "\td=", d, "\t|ad|=", vecnorm(a0*d0))
        end
        # Test on convergence
        conditions  = [max(a0*d0...), abs(f(x...)-f(x0...)), rs]
        denominator = [max(1, max(a0*d0...)), max(1, f(x0...)), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        #println(conditions .< [accuracy_x, accuracy_f, accuracy_g])
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", vecnorm(a*d))
            println("Norm of the last gradient:", rs)
            return x, f(x...)
        end
    end
    println("Max. iterations have been calculated.")
    println("Final step r^2 = ", rs0)
    return x0, f(x0...)
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  accuracy_x=0.001,
                  beta_method="FR",
                  verbose=false)

ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  beta_method="HS",
                  accuracy_x=0.01,
                  verbose=false)

ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  beta_method="PR",
                  accuracy_x=0.01,
                  verbose=false)

ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  beta_method="DY",
                  accuracy_x=0.01,
                  verbose=false)

map(b ->  
    map(a ->
        println("\n",a,"\t", b, "\t",
ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  beta_method=b,
                  alpha_method=a,
                  accuracy_x=0.01,
                  verbose=false), "\n\n"),
        ["Wolfe", "Goldstein", "Armijo"]),
    ["FR", "PR", "HS", "DY"])

#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                          200*(y-x^2)],
                [0, 0], 
                beta_method ="HS",
                iterations = 256,
                accuracy=0.0001,
                verbose=false)

               
map(b ->  
    map(a ->
        println("\n",a,"\t", b, "\t",
ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                          200*(y-x^2)],
                [0, 0], 
                beta_method =b,
                alpha_method=a,
                iterations = 256,
                accuracy=0.0001,
                verbose=false),
                "\n\n"),
        ["Wolfe", "Goldstein", "Armijo"]),
    ["FR", "PR", "HS", "DY"])


ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                          200*(y-x^2)],
                [0, 0], 
                beta_method ="FR",
                alpha_method="Wolfe",
                verbose=false)
