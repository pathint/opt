###########################################################
#
# Collection of Optimization/Search Algorithms
# Quasi-Newton Algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################

#..........................................................
# Symmetric Rank One (SR1) Algorithm
# Input: function f, gradients, [optional Hessian matrix]
#        start point
# Output: minimmum point and value
#..........................................................

function SR1(f,  # objective function
             fd, # Gradient (1st derivative), vector-valued functions
             start; # Guessed parameters, default = zero vectors  
             #hessian; # Guessed hessian, can be I, the identiify matrix
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
    # if hessian is given for the starting point
    # solve its inverse first
    # here, the first step is the steepest descent method
    x = x0
    g = fd(x0...)
    #if hessian or inverse hessian is given 
    #the search direction should be -hi*g
    hi = eye(n)
    #NOTE: macro is needed!
    # search for alpha, fa is discarded
    if alpha_method == "Armijo"
        a, fa = Armijo(f, x, -g, f(x...), g)
    elseif alpha_method == "Goldstein"
        a, fa = Goldstein(f, x, -g, f(x...), g)
    elseif alpha_method == "Wolfe"
        a, fa = Wolfe(f, fd, x, -g, f(x...), g)
    else
        a, fa = Wolfe(f, fd, x, -g, f(x...), g)
    end
    d= -a*g
    for i = 1:iterations
        x0= x
        g0= g
        fa0=fa
        x = x .+ d
        g = fd(x...)
        gd=g .- g0 # vector
        hg = d .- hi*gd
        hi = hi .+ hg*hg'/(gd'*hg)
        #DFP
        #hg = hi*g
        #hi = hi .+ d*d'/(d'*gd) .- hg*hg'/(gd'*hi*gd)
        d0=d
        d = -hi * g
        # search for alpha0, fa0 is discarded
        if alpha_method == "Armijo"
            a, fa = Armijo(f, x, d, f(x...), g)
        elseif alpha_method == "Goldstein"
            a, fa = Goldstein(f, x, d, f(x...), g)
        elseif alpha_method == "Wolfe"
            a, fa = Wolfe(f, fd, x, d, f(x...), g)
        else
            a, fa = Wolfe(f, fd, x, d, f(x...), g)
        end
        d = a*d
        # Verbose output
        if verbose 
            println(i, "\tx=", x, "\tf(x)=", f(x...))
        end
        # Test on convergence
        conditions  = [max(d0...), abs(fa-fa0), vecnorm(g)]
        denominator = [max(1, max(d0...)), max(1, f(x0...)), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        #println(conditions .< [accuracy_x, accuracy_f, accuracy_g])
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", vecnorm(d0))
            println("Norm of the last gradient:", vecnorm(g))
            return x, f(x...)
        end
    end
    println("Max. iterations have been calculated.")
    println("Final step size = ", vecnorm(d0))
    println("Final gradient  = ", vecnorm(g))
    return x, fa
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
SR1((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    accuracy_x=0.001,
    alpha_method="Wolfe",
    verbose=false)

SR1((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    accuracy_x=0.001,
    alpha_method="Goldstein",
    verbose=false)

SR1((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    accuracy_x=0.001,
    alpha_method="Armijo",
    verbose=false)

#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
SR1((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
    (x, y)-> [-2*(1-x)-400*x*(y-x^2)
              200*(y-x^2)],
    [0, 0], 
    alpha_method ="Wolfe",
    accuracy=0.0001,
    verbose=false)
               
map(a -> 
     SR1((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
          (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                    200*(y-x^2)],
          [0, 0], 
          alpha_method =a,
          accuracy=0.0001,
          verbose=false),
        ["Wolfe", "Goldstein", "Armijo"])

#..........................................................
# Davidon-Fletcher-Powell (DFP) Algorithm
# Input: function f, gradients, [optional Hessian matrix]
#        start points
# Output: minimmum point and value
#..........................................................

function DFP(f,  # objective function
             fd, # Gradient (1st derivative), vector-valued functions
             start; # Guessed parameters, default = zero vectors  
             #hessian; # Guessed hessian, can be I, the identiify matrix
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
    # if hessian is given for the starting point
    # solve its inverse first
    # here, the first step is the steepest descent method
    x = x0
    g = fd(x0...)
    #if hessian or inverse hessian is given 
    #the search direction should be -hi*g
    hi = eye(n)
    #NOTE: macro is needed!
    # search for alpha, fa is discarded
    if alpha_method == "Armijo"
        a, fa = Armijo(f, x, -g, f(x...), g)
    elseif alpha_method == "Goldstein"
        a, fa = Goldstein(f, x, -g, f(x...), g)
    elseif alpha_method == "Wolfe"
        a, fa = Wolfe(f, fd, x, -g, f(x...), g)
    else
        a, fa = Wolfe(f, fd, x, -g, f(x...), g)
    end
    d= -a*g
    for i = 1:iterations
        x0= x
        g0= g
        fa0=fa
        x = x .+ d
        g = fd(x...)
        gd=g .- g0 # vector
        hg = hi*gd
        hi = hi .+ d*d'/(d'*gd) .- hg*hg'/(gd'*hi*gd)
        d0=d
        d = -hi * g
        # search for alpha0, fa0 is discarded
        if alpha_method == "Armijo"
            a, fa = Armijo(f, x, d, f(x...), g)
        elseif alpha_method == "Goldstein"
            a, fa = Goldstein(f, x, d, f(x...), g)
        elseif alpha_method == "Wolfe"
            a, fa = Wolfe(f, fd, x, d, f(x...), g)
        else
            a, fa = Wolfe(f, fd, x, d, f(x...), g)
        end
        d = a*d
        # Verbose output
        if verbose 
            println(i, "\tx=", x, "\tf(x)=", f(x...))
        end
        # Test on convergence
        conditions  = [max(d0...), abs(fa-fa0), vecnorm(g)]
        denominator = [max(1, max(d0...)), max(1, f(x0...)), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        #println(conditions .< [accuracy_x, accuracy_f, accuracy_g])
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", vecnorm(d0))
            println("Norm of the last gradient:", vecnorm(g))
            return x, f(x...)
        end
    end
    println("Max. iterations have been calculated.")
    println("Final step size = ", vecnorm(d0))
    println("Final gradient  = ", vecnorm(g))
    return x, fa
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
DFP((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    accuracy_x=0.001,
    alpha_method="Wolfe",
    verbose=false)

DFP((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    accuracy_x=0.001,
    alpha_method="Goldstein",
    verbose=false)

DFP((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    accuracy_x=0.001,
    alpha_method="Armijo",
    verbose=false)

#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
DFP((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
     (x, y)-> [-2*(1-x)-400*x*(y-x^2)
               200*(y-x^2)],
     [0, 0], 
     alpha_method ="Wolfe",
     iterations = 256,
     accuracy=0.0001,
     verbose=false)

               
map(a -> 
     DFP((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
           (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                     200*(y-x^2)],
           [0, 0], 
           alpha_method =a,
           iterations = 256,
           accuracy=0.0001,
           verbose=false),
        ["Wolfe", "Goldstein", "Armijo"])


#..........................................................
# Broyden-Fletcher-Goldfarb-Shanno (BFGS) Algorithm
# Input: function f, gradients, [optional Hessian matrix]
#        start points m0
# Output: minimmum point and value
#..........................................................

function BFGS(f,  # objective function
              fd, # Gradient (1st derivative), vector-valued functions
              start; # Guessed parameters, default = zero vectors  
              #hessian; # Guessed hessian, can be I, the identiify matrix
              #beta_method = "FR", #FR, PR, HS, or DY
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
    # if hessian is given for the starting point
    # solve its inverse first
    # here, the first step is the steepest descent method
    x = x0
    g = fd(x0...)
    hi = eye(n)
    #NOTE: macro is needed!
    # search for alpha0, fa0 is discarded
    if alpha_method == "Armijo"
        a, fa = Armijo(f, x, -g, f(x...), g)
    elseif alpha_method == "Goldstein"
        a, fa = Goldstein(f, x, -g, f(x...), g)
    elseif alpha_method == "Wolfe"
        a, fa = Wolfe(f, fd, x, -g, f(x...), g)
    else
        a, fa = Wolfe(f, fd, x, -g, f(x...), g)
    end
    d = a*g
    for i = 1:iterations
        x0= x
        g0= g
        fa0=fa
        x = x .+ d
        g = fd(x...)
        gd=g .- g0 # vector
        gx=gd'*d   # scalar
        hi = hi .+ (1 + gd'*hi*gd/gx)/(d'*gd)*d*d' .- (hi*gd*d' .+ d*gd'*hi)/gx
        d0=d
        d = -hi * g
        # search for alpha0, fa0 is discarded
        if alpha_method == "Armijo"
            a, fa = Armijo(f, x, d, f(x...), g)
        elseif alpha_method == "Goldstein"
            a, fa = Goldstein(f, x, d, f(x...), g)
        elseif alpha_method == "Wolfe"
            a, fa = Wolfe(f, fd, x, d, f(x...), g)
        else
            a, fa = Wolfe(f, fd, x, d, f(x...), g)
        end
        d = a*d
        # Verbose output
        if verbose 
            println(i, "\tx=", x, "\tf(x)=", f(x...))
        end
        # Test on convergence
        conditions  = [max(d0...), abs(fa-fa0), vecnorm(g)]
        denominator = [max(1, max(d0...)), max(1, f(x0...)), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
        #println(conditions .< [accuracy_x, accuracy_f, accuracy_g])
        if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i)
            println("Norm of last step size: ", vecnorm(d0))
            println("Norm of the last gradient:", vecnorm(g))
            return x, f(x...)
        end
    end
    println("Max. iterations have been calculated.")
    println("Final step size = ", vecnorm(d0))
    println("Final gradient  = ", vecnorm(g))
    return x, fa
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
BFGS((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
     (x1, x2) -> [5.0x1+2x2-3
                 x2+2x1-1],
     [0, 0],
     accuracy_x=0.001,
     alpha_method="Wolfe",
     verbose=false)

BFGS((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
     (x1, x2) -> [5.0x1+2x2-3
                 x2+2x1-1],
     [0, 0],
     accuracy_x=0.001,
     alpha_method="Goldstein",
     verbose=false)

BFGS((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
     (x1, x2) -> [5.0x1+2x2-3
                 x2+2x1-1],
     [0, 0],
     accuracy_x=0.001,
     alpha_method="Armijo",
     verbose=false)

#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
BFGS((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
      (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                200*(y-x^2)],
      [0, 0], 
      alpha_method ="Wolfe",
      iterations = 256,
      accuracy=0.0001,
      verbose=false)

               
map(a -> 
     BFGS((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
           (x, y)-> [-2*(1-x)-400*x*(y-x^2)
                     200*(y-x^2)],
           [0, 0], 
           alpha_method =a,
           iterations = 256,
           accuracy=0.0001,
           verbose=false),
        ["Wolfe", "Goldstein", "Armijo"])

