###########################################################
#
# Collection of Optimization/Search Algorithms
# Quasi-Newton Algorithms
# Xianlong Wang, Ph.D.
# Wang.Xianlong@139.com
# *** FOR TEACHING PURPOSE ONLY ***
###########################################################


using LinearAlgebra

#Inexact method, Wolfe condtions
# ϕ(α) <=ϕ(0) + ϵα ϕ'(0)
# ϕ'(0) = d'*g
# ϕ'(α) >= η ϕ'(0)
# NOTE: accuracy is not used, just for the same interface.
function inexact_alpha(f, g, xk, fk, gk, d; α0=1, ϵ=0.1, τ=0.5, η=0.5, ζ=2.0, accuracy)
    α = α0
    ϕ0= d'*gk
    δ = α .* d
    xn = xk .+ δ
    fn = f(xn...)
    gn = g(xn...)
    # Armijo condition 
    while fn > fk + ϵ*α*ϕ0
        α = τ*α
        δ = α .* d
        xn = xk .+ δ
        fn = f(xn...)
        gn = g(xn...)
    end
    # Wolfe condition 
    while d'*gn < η*ϕ0
        α = ζ*α
        δ = α .* d
        xn = xk .+ δ
        fn = f(xn...)
        gn = g(xn...)
    end
    return α, δ, xn, fn, gn
end

# Search for alpha using the secant method
# looking for ϕ'(a) = 0
# ϕ'(α) = d'*g
# use the same interface as "search_alpha"
# ϵ is the accuracy here.
# x(k+1)  := (x(k) -(x(k-1)))/(f'(x(k)) - f'(x(k-1))) * f'(x(k)) 
function secant_alpha(f, g, xk, fk, gk, d; accuracy = 0.01, maxIter::Int64=128)
    # _l is the current iteration, k-1
    # _u is the next iteration, k
    # to obtain the approximate 2nd derivative, we need to precompute two points first 
    αl = 0
    αu = 0.001 #  0.001 is an arbitrary choice 
    xl = xk # 1st point
    xu = xk .+ αu .* d # 2nd point
    gl = gk # gradient
    gu = g(xu...)
    # ϕ'(α)
    ϕl = d'*gl 
    ϕu = d'*gu
    zero = eps()
    for i = 1:maxIter
        Δ = ϕu - ϕl
	# avoid division-zero error
	if abs(Δ) <= zero 
            println("ERROR: the appr. 2nd derivative is 0.")
	    return αu, αu.*d, xu, f(xu...), gu
        else
            Δ = ϕu*(αu-αl)/Δ
        end
        if abs(Δ) <= abs(accuracy)
	    return αu, αu.*d, xu, f(xu...), gu
        end
        αl = αu
	xl = xu
        gl = gu
	ϕl = ϕu
	αu = αu - Δ
        xu = xk .+ αu .* d
        gu = g(xu...)
	ϕu = d'*gu
    end
    println("WARNING: accuracy is not reached yet. Please increase iteration numbers")
    return αu, αu.*d, xu, f(xu...), gu
end

#..........................................................
# Symmetric Rank One (SR1) Algorithm
# Input: function f, gradients, [optional Hessian matrix]
#        start point
# Output: minimmum point and value
#..........................................................

function SR1(f,  # objective function
             g, # Gradient (1st derivative), vector-valued functions
             x0; # Guessed parameters
             B0 = I, # Guessed inverse hessian, default = I, the identity matrix
             fα = secant_alpha,  # Default method to search step size
             α0::Float64 = 4.0, # Upper alpha for line search 
             ϵ ::Float64 = 0.001,
             ϵ1::Float64 = 0.001, # accuracy for line search for optimal alpha
             ϵx::Float64 = ϵ, # arguments | geometry
             ϵf::Float64 = ϵ, # objective | energy
             ϵg::Float64 = ϵ, # gradient  | force
	     convergence_rule = x -> x[3], # Test on gradient to true only only
                              # x ->reduce(&, x) for all to be true,
                              # x ->reduce(|, x) for any one to be true
             convergence_abs = true, # absolute convergence threshold or relative threshold
             iterations::Int64 = 128,
             verbose::Bool = false)
    # TODO: check on input
    # initial step
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    # NOTE: If the default I is used, its dimensions are unknown.
    # So we need to let the compiler to know its sizes
    n  = length(xk)
    B  = zeros(Float64, n, n) + B0
    pts = []
    push!(pts, xk)
    for i = 1:iterations
        d = -B*gk
        # search for optimal step size
        α, δ, xn, fn, gn  = fα(f, g, xk, fk, gk, d, accuracy = ϵ1)
	push!(pts, xn)
        if verbose 
		println(i, "\tx:", xn, "\tf:", fn, "\n\tg:", gn, "\n\tα:", α, "\tδ:", δ, "\tnorm(δ):", norm(δ), "\teigenvalues:", eigvals(B))
        end
        # Test on convergence
	conditions  = [norm(δ),	abs(fn-fk), norm(gn)]
	denominator = [max(1, norm(xk)), max(1, fk), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
	if convergence_rule(conditions .< [ϵx, ϵf, ϵg])
            println("Number of steps: ", i)
            println("Norm of the last step size: ", norm(δ))
	    println("Norm of the last gradient:", norm(gn))
            return xn, fn, gn, pts
        end
	#Update on g
	δg = gn .- gk
	z  = δ .- B*δg
	B  = B .+ (1/(δg'*z) .* z)*z'
	xk = xn
	fk = fn
	gk = gn
    end
    println("Max. iterations have been calculated.")
    println("Final gradient  = ", norm(gk))
    return xk, fk, gk, pts
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
x, f, g, pts =
SR1((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3,
                 x2+2x1-1],
    [0, 0],
    verbose=true)

SR1((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3,
                x2+2x1-1],
    [0, 0],
    fα = inexact_alpha,
    verbose=true)

#4-dimension Powell function, minimum point [0, 0..., 0]
# search range x_i belongs to (-4, 5)
SR1( (x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
     (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                         20(x1+10x2)+4(x2-2x3)^3
                         10(x3-x4)-8(x2-2x3)^3
                         -10(x3-x4)-40(x1-x4)^3],
      [3, -1, 0, 1], 
      ϵ1 = 0.0001,
      verbose=false)

#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
x, f, g, pts =
SR1((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
    (x, y)-> [-2*(1-x)-400*x*(y-x^2),
               200*(y-x^2)],
    [0, 0], 
    ϵ1  = 0.1,
    iterations = 128,
    verbose=false)

#..........................................................
# Davidon-Fletcher-Powell (DFP) Algorithm
# Input: function f, gradients, [optional Hessian matrix]
#        start points
# Output: minimmum point and value
#..........................................................

function DFP(f,  # objective function
             g, # Gradient (1st derivative), vector-valued functions
             x0; # Guessed parameters
             B0 = I, # Guessed inverse hessian, default = I, the identity matrix
             fα = secant_alpha,  # Default method to search step size
             α0::Float64 = 4.0, # Upper alpha for line search 
             ϵ ::Float64 = 0.001,
             ϵ1::Float64 = 0.001, # accuracy for line search for optimal alpha
             ϵx::Float64 = ϵ, # arguments | geometry
             ϵf::Float64 = ϵ, # objective | energy
             ϵg::Float64 = ϵ, # gradient  | force
	     convergence_rule = x -> x[3], # Test on gradient to true only only
                              # x ->reduce(&, x) for all to be true,
                              # x ->reduce(|, x) for any one to be true
             convergence_abs = true, # absolute convergence threshold or relative threshold
             iterations::Int64 = 128,
             verbose::Bool = false)
    # TODO: check on input
    # initial step
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    # NOTE: If the default I is used, its dimensions are unknown.
    # So we need to let the compiler to know its sizes
    n  = length(xk)
    B  = zeros(Float64, n, n) + B0
    pts = []
    push!(pts, xk)
    for i = 1:iterations
        d = -B*gk
        # search for optimal step size
        α, δ, xn, fn, gn  = fα(f, g, xk, fk, gk, d, accuracy = ϵ1)
	push!(pts, xn)
        if verbose 
		println(i, "\tx:", xn, "\tf:", fn, "\n\tg:", gn, "\n\tα:", α, "\tδ:", δ, "\tnorm(δ):", norm(δ), "\teigenvalues:", eigvals(B))
        end
        # Test on convergence
	conditions  = [norm(δ),		abs(fn-fk), norm(gn)]
	denominator = [max(1, norm(xk)), max(1, fk), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
	if convergence_rule(conditions .< [ϵx, ϵf, ϵg])
            println("Number of steps: ", i)
            println("Norm of the last step size: ", norm(δ))
	    println("Norm of the last gradient:", norm(gn))
            return xn, fn, gn, pts
        end
	#Update on g
	δg = gn .- gk
	z  = B*δg
	B  = B .+ (1/(δ'*δg) .* δ)*δ' .- (1/(δg'*z) .* z)*z'
	xk = xn
	fk = fn
	gk = gn
    end
    println("Max. iterations have been calculated.")
    println("Final gradient  = ", norm(gk))
    return xk, fk, gk, pts
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
x, f, g, pts=
DFP((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    verbose=false)

DFP((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    fα = inexact_alpha,
    verbose=false)

DFP((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
    (x1, x2) -> [5.0x1+2x2-3
                x2+2x1-1],
    [0, 0],
    verbose=false)

#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
x, f, g, pts=
DFP((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
     (x, y)-> [-2*(1-x)-400*x*(y-x^2)
               200*(y-x^2)],
     [0, 0], 
     ϵ1 = 0.01,
     verbose=false)

DFP( (x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
     (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                         20(x1+10x2)+4(x2-2x3)^3
                         10(x3-x4)-8(x2-2x3)^3
                         -10(x3-x4)-40(x1-x4)^3],
      [3, -1, 0, 1], 
      ϵ1 = 0.01,
      verbose=false)

#..........................................................
# Broyden-Fletcher-Goldfarb-Shanno (BFGS) Algorithm
# Input: function f, gradients, [optional Hessian matrix]
#        start points m0
# Output: minimmum point and value
#..........................................................

function BFGS(f,  # objective function
             g, # Gradient (1st derivative), vector-valued functions
             x0; # Guessed parameters
             B0 = I, # Guessed inverse hessian, default = I, the identity matrix
             fα = secant_alpha,  # Default method to search step size
             α0::Float64 = 4.0, # Upper alpha for line search 
             ϵ ::Float64 = 0.001,
             ϵ1::Float64 = 0.001, # accuracy for line search for optimal alpha
             ϵx::Float64 = ϵ, # arguments | geometry
             ϵf::Float64 = ϵ, # objective | energy
             ϵg::Float64 = ϵ, # gradient  | force
	     convergence_rule = x -> x[3], # Test on gradient to true only only
                              # x ->reduce(&, x) for all to be true,
                              # x ->reduce(|, x) for any one to be true
             convergence_abs = true, # absolute convergence threshold or relative threshold
             iterations::Int64 = 128,
             verbose::Bool = false)
    # TODO: check on input
    # initial step
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    # NOTE: If the default I is used, its dimensions are unknown.
    # So we need to let the compiler to know its sizes
    n  = length(xk)
    B  = zeros(Float64, n, n) + B0
    pts = []
    push!(pts, xk)
    for i = 1:iterations
        d = -B*gk
        # search for optimal step size
        α, δ, xn, fn, gn  = fα(f, g, xk, fk, gk, d, accuracy = ϵ1)
	push!(pts, xn)
        if verbose 
		println(i, "\tx:", xn, "\tf:", fn, "\n\tg:", gn, "\n\tα:", α, "\tδ:", δ, "\tnorm(δ):", norm(δ), "\teigenvalues:", eigvals(B))
        end
        # Test on convergence
	conditions  = [norm(δ),		abs(fn-fk), norm(gn)]
	denominator = [max(1, norm(xk)), max(1, fk), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
	if convergence_rule(conditions .< [ϵx, ϵf, ϵg])
            println("Number of steps: ", i)
            println("Norm of the last step size: ", norm(δ))
	    println("Norm of the last gradient:", norm(gn))
            return xn, fn, gn, pts
        end
	#Update on g
	δg = gn .- gk
	de= 1/(δg'*δ)
	z  = B*δg
	z2 = z*δ'
	B  = B .+ (1 + de * (δg'*z))*de .* δ*δ' .- de.*(z2+z2')
	xk = xn
	fk = fn
	gk = gn
    end
    println("Max. iterations have been calculated.")
    println("Final gradient  = ", norm(gk))
    return xk, fk, gk, pts
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
BFGS((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
     (x1, x2) -> [5.0x1+2x2-3,
                 x2+2x1-1],
     [0, 0],
     verbose=true)

BFGS((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
     (x1, x2) -> [5.0x1+2x2-3,
                 x2+2x1-1],
     [0, 0],
     fα=inexact_alpha,
     verbose=false)


#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
x, f, g, pts = 
BFGS((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
     (x, y)-> [-2*(1-x)-400*x*(y-x^2),
                200*(y-x^2)],
      [0, 0], 
      #ϵ1  = 0.00001,
      fα = inexact_alpha,
      verbose=false)

               
BFGS( (x1, x2, x3, x4) -> (x1+10x2)^2+5(x3-x4)^2+(x2-2x3)^4+10(x1-x4)^4,
     (x1, x2, x3, x4) -> [2(x1+10x2)+40(x1-x4)^3
                         20(x1+10x2)+4(x2-2x3)^3
                         10(x3-x4)-8(x2-2x3)^3
                         -10(x3-x4)-40(x1-x4)^3],
      [3, -1, 0, 1], 
      #ϵ1 = 0.1,
      #fα = inexact_alpha,
      verbose=false)
