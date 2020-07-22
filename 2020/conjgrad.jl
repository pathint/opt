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

using LinearAlgebra
# Objective Function
# 1/2 x'Hx - bx
function ConjGradQuad(b,  # 1st order term
                      H,  # Hessian matrix, constants
                      start; # Starting point
                      accuracy::Float64 = eps() ,
                      verbose::Bool = false)
    # check hessian
    if ndims(H) != 2
        return false
    end
    nr, nc = size(H)
    if nr != nc # Square matrix?
        return false
    end
    h = Symmetric(H) # Force a symmetric Hessian matrix from the upper triangle of `hessian`
    # check gussed starting point
    x0 = start
    if length(x0) != nc
        return false 
    end
    r0 = b .- h*x0
    d0 = r0
    pts = []
    push!(pts, x0)
    for i = 1:nc
        α = d0'*r0/(d0'*h*d0)
        x = x0 .+ α.*d0
        r = r0 .- α.*h*d0
        if norm(r) < accuracy
            return x, r, pts
        end
        β = - r'*h*d0/(d0'*h*d0)
        d = r .+ β.*d0
        # Verbose output
        if verbose 
            println(i, "\tx:", x, "\tα:", α, "\tr:", r,"\tβ:", β,"\td:", d)
        end
        x0 = x
        r0 = r
        d0 = d
        push!(pts, x0)
    end
    println(nc, " iterations have been calculated.")
    println("Norm of the final step: ", norm(r0))
    return x0, r0, pts
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
ConjGradQuad( [-3, -1],
              [5.0 2.0;
               2.0 1.0],
              [0, 0], 
              verbose=true)

x,r,pts=
ConjGradQuad( [4, 4],
              [ 5 -3;
               -3  5],
              [1/3, 1], 
              verbose=true)

ConjGradQuad((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
             (x1, x2) -> [5.0x1+2x2-3
                          x2+2x1-1],
              [5.0 2.0;
               2.0 1.0],
              verbose=true)

ConjGradQuad((x1, x2) -> 2.5x1^2+x2^2-3x1*x2-x2 -7,
             (x1, x2) -> [5x1-3x2,
                          2x2-3x1-1],
             [5 -3;
              -3 2],
             verbose = true)

# Mininum Point: x=[1, 0, 0], f(x)=-3/2 
ConjGradQuad((x1, x2, x3) -> (3/2)*x1^2 + 2x2^2 +(3/2)*x3^2 + x1*x3 + 2x2*x3 -3x1 - x3,
             (x1, x2, x3) -> [3x1+x3-3,
                              4x2+2x3,
                              x1+2x2+3x3-1],
              [3 0 1;
               0 4 2;
               1 2 3],
              verbose=true)


#Inexact method, Wolfe condtions
# ϕ(α) <=ϕ(0) + ϵα ϕ'(0)
# ϕ'(0) = d'*g
# ϕ'(α) >= η ϕ'(0)
function inexact_alpha(f, g, xk, fk, gk, d; α0=1, ϵ=0.1, τ=0.5, η=0.5, ζ=2.0)
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
function secant_alpha(f, g, xk, fk, gk, d;α0=1, accuracy = 0.01, maxIter::Int64=128)
    # _l is the current iteration, k-1
    # _u is the next iteration, k
    # to obtain the approximate 2nd derivative, we need to precompute two points first 
    αl = α0
    αu = α0*1.1 # 1.1 is an arbitrary choice 
    xl = xk .+ αl .* d # 1st point
    xu = xk .+ αu .* d # 2nd point
    gl = g(xl...) # gradient
    gu = g(xu...)
    # ϕ'(α)
    ϕl = d'*gl 
    ϕu = d'*gu
    for i = 1:maxIter
        Δ = ϕu - ϕl
	# avoid division-zero error
	if Δ == 0.0
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

# Various implementation for β approximation
# They share the same interface
# Hestenes-Stiefel
function Hestenes_Stiefel(d, gk, gn)
	δ = gn .- gk
	return (gn'*δ)/(d'*δ)
end
function Polak_Ribiere(d, gk, gn)
	return (gn'*(gn .- gk))/(gn'*gk)
end
function Fletcher_Reeves(d, gk, gn)
	return (gn'*gn)/(gk'*gk)
end
function Powell(d, gk, gn)
	return max(0, (gn'*(gn .- gk))/(gk'*gk))
end
function Dai_Yuan(d, gk, gn)
	return (gn'*gn)/(d'*(gn .- gk))
end

# Test Method for Convergence
#

#..........................................................
# Conjugate Gradient Algorithm, for general functions
# the so-called nonlinear conjugate gradient algorithm
# Input: function f, gradients
#        start points m0
# Output: minimmum point and value
#..........................................................

function ConjugateGradient(f,  # objective function
                      g, # Gradient (1st derivative), vector-valued functions
                      x0; # Guessed parameters, default = zero vectors  
                      fβ = Fletcher_Reeves, #Function to calculate β
                      fα = secant_alpha,    #Function to calcualte α
                      accuracy::Float64 = 0.001,
                      accuracy_1::Float64 = 0.001, # accuracy for line search for optimal alpha
                      accuracy_x::Float64 = accuracy, # arguments | geometry
                      accuracy_f::Float64 = accuracy, # objective | energy
                      accuracy_g::Float64 = accuracy, # gradient  | force
		      convergence_rule = x -> x[3], # Test on gradient to true only only
		                                    # x ->reduce(&, x) for all to be true, 
		                                    # x ->reduce(|, x) for any one to be true
                      convergence_abs = true, # absolute convergence threshold or relative threshold
                      iterations::Int64 = 128,
                      verbose::Bool = false)
    # Initial iteration
    n = length(x0) # Dimension
    # NOTE: we should check if "f" and "g" match with x0 in their dimensions.
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    rk = -gk
    d  = rk
    pts = []
    push!(pts, xk)
    for i = 0:iterations
	# Search for optimal alpha
	α, δ, xn, fn, gn  = fα(f, g, xk, fk, gk, d)
	rn = -gn
        if mod(i, n) == 0 # reset search direction to -g every n steps
            β =0
	else
	    β = fβ(d, gk, gn)
        end
	d = rn .+ β*d
	xk = xn
	fk = fn
	gk = gn
	rk = rn
	push!(pts, xk)
        # Verbose output
        if verbose 
	    println(i, "\tx:", xn, "\tf:", fn, "\n\tg:", gn, "\n\tα:", α, "\tδ:", δ, "\tnorm(δ):", norm(δ))
        end
        # Test on convergence
	conditions  = [norm(δ), abs(fn-fk), norm(gn)]
	denominator = [max(1, norm(xk)), max(1, fk), 1]
        if !convergence_abs
            conditions = conditions ./ denominator
        end
	if convergence_rule(conditions .< [accuracy_x, accuracy_f, accuracy_g])
            println("Number of steps: ", i+1)
            println("Norm of the last step size: ", norm(δ))
	    println("Norm of the last gradient:", norm(gn))
            return xn, fn, gn, pts
        end
    end
    println("Max. iterations have been calculated.")
    println("Final step gradient norm:", norm(gk))
    return xk, fk, gk, pts
end

#...................................
# Simple test cases
#...................................

# Mininum Point: x=[1, -1], f(x)=-1 
ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  accuracy=0.001,
                  verbose=false)

map(b ->  
    map(a ->
        println("\n",a,"\t", b, "\t",
ConjugateGradient((x1, x2) -> 2.5x1^2 + 0.5x2^2 +2x1*x2-3x1-x2,
                  (x1, x2) -> [5.0x1+2x2-3
                              x2+2x1-1],
                  [0, 0],
                  fβ=b,
                  fα=a,
                  accuracy=0.01,
                  verbose=false), "\n\n"),
        [secant_alpha, inexact_alpha]),
    [Hestenes_Stiefel, Polak_Ribiere, Fletcher_Reeves, Powell, Dai_Yuan])



#Rosenbrock function, minimum point [a, a^2]
#a=1, b=100
# test on start points: [0.0], [100, 100]
ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2),
                          200*(y-x^2)],
                [0, 0], 
                iterations = 256,
		convergence_rule = x -> reduce(&, x),
		accuracy=0.0001,
                verbose=false)


x, f, g, pts =
ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2),
                          200*(y-x^2)],
                [0, 0], 
		fα = inexact_alpha,
		fβ = Hestenes_Stiefel,
                iterations = 256,
                accuracy=0.0001,
                verbose=false)

ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)-> [-2*(1-x)-400*x*(y-x^2),
                          200*(y-x^2)],
                [0, 0], 
		#fα = inexact_alpha,
		fβ = Hestenes_Stiefel,
                iterations = 256,
                accuracy=0.0001,
                verbose=true)

               
map(b ->  
    map(a ->
        println("\n",a,"\t", b, "\n",
        ConjugateGradient((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
                (x, y)->[-2*(1-x)-400*x*(y-x^2),
                          200*(y-x^2)],
                [0, 0], 
                fβ=b,
                fα=a,
                accuracy=0.01,
                verbose=false), "\n\n"),
        [secant_alpha, inexact_alpha]),
    [Hestenes_Stiefel, Polak_Ribiere, Fletcher_Reeves, Powell, Dai_Yuan])

# Test on Gadfly plot

using Gadfly
set_default_plot_size(25cm, 15cm)

plot(y=[1,3,4,5], Geom.point, Geom.line)

plot([sin, cos], -pi, 2pi)


plot((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
     -1, 2,
     -1, 2,
     Geom.contour)

plot((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
     -1, 2,
     -1, 2,
     Geom.contour(levels=30))

plot(
     x = convert(Vector{Float64}, [pts[i][1] for  i in 1:length(pts)-1]),
     y = convert(Vector{Float64}, [pts[i][2] for  i in 1:length(pts)-1]),
     xend = convert(Vector{Float64}, [pts[i][1] for  i in 2:length(pts)]),
     yend = convert(Vector{Float64}, [pts[i][2] for  i in 2:length(pts)]),
     Scale.x_continuous(minvalue = 0.0, maxvalue = 1.1),
     Scale.y_continuous(minvalue = 0.0, maxvalue = 1.1),
     Geom.point,
     Coord.cartesian(aspect_ratio=1),
     Geom.segment(arrow=true, filled=true))

rosen=
layer((x, y)-> (1-x)^2 + 100*(y-x^2)^2,
     -1, 2,
     -1, 2,
     Geom.contour)

steps=
layer(
     x = convert(Vector{Float64}, [pts[i][1] for  i in 1:length(pts)-1]),
     y = convert(Vector{Float64}, [pts[i][2] for  i in 1:length(pts)-1]),
     xend = convert(Vector{Float64}, [pts[i][1] for  i in 2:length(pts)]),
     yend = convert(Vector{Float64}, [pts[i][2] for  i in 2:length(pts)]),
    # Scale.x_continuous(minvalue = 0.6, maxvalue = 1.1),
    # Scale.y_continuous(minvalue = 0.6, maxvalue = 1.1),
     Geom.point,
     Geom.segment(arrow=true, filled=true))

plot(rosen, 
     steps,
     Scale.x_continuous(minvalue = 0.6, maxvalue = 1.1),
     Scale.y_continuous(minvalue = 0.6, maxvalue = 1.1)
    )

# pts is array of points (2-element array)
function draw_points(pts)
  n = length(pts)
  layer(
       x = convert(Vector{Float64}, [pts[i][1] for  i in 1:n-1]),
       y = convert(Vector{Float64}, [pts[i][2] for  i in 1:n-1]),
       xend = convert(Vector{Float64}, [pts[i][1] for  i in 2:n]),
       yend = convert(Vector{Float64}, [pts[i][2] for  i in 2:n]),
      # Scale.x_continuous(minvalue = 0.6, maxvalue = 1.1),
      # Scale.y_continuous(minvalue = 0.6, maxvalue = 1.1),
       Geom.point,
       Geom.segment(arrow=true, filled=true))
end

p=plot(rosen, 
       draw_points(pts),
     Scale.x_continuous(minvalue = 0.6, maxvalue = 1.1),
     Scale.y_continuous(minvalue = 0.6, maxvalue = 1.1)
    )

draw(PDF("test.pdf"), p)
