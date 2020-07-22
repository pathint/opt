using LinearAlgebra

function constant_gradient(f, g, x0, α; 
        ϵx=0.01, # precision for step size
        ϵf=0.01, 
        ϵg=0.01, 
        maxIterations=128,
        debug=false) 
    xk = x0
    fk = f(xk...)
    for i in 1:maxIterations
        # iteration
        d = -g(xk...)
        δ = α*d
        xn = xk .+ δ
        fn = f(xn...)
        # convergence?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(d)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xk, fk, d, δ
        end
        if debug
            println("i=",i, " xk=", xk, " d=", d, " δ= ",δ)
        end
        xk = xn
        fk = fn
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

function constant_gradient(f, g, x0, α; 
        ϵx=0.01, # precision for step size
        ϵf=0.01, 
        ϵg=0.01, 
        maxIterations=128,
        debug=false) 
    xk = x0
    fk = f(xk...)
    pts = []
    for i in 1:maxIterations
        push!(pts, xk)
        # iteration
        d = -g(xk...)
        δ = α*d
        xn = xk .+ δ
        fn = f(xn...)
        # convergence?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(d)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xk, fk, d, δ, pts
        end
        if debug
            println("i=",i, " xk=", xk, " d=", d, " δ= ",δ)
        end
        xk = xn
        fk = fn
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

# using DelimitedFiles
# writedlm("f1.tsv", pts)
x,f,d,delta, pts=constant_gradient(
    (x,y)->0.2x^2+y^2,
    (x,y)->[0.4x; 2y],
    [1.; 1.],
    0.1,
    maxIterations = 1000,
    debug=false
)
writedlm("f1_constant.tsv", pts)

x,f,d,delta, pts=constant_gradient(
    (x,y)->100(y-x^2)^2+(1-x)^2,
    (x,y)->[2(-1+x+200x^3-200x*y); 200(y-x^2)],
    [0.;0.],
    0.002,
    maxIterations = 10000000,
    debug=false
)
writedlm("f2_constant.tsv", pts)


function second_order_gradient(f, g, h, x0;
        ϵx=0.01, # precision for step size
        ϵf=0.01,
        ϵg=0.01,
        maxIterations=128,
        debug=false)
    xk = x0
    fk = f(xk...)
    pts = []
    for i in 1:maxIterations
        # iteration
        push!(pts, xk)
        gk = g(xk...)
        α = gk'*gk/(gk'*h*gk)
        δ = -α .* gk
        xn = xk .+ δ
        fn = f(xn...)
        # convergent?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(gk)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xn, fn, pts
        end
        if debug
            println("i=",i, " α=", α, " xk=", xk, " d=", d, " δ= ",δ)
        end
        xk = xn
        fk = fn
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

arg, value, pts =
second_order_gradient(
    (x,y)->0.2x^2+y^2,
    (x,y)->[0.4x, 2y],
    [0.4 0; 0 2],
    [1.,1.],
    debug=false
)

function second_order_gradient(H, b, x0;
        ϵx=0.01, # precision for step size
        ϵf=0.01,
        ϵg=0.01,
        maxIterations=128,
        debug=false)
    xk = x0
    fk = 0.5*xk'*H*xk-b'*xk
    pts = []
    for i in 1:maxIterations
        # iteration
        push!(pts, xk)
        gk = H*xk - b
        α = gk'*gk/(gk'*H*gk)
        δ = -α .* gk
        xn = xk .+ δ
        fn = 0.5*xn'*H*xn-b'*xn
        # convergent?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(gk)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xn, fn, pts
        end
        if debug
            println("i=",i, " α=", α, " xk=", xk, " d=", d, " δ= ",δ)
        end
        xk = xn
        fk = fn
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

arg, value, pts =
second_order_gradient(
    [0.4 0; 0 24],
    [0; 0],
    [10; 0.2],
    maxIterations=200,
    debug=false
)

arg, value, pts =
second_order_gradient(
    [5 -3; -3 5],
    [4; 4],
    [1/3; 1],
    maxIterations=200,
    debug=false
)

[(results[i+2]-results[i+1])'*(results[i+1]-results[i]) 
  for i in 1:length(results)-2]
    
function steepest_descent(f, g, x0; 
        ϵx=0.01, # precision for step size
        ϵf=0.01, 
        ϵg=0.01, 
        maxIterations=128,
        debug=false) 
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    d = -gk
    for i in 1:maxIterations
        # iteration
        # linear search
        α, δ, xn, fn, gn  = search_alpha(f, g, xk, fk, gk, d)
        # convegence?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(gn)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xn, fn, gn
        end
        if debug
            println("i=",i, " α=", α, " xk=", xn, " d=", dn, " δ= ",δ)
        end
        xk = xn
        fk = fn
        gk = gn
        d = -gk
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

function steepest_descent(f, g, x0; 
        ϵx=0.01, # precision for step size
        ϵf=0.01, 
        ϵg=0.01, 
        maxIterations=128,
        debug=false) 
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    d = -gk
    pts = []
    for i in 1:maxIterations
        # iteration
        push!(pts, xk)
        # linear search
        α, δ, xn, fn, gn  = search_alpha(f, g, xk, fk, gk, d)
        # convegence?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(gn)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xn, fn, gn, pts
        end
        if debug
            println("i=",i, " α=", α, " xk=", xn, " d=", dn, " δ= ",δ)
        end
        xk = xn
        fk = fn
        gk = gn
        d = -gk
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end
#Inexact method, Wolfe condtions
# ϕ(α) <=ϕ(0) + ϵα ϕ'(0)
# ϕ'(0) = d'*g
# ϕ'(α) >= η ϕ'(0)
function search_alpha(f, g, xk, fk, gk, d; α0=1, ϵ=0.1, τ=0.5, η=0.5, ζ=2.0)
    α = α0
    ϕ0= d'*gk
    δ = α .* d
    xn = xk + δ
    fn = f(xn...)
    gn = g(xn...)
    # Armijo condition 
    while fn > fk + ϵ*α*ϕ0
        α = τ*α
        δ = α .* d
        xn = xk + δ
        fn = f(xn...)
        gn = g(xn...)
    end
    # Wolfe condition 
    while d'*gn < η*ϕ0
        α = ζ*α
        δ = α .* d
        xn = xk + δ
        fn = f(xn...)
        gn = g(xn...)
    end
    return α, δ, xn, fn, gn
end

x1, f1, g1, pts = steepest_descent(
    (x,y)->0.2x^2+y^2,
    (x,y)->[0.4x; 2y],
    [1.;1.],
    maxIterations = 1000,
    debug=false
)
writedlm("f1_steep.tsv", pts)

x1, f1, g1, pts = steepest_descent(
    (x,y)->100(y-x^2)^2+(1-x)^2,
    (x,y)->[2(-1+x+200x^3-200x*y); 200(y-x^2)],
    [0; 0],
    maxIterations = 10000,
    debug=false
)
writedlm("f2_steep.tsv", pts)

#Barzilai-Borwein
function bb_descent(f, g, x0; 
        ϵx=0.01, # precision for step size
        ϵf=0.01, 
        ϵg=0.01, 
        maxIterations=128,
        debug=false) 
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    d = -gk
    α, δ, xn, fn, gn  = search_alpha(f, g, xk, fk, gk, d)
    for i in 1:maxIterations
        # convergence?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(gn)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xn, fn, gn
        end
        if debug
            println("i=",i, " α=", α, " xk=", xn, " d=", dn, " δ= ",δ)
        end
        z = gn - gk
        α = (δ'*z)/(z'*z)
        δ = α .* d
        xk = xn
        xn = xn + α *d
        fk = fn
        fn = f(xn...)
        gk = gn
        gn = g(xn...)
        d = -gn
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

function bb_descent(f, g, x0; 
        ϵx=0.01, # precision for step size
        ϵf=0.01, 
        ϵg=0.01, 
        maxIterations=128,
        debug=false) 
    pts = []
    xk = x0
    fk = f(xk...)
    gk = g(xk...)
    d = -gk
    α, δ, xn, fn, gn  = search_alpha(f, g, xk, fk, gk, d)
    push!(pts, xk)
    for i in 1:maxIterations
        push!(pts, xn)
        # convergence?
        if (norm(δ)<=ϵx)&&(abs(fn-fk)<=ϵf)&&(norm(gn)<=ϵg)
            println("Convergence is reached after ", i, " iterations.")
            return xn, fn, gn, pts
        end
        if debug
            println("i=",i, " α=", α, " xk=", xn, " d=", dn, " δ= ",δ)
        end
        z = gn - gk
        α = (δ'*z)/(z'*z)
        δ = α .* d
        xk = xn
        xn = xn + α *d
        fk = fn
        fn = f(xn...)
        gk = gn
        gn = g(xn...)
        d = -gn
    end
    println("WARN:", maxIterations, " iterations have been exceeded!")
end

x,f,g,pts = bb_descent(
    (x,y)->0.2x^2+y^2,
    (x,y)->[0.4x; 2y],
    [1.; 1.],
    maxIterations = 1000,
    debug=false
)
writedlm("f1_bb.tsv", pts)

x1, f1, g1, pts = bb_descent(
    (x,y)->100(y-x^2)^2+(1-x)^2,
    (x,y)->[2(-1+x+200x^3-200x*y); 200(y-x^2)],
    [0; 0],
    maxIterations = 10000,
    debug=false
)
writedlm("f2_bb.tsv", pts)
