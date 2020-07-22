
using LinearAlgebra

function Kaczmarz(X,
		  y,
		  w;
		  μ = 1,
		  ϵ = 0.001, 
		  verbose = false)
  r, c = size(X)
  wk = w
  wn = w
  i = 0
  X0 = mapslices(r -> r .* (1 /(r'*r)), X, dims=[2])
  while i == 0 || norm(wn .- wk) > ϵ
      i  = i + 1
      for j in 1:r
         wk = wn
	 ek = y[j] - X[j, 1:end]'*wk
	 wn = wn .+ μ*ek .* X0[j,1:end]
	 if verbose
	    println("w: ", wn, "\te: ",ek)
	 end
      end
  end
  return wn
end

# Gradient descent algorithum
#  w += - α g
function ConstantGradientDescent(X,
	    	  y,
	    	  w;
	    	  α = 0.1, 
	    	  ϵ = 0.001, 
		  iterations = 128,
	    	  verbose = false)
  r, c = size(X)
  wk = w
  wn = w
  i  = 0
  while i == 0 || norm(wn .- wk) > ϵ
      i  = i + 1
      wk = wn
      ek =  y .- X*wk       # m*n, n*1
      wn = wn .+ α .* X'*ek # n*m, m*1
      if verbose
         println("w: ", wn)
      end
      if i > iterations
	 println("WARN: Maximum iterations have been exceeded without convergence.")
         return wn
      end
  end
  return wn
end


#Adaptive Linear Neuron
function adaline(X,
		 y;
		 step = 1, # Step Size
		 verbose = false
		 )
  #Check if the dimensions match
  if !(ndims(X) == 2 && ndims(y) == 1) # MUST be double '&'
     error("X should be a matrix (dims = 2) and y should be a vector (dim = 1)")
  end
  r, c = size(X)
  l = length(y)
  if l != r
     error("# of rows in X should match with # of elements in y")
  end
  # Initial weights
  w = zeros(Float64, c)
  if r > c
     w = ConstantGradientDescent(X, y, w, α = step, verbose = verbose)
  else
     w = Kaczmarz(X, y, w, μ = step, verbose = verbose)
  end
end

function adaline(X,
		 y;
		 step = 1, # Step Size
		 bias = true,
		 verbose = false
		 )
  #Check if the dimensions match
  if !(ndims(X) == 2 && ndims(y) == 1) # MUST be double '&'
     error("X should be a matrix (dims = 2) and y should be a vector (dim = 1)")
  end
  r, c = size(X)
  l = length(y)
  if l != r
     error("# of rows in X should match with # of elements in y")
  end
  if bias
     X0 = hcat(X, ones(r))
     c  = c + 1
  else
     X0 = X
  end
  # Initial weights
  w = zeros(Float64, c)
  if r > c
     w = ConstantGradientDescent(X0, y, w, α = step, verbose = verbose)
  else
     w = Kaczmarz(X0, y, w, μ = step, verbose = verbose)
  end
end

# [0.0952, 0.333, -0.238]
adaline([1  2 -1;
	 4  1  3],
	 [1, 0],
	 verbose = true)

# [5, 3]
adaline([1 -1;
	 0  1],
	 [2, 3],
	 verbose = false)

# Test Cases
#  XOR True Table
#  0 0 | 0
#  0 1 | 1
#  1 0 | 1
#  1 1 | 0

X =
[0 0;
 0 1;
 1 0;
 1 1]
y=[0, 1, 1, 0]

w =
adaline(X, y,
	step = 0.1,
	verbose = false)

X*w

X*(w[1:2]) .+ w[3]


# Bipolar AND
#  1  1 |  1
#  1 -1 | -1
# -1  1 | -1
# -1 -1 | -1

X =
[1  1;
 1 -1;
-1  1;
-1 -1]
y=[1, -1, -1, -1]

w =
adaline(X, y,
	step = 0.1,
	verbose = false)

X*w

X*(w[1:2]) .+ w[3]
