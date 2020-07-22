using LinearAlgebra

# Gradient descent algorithum
#  w = w - α g
function GradientDescent(X,
	    	  y,
	    	  w;
	    	  f = sigmoid,
	    	  α = 1, 
	    	  ϵ = 0.001, 
		  iterations = 128,
	    	  verbose = false)
  m  = length(y) # number of samples
  wn = w    # do not change the input arguments
  c  = Inf  # cost, root mean square difference, RMSE
  cδ = Inf  # cost changes
  ck = Inf
  i  = 0
  while !(c < ϵ || cδ < ϵ) && i <= iterations 
      i  = i + 1
      v  = X*wn # m*1, sum vector for m samples
      z  = mapreduce(f, hcat, v)
      e  = y .- z[1, 1:end]
      δ  = e .* z[2, 1:end]
      wn = wn .+ α .* X'*δ
      ck = c  # To save the result from the previous step
      c  = sqrt(e'*e/m)
      cδ = abs(c - ck)
      if verbose
         println("c: ", c, "\tδ ", cδ,"\n\te: ", e, "\n\tw: ", wn)
      end
  end
  if i > iterations
     println("WARN: Maximum iterations have been exceeded without convergence.")
  end
  println("RMSE:", c)
  return wn
end

function Kaczmarz(X,
		  y,
		  w;
		  f = sigmoid,
		  μ = 1,
		  ϵ = 0.001, 
		  iterations = 128,
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
	 ek = y[j] - (f(X[j, 1:end]'*wk))[1]
	 wn = wn .+ μ*ek .* X0[j,1:end]
	 if verbose
	    println("w: ", wn, "\te: ",ek)
	 end
      end
      if i > iterations
	 println("WARN: Maximum iterations have been exceeded without convergence.")
         return wn
      end
  end
  return wn
end

# Step size in Gradient Descent algorithum
# for the Sigmoid activation function, which satisfies the following relation.
# f'(v) = f(v)(1-f(v)
function sigmoid(v)
   z = 1/(1+exp(-v))  # y = f(z)
   return [z, z*(1-z)] # f(z), f'(z)
end

function self(v)
   return [v, 1] # f(z), f'(z)
end

#Adaptive Linear Neuron
function adaline(X,
		 y;
		 f = sigmoid, # activation function
		 η = 1, # step size
		 weight = zeros, # Initial weigth method
		 iterations = 128, 
		 bias = false,
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
  w = weight(Float64, c)
  #w = rand(Float64, c)
  #w = zeros(Float64, c)
  if r > c
     w = GradientDescent(X0, y, w, 
			 α = η, 
			 f = f, 
			 iterations = iterations,
			 verbose = verbose)
  else
     w = Kaczmarz(X0, y, w, 
		  μ = η, 
		  f = f, 
		  iterations = iterations,
		  verbose = verbose)
  end
end



X = [ -0.5 -0.5;
      -0.5  0. ;
      -0.5  0.5;
       0.  -0.5;
       0.   0. ;
       0.   0.5;
       0.5 -0.5;
       0.5  0. ;
       0.5  0.5]
y = [-0.42, -0.48, -0.42, 0., 0., 0., 0.42, 0.48, 0.42]
w =
adaline(X, y, 
	η = 1, 
	f = self,
	verbose = true)
X*w .- y

w =
adaline(X, y, 
	η = 1, 
	f = sigmoid,
	verbose = true)
map(v->(sigmoid(v))[1], X*w) .- y

w =
adaline(X, y, 
	η = 1, 
	f = sigmoid,
	bias = true,
	verbose = true)
map(v->(sigmoid(v+w[3]))[1], X*w[1:2]) .- y

# [0.0952, 0.333, -0.238]
adaline([1  2 -1;
	 4  1  3],
	 [1, 0],
	 f = self,
	 verbose = true)


# [5, 3]
adaline([1 -1;
	 0  1],
	 [2, 3],
	 f = self,
	 verbose = true)

