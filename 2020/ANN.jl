using LinearAlgebra
using Random

# Artificial Neural Network
# Assuming a three-layer complete-connected network
# Input layer is decided by the input data
# Output layer is decided by the output data, scalar output here only
# Data structure
# Weights 
#  Wh = [w_11 w_21 ...  w_m1; #weights for the input to the 1st node in the hidden layer
#        w_12 w_22 ...  w_m2;  # 2nd node
#          ...
#        w_1l w_2l ...  w_ml]
#  wo = [w_1 w_2 ... w_l] # row vector
# Node values
#  zh = [z_1 g_1;
#        z_2 g_2;
#          ...
#        z_l g_l]
#  yo = [y, g]

function TrainTLN(X, y;
		  l = 3,
		  fh = sigmoid,
		  fo = sigmoid,
		  ϵ  = 0.001,
		  epochs = 16,
		  iterations = 128,
		  weight = zeros,
		  bias = false,
		  verbose = false
		  )
  #Check if the dimensions match
  if !(ndims(X) == 2 && ndims(y) == 1) # MUST be double '&'
     error("X should be a matrix (dims = 2) and y should be a vector (dim = 1)")
  end
  # row, col sizes
  r, c = size(X)
  if length(y) != r
     error("# of rows in X should match with # of elements in y")
  end
  if bias
     X0 = hcat(X, ones(r))
     c  = c + 1
  else
     X0 = X
  end
  # Initial weights
  Wh = weight(Float64, l, c)
  wo = weight(Float64, l)
  cost  = Inf
  cost_old = Inf
  if verbose
     println("Three-layer complete-connection network ")
     println("# input nodes: ", c)
     println("# hidden nodes: ", l)
     println("# ouput nodes: ", 1)
     println("#Initial weights for input-to-hidden,", Wh)
     println("#Initial weights for hidden-to-output,", wo)
  end
  # Iterations over Epochs
  for j in 1:epochs
     # Update on each sample
     # Random shuffle
     idxs = randperm(r) 
     for i in 1:r
	Wh, wo, e = TrainPtn(X0[idxs[i], 1:end], y[idxs[i]], 
                Wh, wo, 
                fh = fh, fo = fo,
                verbose = verbose, 
                iterations = iterations,
                ϵ = ϵ)
        if verbose
            println("\tError over ", i,"th sample:", e)
            println("\tCum. cost:", cost)
        end
     end
     # RMSE cost function
     ypred = mapslices(x -> (Forward(x, Wh, wo))[end][1], X0, dims = [2])
     e = y .- ypred
     cost = sqrt(sum(e.*e)/r)
     if verbose
	println("Error: ", e)
	println("RMSE: ", cost)
     end
     if (abs(cost) < ϵ || abs(cost_old - cost) < ϵ)
	return (Wh, wo, cost)
     end
     cost_old = cost
  end
  return (Wh, wo, cost)
end

# Update the weights incrementally for each sample
function TrainPtn(x, yobs,
		  Wh, wo;
		  fh = sigmoid,
		  fo = sigmoid,
		  ϵ = 0.001,
		  η = 1,
		  iterations = 128,
		  verbose = false
		 )
    Whn= Wh
    won= wo
    v, z, o, y = Forward(x, Whn, won, fh = fh, fo = fo)
    yk = y[1]
    e  = Inf
    for i in 1:iterations
       Whn, won = Backpropagation(x, yobs, Whn, won, v, z, o, y, η = η)
       v, z, o, y = Forward(x, Whn, won, fh = fh, fo = fo)
       yd = y[1] - yk
       yk = y[1]
       e = yobs - yk
       if (abs(e) < ϵ || abs(yd) < ϵ)
          return (Whn, won, e)
       end
       if verbose
          println("y given:", yobs, "\ty ouput:", y[1], "\te:", e)
       end
    end
    println("WARN: convergence has not been reached after", iterations, "iterations.")
    return (Whn, won, e)
end

function Forward(x, 
		 Wh, wo; 
		 fh = sigmoid, 
		 fo = sigmoid)
   # x: m-element 1-d array
   # Wh:l*m array
   # v: l-element array
   v = Wh*x
   # fh return a column vector, 2-element array
   # mapreduce will generate a 2*l array
   # 1st row is the f(v) list
   # 2nd row is the f'(v) list
   z = mapreduce(fh, hcat, v) 
   # row vector * column vector
   o = z[1, 1:end]'*wo
   # 2-element array, f(o) and f'(o)
   y = fo(o)
   return (v, z, o, y)
end


function Backpropagation(x, yobs, # the observed y value
			 Wh, wo, v, z, o, y; η = 1)
   # Update weights for the output layer, first
   # if the output layer consists of more than 1 node,
   #  δ will become a vector
   δ  = (yobs - y[1])*y[2] 
   won = wo .+ (η*δ) .* z[1, 1:end] 
   # Update wights for the hidden layer
   # Wh: l
   # x: m-element
   # z[2, 1:end], row vector, l-element 
   Whn = Wh .+ (((η*δ).*wo) .* z[2, 1:end])*x'
   return (Whn, won)
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


# Unit Test Case
Wh =[0.1 0.3;
     0.3 0.4]
wo = [0.4, 0.6]
x = [0.2, 0.6]
yobs = 0.7
η = 10
v, z, o, y =
Forward(x, Wh, wo)
Backpropagation(x, yobs, 
		Wh, wo,
	        v, z, o, y; η = η)
Wh, wo, e =
TrainPtn(x, yobs, Wh, wo, verbose=true, ϵ = 0.0001)
Forward(x, Wh, wo)

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
Wh, wo, c =
TrainTLN(X, y, l = 3,
	 epochs = 20,
	 weight = zeros,
	 verbose = true)

Forward(X[4, 1:end], Wh, wo)

# Apply the model to predict
mapslices(x -> Forward(x, Wh, wo), X, dims = [2]) 
predict = mapslices(x -> Forward(x, Wh, wo)[4][1], X, dims = [2]) 
e =  predict - y
sqrt((e'*e)/length(y))


norm(mapslices(x -> (Forward(x, Wh, wo))[end][1], X, dims = [2]) .- y)/i

# Test Case: QSAR Oral Toxicity Dataset
data = readdlm("qsar_oral_toxicity.csv", ';', Int, '\n')

using DelimitedFiles
# Test Case: Wisconsin breast cancer dataset 
data = readdlm("wdbc.data", ',', Float64, '\n')
dataX = mapslices(normalize, data[1:end, 3:end], dims = [1])

trainX = dataX[1:400, 1:end]
trainy = data[1:400, 2]
count(x->x==1.0, trainy)
count(x->x==0.0, trainy)

validationX = dataX[401:end, 1:end]
validationy = data[401:end, 2]
count(x->x==1.0, validationy)
count(x->x==0.0, validationy)


Wh, wo, c =
TrainTLN(trainX, trainy, l = 10,
	epochs = 5, 
	weight = rand,
        verbose = false)

prob =
mapslices(x -> (Forward(x, Wh, wo))[end][1], 
	  validationX, dims = [2]) 

predict = 
map(x -> x > 0.5 ? 1.0 : 0.0, prob)

mapreduce(==, +, predict, validationy)

#
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
	 ek = y[j] - f(X[j, 1:end]'*wk)
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

# Gradient descent algorithum
#  w += - α g
function ConstantGradientDescent(X,
	    	  y,
	    	  w;
	    	  f = sigmoid,
	    	  α = 1, 
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
      z  = X*wk # m*n, n*1
      re = map((z,y)->f(z,y), z, y)
      ek = [re[j][2] for j in 1:r] 
      δ  = [re[j][3] for j in 1:r] 
      wn = wn .+ α .* X'*δ
      if verbose
         println("w: ", wn, "\te: ", ek)
      end
      if i > iterations
	 println("WARN: Maximum iterations have been exceeded without convergence.")
         return wn
      end
  end
  return wn
end

# Identity function
function self(z)
   z
end
function self(zi, yo)
   yi = zi      # y = f(z)
   e  = yo - yi # error
   δ  = e       # e*f'(z)
   return (yi, e, δ)
end
# Heaviside step function
function heaviside(z)
   (z > 0) ? 1 : -1
end
function heaviside(zi, yo)
   yi = (zi > 0) ? 1 : -1
   e  = yo - yi
   δ  = e
   return (yi, e, δ)
end
# Sigomid activation function
function sigmoid(z)
   1/(1+exp(-z))
end
# Step size in Gradient Descent algorithum
# for the Sigmoid activation function, which satisfies the following relation.
# f'(v) = f(v)(1-f(v)
function sigmoid(zi, yo)
   yi = 1/(1+exp(-zi)) # y = f(z)
   e  = yo - yi        # error
   δ  = e*yi*(1-yi)    # e*f'(z)
   return (yi, e, δ)
end

#Adaptive Linear Neuron
function adaline(X,
		 y;
		 f = sigmoid, # activation function
		 step = 1, # step size
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
  w = zeros(Float64, c)
  if r > c
     w = ConstantGradientDescent(X0, y, w, 
				 α = step, 
				 f = f, 
				 iterations = iterations,
				 verbose = verbose)
  else
     w = Kaczmarz(X0, y, w, 
		  μ = step, 
		  f = f, 
		  iterations = iterations,
		  verbose = verbose)
  end
end


# [0.0952, 0.333, -0.238]
adaline([1  2 -1;
	 4  1  3],
	 [1, 0],
	 f = self,
	 verbose = false)

adaline([1  2 -1;
	 4  1  3],
	 [1, 0],
	 verbose = false)


# [5, 3]
adaline([1 -1;
	 0  1],
	 [2, 3],
	 f = self,
	 verbose = false)

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
	f = self,
	step = 0.1,
	verbose = false)

X*w

X*(w[1:2]) .+ w[3]

w =
adaline(X, y,
	f = sigmoid,
	step = 0.001,
	bias = true,
	verbose = true)

X*w

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
	step = 0.001,
	iterations = 1000,
	bias = true,
	verbose = false)

X*w

X*(w[1:2]) .+ w[3]
