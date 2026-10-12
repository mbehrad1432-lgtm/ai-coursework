# Submitted primitives with portfolio corrections; see README.
import numpy as np

def affine_forward(x, w, b):
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x_reshaped = x.reshape(x.shape[0], -1)  # Reshape to (N, D)
    out = x_reshaped.dot(w) + b  # compute out
    cache = (x, w, b)  # Values we need to compute gradients

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def affine_backward(dout, cache):
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b = cache
    N = x.shape[0]
    M = w.shape[1]

    #assert dout.shape == (N,M)

    # Reshape x to 2D if it's not already
    x_reshaped = x.reshape(N, -1)
    
    dx = dout.dot(w.T)  # Derivative of loss with respect to x
    dx = dx.reshape(x.shape)  # Reshape back to original shape
    
    dw = x_reshaped.T.dot(dout)  # Derivative of loss with respect to w
    db = np.sum(dout, axis=0)  # Derivative of loss with respect to b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

def relu_forward(x):
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    out = np.maximum(0, x)  # compute ReLU(x)
    cache = x

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache

def relu_backward(dout, cache):
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dx = dout * (cache > 0)  # Derivative of loss with respect to x

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def sigmoid_forward(x):
    x = np.asarray(x,dtype=float)
    out = np.empty_like(x)
    mask = x >= 0
    out[mask] = 1 / (1 + np.exp(-x[mask]))
    exponent = np.exp(x[~mask])
    out[~mask] = exponent / (1 + exponent)
    return out, x


def sigmoid_backward(dout,cache):
    out,_ = sigmoid_forward(cache)
    return dout*out*(1-out)


def mse_loss(x,y):
    x,y = np.asarray(x),np.asarray(y)
    if x.size == 0:
        raise ValueError("MSE requires nonempty predictions")
    if x.shape != y.shape:
        # Permit equivalent scalar-regression vector/column shapes only.
        if x.ndim == 2 and x.shape[1] == 1 and y.shape == (x.shape[0],):
            y = y.reshape(x.shape)
        elif x.ndim == 1 and y.shape == (x.shape[0],1):
            y = y.reshape(x.shape)
        else:
            raise ValueError("Prediction and target shapes must match")
    difference = x-y
    return float(np.mean(difference**2)), 2*difference/x.size


class FullyConnectedNet(object):
    """Regression MLP: hidden affine/ReLU blocks, linear output, MSE + L2.

    No classification or softmax implementation is provided.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=8,
        output_dim=1,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:

        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - output_dim: An integer giving the number of regression outputs. It
            is 1 for a regression problem.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        """
        self.category = 'regression'
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Initialize parameters
        dims = [input_dim] + list(hidden_dims) + [output_dim]
    
        for i in range(self.num_layers):
            self.params['W' + str(i+1)] = weight_scale * np.random.randn(dims[i], dims[i+1])
            self.params['b' + str(i+1)] = np.zeros(dims[i+1])


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels / target values, of shape (N,). y[i] gives the
            label / target value for X[i].
        - show_distributions: Boolean that if enabled and y in none, show output
        of each layer to see effects of diffrent approaches

        Returns:
        If y is None, then run a test-time forward pass of the model and return
        scores for a classification problem or the predicted_values for
        a regression problem:
        - out: Array of shape (N, C) / (N, ) giving classification scores / predicted values, where
        scores[i, c] is the classification score for X[i] and class c / predicted_values[i]
        is the predicted value for X[i].


        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores / target values for X and storing them in the out       #
        #  variable.                                                               #
        #                                                                          #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = X
        self.layer_cache = {}
        layer_outputs = []  # Store layer outputs for visualization
        for i in range(self.num_layers - 1):
              out, cache = affine_forward(out, self.params['W' + str(i+1)], self.params['b' + str(i+1)])
              self.layer_cache['affine' + str(i+1)] = cache
        
              out, cache = relu_forward(out)
              self.layer_cache['relu' + str(i+1)] = cache
              layer_outputs.append(out)  # Save output for visualization

    # Forward through output layer
        out, cache = affine_forward(out, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        self.layer_cache['affine' + str(self.num_layers)] = cache
        layer_outputs.append(out)
        # Save output for visualization

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return out

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using mse, and make sure that grads[k] holds the               #
        # gradients for self.params[k]. Don't forget to add L2 regularization!     #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # compute loss
        loss, dout = mse_loss(out, y)

        # Regularization
        reg_loss = 0
        for i in range(self.num_layers):
            W = self.params['W' + str(i+1)]
            reg_loss += 0.5 * self.reg * np.sum(W * W)
        loss += reg_loss

        # Backward pass
        # Backward through output layer
        dout, dw, db = affine_backward(dout, self.layer_cache['affine' + str(self.num_layers)])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db
        
        # Backward through hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            dout = relu_backward(dout, self.layer_cache['relu' + str(i+1)])
            dout, dw, db = affine_backward(dout, self.layer_cache['affine' + str(i+1)])
            grads['W' + str(i+1)] = dw + self.reg * self.params['W' + str(i+1)]
            grads['b' + str(i+1)] = db
            # Regularization

        # Backpropagation affine

        # Regularization


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
    - config: A dictionary containing hyperparameter values such as learning
    rate, momentum.

    Returns:
      - next_w: The next point after the update.
      - config: The config dictionary to be passed to the next iteration of the
        update rule.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces sgd_momentum to stochastic gradient descent.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    v = config["momentum"] * v - config["learning_rate"] * dw  # compute the velocity
    next_w = w + v  # update weights


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config