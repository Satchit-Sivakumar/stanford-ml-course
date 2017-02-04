function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
m=size(X,1);
X=[ones(m,1) X]; %Bias unit in input layer
z=X*Theta1';     %Compute input given to hidden layer 
g1=sigmoid(z);   %Compute Logistic Output 
g1=[ones(size(g1,1),1) g1]; %Bias unit in hidden layer
z2=g1*Theta2';   %Compute input given to output layer
h=sigmoid(z2);   %Compute the output 
a=[];
summer=zeros(size(y));   %To sum up the cost function for all labels
for i=1:num_labels
    r=y==i;   %Converting from label into binary
    a=[a r];  %matrix of y vectors in binary for all training examples 
    summer=summer+(-r.*log(h(:,i))-(1-r).*log(1-h(:,i))); %summing over label
end
J=sum(summer)/m;     %Summing over all training examples

%with regularization
J=J+(lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))); 
% We write Theta(:,2:end) to eliminate first column so as to not regularize
% bias unit

%Gradients
Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));
del3=h-a;
Theta=Theta2(:,2:end);
del2=del3*Theta;
del2=del2.*sigmoidGradient(z);
Delta1=del2'*X;
Delta2=del3'*g1;
Theta1_grad=Delta1/m;
Theta2_grad=Delta2/m;

Theta1_grad=Theta1_grad+lambda/m*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad=Theta2_grad+lambda/m*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
