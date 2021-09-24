# END-3.0  Assignment 1 

##Code

Creating a XOR gate using Neural Network 

Here is an XOR truth table:

![](Resources/001.png)

We want to predict the output based on the two inputs A and B.

At first, for such a simple task, training a perceptron via updating its weights and bias’s may seem like a good idea. However, XOR is not a non-linear function so a perceptron on its own cannot learn how to predict this.


![](Resources/002.png)

Above is the equation for a single perceptron with no activation function. It takes a transpose of the input vector, multiplies it with the weight vector and adds the bias vector. In this case, the input vector is [0, 0], [0, 1], [1, 0] or [1, 1]. Alone, this basic perceptron, no matter what the weights and bias are, can never accurately predict the XOR output since it is a linear function and XOR is not.

Minsky and Papert proved using Perceptrons  that neural networks is incapable of learning very simple functions. Learnings by perceptron in a 2-D space is shown in image 2. They chose Exclusive-OR as one of the example and proved that Perceptron doesn’t have ability to learn X-OR. As described in image 3, X-OR is not separable in 2-D. So, perceptron can’t propose a separating plane to correctly classify the input points.

![](Resources/003.png)

![](Resources/004.png)

So we can effectively Summarize that a single neuron/perceptron is completely incapable of solving an non linear problem no matter what the weights and biases be , it can only be solved once we convert it into non-linear space. This is one of the classical examples why activation functions are so important and how they aid in the overall learning of feature by allowing to reach a separable threshold.

Multi layer perceptron are the networks having stack of neurons and multiple layers. A basic neuron in modern architectures looks like image 4:

![](Resources/005.png)

Image 4: Single Neuron

Each neuron is fed with an input along with associated weight and bias. A neuron has two functions:

\1) Accumulator function: It essentially is the weighted sum of input along with a bias added to it.
\2) Activation function: Activation functions are non-linear function. And as the name suggests is a function to decide whether output of a node will be actively participating in the overall output of the model or not. 

Now to describe a process of solving X-OR with the help of MLP with one hidden layer. So, our model will have an input layer, one hidden layer and an output layer. Out model will look something like below image :

![](Resources/006.png)

To solve this issue of a **single perceptron being linear**, we need to add another layer to the network. This extra layer is called **the sigmoid function**. The sigmoid function is a nonlinear activation function:

![](Resources/007.png)

Source: <https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html>

If 0 were input into this function, as seen in diagram above, the output would be 0.5. An example in code is shown below:

![](Resources/008.png)
```{python}
import torch.nn.functional as F
print(F.sigmoid(torch.tensor([0])))  >> tensor([0.500])
```
In the above code, the PyTorch library ‘functional’ containing the sigmoid function is imported. A tensor with the value 0 is passed into the sigmoid function and the output is printed. The output is 0.5.

After the sigmoid activation function another perceptron will be added as the final layer. 

**Implementation details :** 

Function involves minimalistic changes from assignment

1. As per assignment directive we have to remove the final layer with TanH.
1. It has 11 filters/Kernel instead of 2 from original architecture
1. Also the Bias is kept only on first layer to account for scenario wherein both inputs are 0 .
1. Bias for second layer is more or less redundant hence turned off
1. ***Activation function*** -> TanH
1. ***Number of parameters*** -> 44

\**Training overview \**

1. **Epochs -> 2001**
1. **Loss function -> L1 loss (original function )**
1. **Learning Rate -> 0.03**

![](Resources/0005.png)

**Issue** : Gives -0 for matching outputs surely the result of improper activation function and loss function as the loss topology as visualized above is too erratic.

tensor([1.], grad\_fn=<RoundBackward>)

tensor([1.], grad\_fn=<RoundBackward>)

tensor([-0.], grad\_fn=<RoundBackward>)

tensor([0.], grad\_fn=<RoundBackward>)

**Replacing Loss function and using Sigmoid as Activation function.**

1. We will first change the loss function as L1 loss is designed to penalize the residuals. So we replace it with the most simplistic MSE loss function which is differentiable as well
1. If that does not work we will try to replace Activation function from TanH to Sigmoid whose non linear output values always lies between 0 to 1. We will not go with RELU because We have first off binary input and a single logic output is needed.

import torch.nn as nn 

class XOR(nn.Module):

`    `def \_\_init\_\_(self, input\_dim = 2, output\_dim=1):

`        `super(XOR, self).\_\_init\_\_()

`        `self.lin1 = nn.Linear(input\_dim, 11)

`        `self.lin2 = nn.Linear(11, output\_dim,bias=False)



`    `def forward(self, x):

`        `x = self.lin1(x)

`        `x = F.sigmoid(x)

`        `x = self.lin2(x)

`        `# x = F.sigmoid(x)

`        `return x


XOR(

`  `(lin1): Linear(in\_features=2, out\_features=11, bias=True)

`  `(lin2): Linear(in\_features=11, out\_features=1, bias=False)

)

\----------------------------------------------------------------

`        `Layer (type)               Output Shape         Param #

\================================================================

`            `Linear-1                [-1, 2, 11]              33

`            `Linear-2                 [-1, 2, 1]              11

\================================================================

Total params: 44

Trainable params: 44

Non-trainable params: 0

\----------------------------------------------------------------

Input size (MB): 0.00

Forward/backward pass size (MB): 0.00

Params size (MB): 0.00

Estimated Total Size (MB): 0.00

\----------------------------------------------------------------

Epoch: 0 completed

/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:1805: UserWarning: nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.

`  `warnings.warn("nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.")

Epoch: 500 completed

Epoch: 1000 completed

Epoch: 1500 completed

Epoch: 2000 completed

![](Resources/009.png)

tensor([1.], grad\_fn=<RoundBackward>)

tensor([1.], grad\_fn=<RoundBackward>)

tensor([0.], grad\_fn=<RoundBackward>)

tensor([0.], grad\_fn=<RoundBackward>)



0. ***Also review a secondary solution designed by us, we also have put in a consolidated file with different experiments conducted by us. Any Feedback is appreciated.***

> Consolidated Code Training Fragment.ipynb



