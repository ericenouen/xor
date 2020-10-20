# XOR Problem From Scratch

In this project, I was looking to gain more familiarity with how gradients are passed backwards during gradient descent and to utilize that to solve the toy problem of learning the XOR function from scratch.


A couple of links that I used to find more information or used for inspiration.

https://pages.mtu.edu/~nilufer/classes/cs4811/2016-spring/hw01-neural-networks.pdf

https://medium.com/@jayeshbahire/the-xor-problem-in-neural-networks-50006411840b

https://www.ics.uci.edu/~pjsadows/notes.pdf

## XOR Function for Four Possible Inputs
| Input 1 | Input 2 | Output |
|---------|---------|--------|
|    1    |    1    |    0   |
|    1    |    0    |    1   |
|    0    |    1    |    1   |
|    0    |    0    |    0   |

## Architecture
![alt text](nn.svg)
## Backpropagation

### Derivative for Loss

I used Cross Entropy loss for this model which has the following equation:

<a href="https://www.codecogs.com/eqnedit.php?latex=Loss&space;=&space;-(tlog(y)&plus;(1-t)log(1-y))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Loss&space;=&space;-(tlog(y)&plus;(1-t)log(1-y))" title="Loss = -(tlog(y)+(1-t)log(1-y))" /></a>

Then we must take its derivative:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;y}(-tlog(y)-(1-t)log(1-y))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;y}(-tlog(y)-(1-t)log(1-y))" title="\frac{\partial}{\partial y}(-tlog(y)-(1-t)log(1-y))" /></a>

The derivative is equal to the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{y-t}{y(1-y)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{y-t}{y(1-y)}" title="\frac{y-t}{y(1-y)}" /></a>


### Derivative for Sigmoid Function

The Sigmoid Function has the following equation

<a href="https://www.codecogs.com/eqnedit.php?latex=y=\frac{1}{1&plus;e^{-x}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=\frac{1}{1&plus;e^{-x}}" title="y=\frac{1}{1+e^{-x}}" /></a>

Then we must take the derivative

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{d}{dx}(\frac{1}{1&plus;e^{-x}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{d}{dx}(\frac{1}{1&plus;e^{-x}})" title="\frac{d}{dx}(\frac{1}{1+e^{-x}})" /></a>

The derivative is equal to the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{e^{-x}}{(1&plus;e^{-x})^{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{e^{-x}}{(1&plus;e^{-x})^{2}}" title="\frac{e^{-x}}{(1+e^{-x})^{2}}" /></a>

From there we can use the sigmoid equation to get the derivative in terms of y instead of x

<a href="https://www.codecogs.com/eqnedit.php?latex=y(1-y)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(1-y)" title="y(1-y)" /></a>

### Derivative for Linear Function

The Linear Function is shown below

<a href="https://www.codecogs.com/eqnedit.php?latex=y=Wx&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=Wx&plus;b" title="y=Wx+b" /></a>

The derivatives for each parameter must be taken.

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;W}(Wx&plus;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;W}(Wx&plus;b)" title="\frac{\partial}{\partial W}(Wx+b)" /></a>

The derivative is equal to the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a>

Then for the bias parameter:

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial}{\partial&space;b}(Wx&plus;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;b}(Wx&plus;b)" title="\frac{\partial}{\partial b}(Wx+b)" /></a>

The derivative is equal to the following:

<a href="https://www.codecogs.com/eqnedit.php?latex=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?1" title="1" /></a>

### Chain Rule
Chain rule to combine them for the weights and bias terms

## Optimization

For this problem gradient descent will often get stuck in plateaus. There have been a couple of papers written that highlight that these are not local minima but are actually plateus where the loss function stays relatively constant around a certain point. These plateaus result in almost zero gradient which halts gradient descent on the plateau and results in the model to not be able to reach the absolute minimum of the loss function.

Whether the model gets stuck on a plateau is largely dependent on the initialization of the weights. Since the weights are randomly initialized, sometimes the model will converge to the minima and sometimes the model will not converge in a reasonable amount of epochs.
