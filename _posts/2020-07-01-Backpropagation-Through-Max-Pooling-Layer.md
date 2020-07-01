---
layout: page
data: 2020-07-01
title: "Backpropagation Through Max-Pooling Layer"
---

- > Note: **COPIED** from https://leimao.github.io/blog/Max-Pooling-Backpropagation/

## Introduction 

I have once come up with a question “how do we do back propagation through max-pooling layer?”. The short answer is “there is no gradient with respect to non-maximum values”.

## Proof

Max-pooling is defined as

$$ y = \max(x_1, x_2, \cdots, x_n) $$

where $y$ is the output and $x_i$ is the value of the neuron.

Alternatively, we could consider **max-pooling layer** as an affine layer without bias terms. The weight matrix 
in this affine layer is not trainable though.

Concretely, for the output $y$ after max-pooling, we have

$$y = \sum_{i=1}^{n} w_i x_i$$

where

$$ w_i =  
\begin{cases} 
    1 & \text{if } x_i = \max(x_1, x_2, \cdots, x_n) \\
    0 & \text{otherwise}
\end{cases} $$

The gradient to each neuron is

$$
\begin{aligned}
\frac{\partial y}{\partial x_i} &= w_i \\
&= 
\begin{cases} 
    1 & \text{if } x_i = \max(x_1, x_2, \cdots, x_n) \\
    0 & \text{otherwise}
\end{cases}
\end{aligned}$$

This simply means that the gradient to the neuron is $1$ for the neuron with the maximum value. 
The gradients for all the other neurons are $0$. The neurons whose gradients are $0$ do not contribute 
to the gradients in the earlier neurons due to the chain rule.

## Conclusions

To sum up, there is no gradient with respect to non-maximum values.
