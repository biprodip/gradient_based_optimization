
# Linear models and Optimization

Implementation of linear classifier and training it using stochastic gradient descent modifications and numpy.

## Two-dimensional classification

To make things more intuitive, let's solve a 2D classification problem with synthetic data.

## Logistic regression

To classify objects we will obtain probability of object belongs to class '1'. To predict probability we will use output of linear model and logistic function:
<br>
<p style="text-align: center;">
<!-- $$ a(x; w) = \langle w, x \rangle $$ -->
<!--$$ $$ -->
![Eqn1](http://latex.codecogs.com/gif.latex?a%28x%3B%20w%29%20%3D%20%5Clangle%20w%2C%20x%20%5Crangle)
  <br>
![Eqn2](http://latex.codecogs.com/gif.latex?%5Cinline%20P%28%20y%3D1%20%5C%3B%20%5Cbig%7C%20%5C%3B%20x%2C%20%5C%2C%20w%29%20%3D%20%5Cdfrac%7B1%7D%7B1%20&plus;%20%5Cexp%28-%20%5Clangle%20w%2C%20x%20%5Crangle%29%7D%20%3D%20%5Csigma%28%5Clangle%20w%2C%20x%20%5Crangle%29)
</p>

In logistic regression the optimal parameters w are found by cross-entropy minimization:

Loss for one sample: 
<br>
<p style="text-align: center;">
<!-- $$ l(x_i, y_i, w) = - \left[ {y_i \cdot log P(y_i = 1 \, | \, x_i,w) + (1-y_i) \cdot log (1-P(y_i = 1\, | \, x_i,w))}\right] $$ -->
![Eqn3](http://latex.codecogs.com/gif.latex?%5Cinline%20l%28x_i%2C%20y_i%2C%20w%29%20%3D%20-%20%5Cleft%5B%20%7By_i%20%5Ccdot%20log%20P%28y_i%20%3D%201%20%5C%2C%20%7C%20%5C%2C%20x_i%2Cw%29%20&plus;%20%281-y_i%29%20%5Ccdot%20log%20%281-P%28y_i%20%3D%201%5C%2C%20%7C%20%5C%2C%20x_i%2Cw%29%29%7D%5Cright%5D)
</p>

Loss for many samples: <!-- $$ L(X, \vec{y}, w) =  {1 \over \ell} \sum_{i=1}^\ell l(x_i, y_i, w) $$  -->
<br>
<p style="text-align: center;">
![Eqn4](http://latex.codecogs.com/gif.latex?%5Cinline%20L%28X%2C%20%5Cvec%7By%7D%2C%20w%29%20%3D%20%7B1%20%5Cover%20%5Cell%7D%20%5Csum_%7Bi%3D1%7D%5E%5Cell%20l%28x_i%2C%20y_i%2C%20w%29)
</p>


To train a model with gradient descent, we should compute gradients.

To be specific, we need a derivative of loss function over each weight [6 of them].
<br>
<p style="text-align: center;">
<!-- $$ \nabla_w L = {1 \over \ell} \sum_{i=1}^\ell \nabla_w l(x_i, y_i, w) $$   -->
![Equation 5](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cnabla_w%20L%20%3D%20%7B1%20%5Cover%20%5Cell%7D%20%5Csum_%7Bi%3D1%7D%5E%5Cell%20%5Cnabla_w%20l%28x_i%2C%20y_i%2C%20w%29)
</p>

We have to Figure out a derivative with pen and paper. 

As usual, we've made a small test for you, but if you need more, feel free to check your math against finite differences (estimate how L changes if you shift w by 10^{-5} or so).

## Training
In this section we'll use the functions you wrote to train our classifier using stochastic gradient descent.

You can try change hyperparameters like batch size, learning rate and so on to find the best one, but use our hyperparameters when fill answers.

## Mini-batch SGD

Stochastic gradient descent just takes a random batch of $m$ samples on each iteration, calculates a gradient of the loss on it and makes a step:
<br>
<p style="text-align: center;">
<!-- $$ w_t = w_{t-1} - \eta \dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t) $$  -->
![Equation 6](http://latex.codecogs.com/gif.latex?%5Cinline%20w_t%20%3D%20w_%7Bt-1%7D%20-%20%5Ceta%20%5Cdfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bj%3D1%7D%5Em%20%5Cnabla_w%20l%28x_%7Bi_j%7D%2C%20y_%7Bi_j%7D%2C%20w_t%29)
</p>

## SGD with momentum

Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillations as can be seen in image below. It does this by adding a fraction $\alpha$ of the update vector of the past time step to the current update vector.
<br>
<br>
<p style="text-align: center;">
<!-- $$ \nu_t = \alpha \nu_{t-1} + \eta\dfrac{1}{m} \sum_{j=1}^m \nabla_w l(x_{i_j}, y_{i_j}, w_t) $$ -->
<!-- $$ w_t = w_{t-1} - \nu_t$$-->

![Equation 7](http://latex.codecogs.com/gif.latex?%5Cinline%20%5Cnu_t%20%3D%20%5Calpha%20%5Cnu_%7Bt-1%7D%20&plus;%20%5Ceta%5Cdfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bj%3D1%7D%5Em%20%5Cnabla_w%20l%28x_%7Bi_j%7D%2C%20y_%7Bi_j%7D%2C%20w_t%29)
![Equation 8](http://latex.codecogs.com/gif.latex?%5Cinline%20w_t%20%3D%20w_%7Bt-1%7D%20-%20%5Cnu_t)
</p>

<br>

## RMSprop

Implement RMSPROP algorithm, which use squared gradients to adjust learning rate:
<br>
<!-- $$ G_j^t = \alpha G_j^{t-1} + (1 - \alpha) g_{tj}^2 $$-->
<!-- $$ w_j^t = w_j^{t-1} - \dfrac{\eta}{\sqrt{G_j^t + \varepsilon}} g_{tj} $$-->
<p style="text-align: center;">

![Equation 09](http://latex.codecogs.com/gif.latex?%5Cinline%20G_j%5Et%20%3D%20%5Calpha%20G_j%5E%7Bt-1%7D%20&plus;%20%281%20-%20%5Calpha%29%20g_%7Btj%7D%5E2)
</p>

<p style="text-align: center;">
<br>
![Equation 10](http://latex.codecogs.com/gif.latex?%5Cinline%20w_j%5Et%20%3D%20w_j%5E%7Bt-1%7D%20%20%5Cdfrac%7B%5Ceta%7D%7B%5Csqrt%7BG_j%5Et%20&plus;%20%5Cvarepsilon%7D%7D%20g_%7Btj%7D)
</p>
