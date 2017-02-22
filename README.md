# README #

This repo contains an inferior algorithm to <a href=https://github.com/acutkosky/freerex>freerex</a>.
It's around only for posterity now.

RescaledExp implementation in tensorflow (https://www.tensorflow.org/). RescaledExp is an first-order online convex optimization algorithm that achieves optimal regret in unconstrained problems without knowing a prior bound on gradients.

RescaledExpOptimizer is a coordinate-wise optimizer class.
RescaledExpSphereOptimizer is an optimizer with dimension-free regret bounds.
RescaledExpSphere appears to work better on a couple tested neural network problems.

# Installation #
Include the python file
`import tensorexp`

Then when instantiating an optimizer class (e.g. in place of `tf.GradientDescentOptimizer`), use `tensorexp.RescaledExpOptimizer` or `tensorexp.RescaledExpSphereOptimizer`.

Each optimizer class takes arguments `learning_rate` and `epsilon`. It should ideally be effective to leave both of these as their default values in all cases (if you are optimizing `learning_rate`, then the algorithm isn't working as desired). `epsilon` is a small constant used for numerical reasons in place of zero in expressions that evaluate to 0/0.
