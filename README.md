# README #

RescaledExp implementation in tensorflow. RescaledExp is an first-order online convex optimization algorithm that achieves optimal regret in unconstrained problems without knowing a prior bound on gradients.

RescaledExpOptimizer is a coordinate-wise optimizer class.
RescaledExpSphereOptimizer is an optimizer with dimension-free regret bounds.
RescaledExpSphere appears to work better on a couple tested neural network problems.

# Installation #
Include the python file
`import tensorexp`

Then when instantiating an optimizer class (see e.g. in place of `tf.GradientDescentOptimizer`), use `tensorexp.RescaledExpOptimizer` or `tensorexp.RescaledExpSphereOptimizer`.
