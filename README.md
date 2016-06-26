# README #

RescaledExp implementation in tensorflow. RescaledExp is an first-order online convex optimization algorithm that achieves optimal regret in unconstrained problems without knowing a prior bound on gradients.

RescaledExpOptimizer is a coordinate-wise optimizer class.
RescaledExpSphereOptimizer is an optimizer with dimension-free regret bounds.
RescaledExpSphere appears to work better on a couple tested neural network problems.

# Installation #
Include the python file
`import tensorexp`

Then when instantiating an optimizer class (e.g. in place of `tf.GradientDescentOptimizer`), use `tensorexp.RescaledExpOptimizer` or `tensorexp.RescaledExpSphereOptimizer`.

Each optimizer class takes arguments `learning_rate` and `epsilon`. It should be safe to leave both of these as their default values in all cases. `learning_rate` will only affect some constants in the regret analysis (in contrast to learning rates in most algorithms) and `epsilon` is a small constant used for numerical reasons in place of zero to in expressions that evaluate to 0/0.
