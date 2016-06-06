# README #

RescaledExp implementation in tensorflow. RescaledExp is an first-order online convex optimization algorithm that achieves optimal regret in unconstrained problems without knowing a prior bound on gradients.

RescaledExpOptimizer is a coordinate-wise optimizer class
RescaledExpSphereOptimizer is a non-coordinate-wise (and hence dimension-independent) optimizer.
RescaledExpSphere appears to work better on a couple tested neural network problems.