
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
import tensorflow as tf
import numpy as np

class RescaledExpOptimizer(optimizer.Optimizer):
  """Optimizer that implements the rescapedexp algorithm.


  @@__init__
  """

  def __init__(self, epsilon=1e-8,
               use_locking=False, name="RescaledExp"):
    """Construct a new rescaledexp optimizer.
    """
    super(RescaledExpOptimizer, self).__init__(use_locking, name)
    self._epsilon = epsilon

    self._k = np.sqrt(2)
    self._initialized = False

    # Tensor versions of the constructor arguments, created in _prepare().
    self._epsilon_t = None

  def _create_slots(self, var_list):
    # Create the beta1 and beta2 accumulators on the same device as the first
    # variable.
    for v in var_list:
        with ops.device(v.device):
            Gsq = constant_op.constant(self._epsilon,shape = v.get_shape())
            Gsum = constant_op.constant(0.0,shape = v.get_shape())
            L = constant_op.constant(self._epsilon,shape = v.get_shape())
            M = constant_op.constant(0.0,shape = v.get_shape())
            center = constant_op.constant(0.0,shape = v.get_shape())
            initialized = constant_op.constant(0.0,shape = v.get_shape())
        self._get_or_make_slot(v,Gsq,"Gsq",self._name)
        self._get_or_make_slot(v,Gsum,"Gsum",self._name)
        self._get_or_make_slot(v,L,"L",self._name)
        self._get_or_make_slot(v,Gsq,"M",self._name)
        self._get_or_make_slot(v,center,"center",self._name)
        self._get_or_make_slot(v,initialized,"initialized",self._name)

  def _prepare(self):
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

  def _apply_dense(self, grad, var):

      L = self.get_slot(var,"L")
      Gsq = self.get_slot(var,"Gsq")
      Gsum = self.get_slot(var,"Gsum")
      M = self.get_slot(var,"M")
      center = self.get_slot(var,"center")
      initialized = self.get_slot(var,"initialized")



      epsilon_vector = constant_op.constant(self._epsilon,shape=var.get_shape())
      zero_vector = constant_op.constant(0.0,shape=var.get_shape())


      center_t = tf.select(tf.equal(initialized,zero_vector),var,center)
      resets = tf.abs(grad)>2*L

      k = self._k




      Gsq_t = Gsq+tf.square(grad)
      Gsum_t =Gsum+grad
      M_t = tf.maximum(M,tf.abs(Gsum_t)*L-Gsq_t)

      eta = 1.0/(k*tf.sqrt(2*(M_t+Gsq_t)))

      w_t = -tf.sign(Gsum_t)*(tf.exp(eta*tf.abs(Gsum_t))-1)

      Gsq_update = state_ops.assign(Gsq,tf.select(resets,epsilon_vector,Gsq_t))
      Gsum_update = state_ops.assign(Gsum,tf.select(resets,zero_vector,Gsum_t))
      M_update = state_ops.assign(M,tf.select(resets,zero_vector,M_t))

      L_update = state_ops.assign(L,tf.select(resets,tf.abs(grad),L))
      var_update = state_ops.assign(var,tf.select(resets,
          zero_vector,w_t)+center_t)
      center_update = state_ops.assign(center,center_t)
      initialized_update = \
      state_ops.assign(initialized,constant_op.constant(1.0,shape=var.get_shape()))

      return control_flow_ops.group(*[var_update,Gsq_update,Gsum_update,
                                        L_update,M_update,center_update,
                                        initialized_update])



  def _apply_sparse(self, grad, var):
    return self._appy_dense(grad,var)
