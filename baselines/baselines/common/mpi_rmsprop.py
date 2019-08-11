from mpi4py import MPI
import baselines.common.tf_util as U
import tensorflow as tf
import numpy as np

class MpiRmsProp(object):
    def __init__(self, var_list, *, decay=0.9, momentum=0.0, epsilon=1e-08, centered=False, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.decay = decay
        self.momentum = momentum
        # self.beta1 = beta1
        # self.beta2 = beta2
        self.epsilon = epsilon
        self.centered = centered
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(U.numel(v) for v in var_list)

        
        self.mean_square = np.zeros(size, 'float32') 
        self.mom = np.zeros(size, 'float32')
        if centered:
            self.mean_grad = np.zeros(size, 'float32') 
        self.t = 0
        self.setfromflat = U.SetFromFlat(var_list)
        self.getflat = U.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()
        
        self.t += 1
        if self.centered:
            self.mean_grad = self.decay * self.mean_grad + (1 - self.decay) * globalg
            self.mean_square = self.decay * self.mean_square + (1 - self.decay) * (globalg * globalg)
            
            self.mom = self.momentum * self.mom + stepsize * globalg / np.sqrt(\
                    self.mean_square - self.mean_grad ** 2 + self.epsilon)
        else:
            self.mean_square = self.decay * self.mean_square + (1- self.decay) * (globalg * globalg)
            self.mom = self.momentum * self.mom + stepsize * globalg / np.sqrt(\
                    self.mean_square + self.epsilon)
        
        delta = - self.mom
        self.setfromflat(self.getflat() + delta)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

@U.in_session
def test_MpiRmsProp():
    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float32'))
    b = tf.Variable(np.random.randn(2,5).astype('float32'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    update_op = tf.train.RMSPropOptimizer(stepsize, centered=True).minimize(loss)
    do_update = U.function([], loss, updates=[update_op])

    tf.get_default_session().run(tf.global_variables_initializer())
    for i in range(20):
        print(i,do_update())

    tf.set_random_seed(0)
    tf.get_default_session().run(tf.global_variables_initializer())

    var_list = [a,b]
    lossandgrad = U.function([], [loss, U.flatgrad(loss, var_list)], updates=[update_op])
    rmsprop = MpiRmsProp(var_list, centered=True)

    for i in range(20):
        l,g = lossandgrad()
        rmsprop.update(g, stepsize)
        print(i,l)
