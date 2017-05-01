import numpy

from chainer import cuda
from chainer import optimizer


class SSA(optimizer.GradientMethod):

    """Stochastic Simulated Annealing."""

    def __init__(self, var=0.001, initial_beta=1.0, annealing_rate=1.0):
        self.var = var            # Variance to move for next param
        self.beta = initial_beta  # Inverse temperature
        self.ar = annealing_rate

    def prepare(self):
        # Called everytime before update()
        super(SSA, self).prepare()
        self.beta *= self.ar
        if self.t % 1000 == 0:
            print('DEBUG prepare: t {}, beta {}, ar {}'
                  .format(self.t, self.beta, self.ar))

    def update_one_cpu(self, param, state):
        param_size = param.grad.shape
        delta_param = numpy.random.normal(scale=self.var, size=param_size)
        delta_E = delta_param * param.grad
        #adopt_array = numpy.exp(-self.beta * delta_E) > numpy.random.uniform(size=param_size)
        # To avoid overflow, use log instead of exp.
        adopt_array = -self.beta * delta_E > numpy.log(numpy.random.uniform(size=param_size))
        param.data += delta_param * adopt_array

    def update_one_gpu(self, param, state):
        g = param.grad
        param_size = g.shape
        delta_param = cuda.cupy.random.normal(scale=self.var, size=param_size, dtype=g.dtype)
        th = cuda.cupy.log(cuda.cupy.random.uniform(size=param_size, dtype=g.dtype))
        # delta_E = cuda.cupy.empty_like(delta_param)
        cuda.elementwise(
            'T delta_param, T g, T beta, T th',
            'T param',
            '''
            T delta_E = delta_param * g;
            param += (th + beta * delta_E) < 0 ? delta_param : (T)0;
            ''',
            'ssa'
        )(delta_param, g, self.beta, th, param.data)
