import numpy

from chainer import optimizer
from chainer import cuda


class HessianFree(optimizer.Optimizer):

    """Base class of all single gradient-based optimizers.

    See:
    http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization
    http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_Martens10.pdf

     Hessian Free optmization is second order optimization method.
     It uses conjugate gradient method to avoid calculating inverse of Hessian,
     which is required in Newton method.
     It also uses finite difference coefficient to calculate the product of
     Hessian and vector.
     """
    def __init__(self,  epsilon=1e-5):
        self.epsilon = epsilon
        self.init = True
        self.calc_inner_product_sum = cuda.reduce(
            'T u, T v',
            'T sum_uv',
            'u * v',
            'a + b',
            'sum_uv = a',
            '0',
            'calc_inner_product_sum'
        )

    def init_state(self, param, state):
        self.init = True

        xp = cuda.get_array_module(param.data)
        with cuda.get_device(param.data):
            state['d'] = xp.zeros_like(param.data)
            state['a'] = xp.zeros_like(param.data)
            state['b'] = xp.zeros_like(param.data)
            state['nabla'] = xp.zeros_like(param.data)
            state['hd'] = xp.zeros_like(param.data)



    def update(self, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then use it as a loss function to compute
          gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        """

        # First forward-backward
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            del loss
        else:
            print('Error lossfun must be set in argument for update in this optimizer.')

        # TODO(unno): Some optimizers can skip this process if they does not
        # affect to a parameter when its gradient is zero.
        for name, param in self.target.namedparams():
            if param.grad is None:
                with cuda.get_device(param.data):
                    xp = cuda.get_array_module(param.data)
                    param.grad = xp.zeros_like(param.data)

        self.call_hooks()
        self.prepare()

        self.t += 1
        states = self._states
        if self.init:
            for name, param in self.target.namedparams():
                with cuda.get_device(param.data) as dev:
                    #self.update_one(param, states[name])
                    state = states[name]
                    state['d'] = -param.grad.copy()
                    state['nabla'] = param.grad.copy()
                    if int(dev) == -1:
                        param.data += self.epsilon * state['d']
                    else:
                        cuda.elementwise(
                            'T eps, T d', 'T param',
                            'param += eps * d',
                            'setup_param'
                        )(self.epsilon, state['d'], param.data)
            self.init = False
        else:
            b_numerator = 0
            #b_denominator = 0
            for name, param in self.target.namedparams():
                with cuda.get_device(param.data) as dev:
                    state = states[name]
                    state['nabla'] = param.grad.copy()

                    #state['hd'] = (param.grad - state['nabla']) / self.epsilon
                    xp = cuda.get_array_module(param.data)
                    if int(dev) == -1:
                        b_numerator += numpy.sum(state['d'] * state['nabla'])
                    else:
                        # TODO: check behavior
                        #b_numerator += xp.sum(state['d'] * state['nabla'])
                        b_numerator += self.calc_inner_product_sum(state['d'], state['nabla'])

                        #cuda.reduce(
                        #    'T d, T nabla',
                        #    'T b_numerator',
                        #    'd * nabla',
                        #    'a + b',
                        #    'b_numerator += a',
                        #    '0',
                        #    'b_numerator_sum'
                        #)(state['d'], state['nabla'], b_numerator)
                    #b_numerator += numpy.sum(state['d'] * state['nabla'])
                    #b_denominator += sum(state['d'] * state['hd'])
            b = b_numerator/(self.ab_denominator+1e-3)
            for name, param in self.target.namedparams():
                with cuda.get_device(param.data) as dev:
                    state = states[name]
                    if int(dev) == -1:
                        state['d'] = -param.grad + b * state['d']
                        param.data += self.epsilon * state['d']
                    else:
                        cuda.elementwise(
                            'T eps, T g, T b', 'T d, T param',
                            '''
                            d = -g + b * d;
                            param += eps * d;
                            ''',
                            'update_param_first'
                        )(self.epsilon, param.grad, b, state['d'], param.data)

        # Second forward-backward
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', False)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            loss.backward()
            del loss
        else:
            print('Error lossfun must be set in argument for update in this optimizer.')

        # TODO(unno): Some optimizers can skip this process if they does not
        # affect to a parameter when its gradient is zero.
        for name, param in self.target.namedparams():
            if param.grad is None:
                with cuda.get_device(param.data):
                    xp = cuda.get_array_module(param.data)
                    param.grad = xp.zeros_like(param.data)

        self.prepare()
        a_numerator = 0
        self.ab_denominator = 0

        for name, param in self.target.namedparams():
            with cuda.get_device(param.data) as dev:

                state = states[name]
                xp = cuda.get_array_module(param.data)
                #print('diff', numpy.sum(param.grad - state['nabla']),
                #      '\n', param.grad - state['nabla'])
                if int(dev) == -1:
                    state['hd'] = (param.grad - state['nabla']) / self.epsilon
                    a_numerator += xp.sum(state['d'] * state['nabla'])
                    #a_numerator += numpy.sum(state['d'] * state['nabla'])
                    #print('state d', state['d'], 'state hd', state['hd'])
                    self.ab_denominator += xp.sum(state['d'] * state['hd'])
                    #self.ab_denominator += numpy.sum(state['d'] * state['hd'])
                else:
                    state['hd'] = cuda.elementwise(
                        'T g, T nabla, T eps',
                        'T hd',
                        'hd = (g - nabla) / eps',
                        'calc_hd'
                    )(param.grad, state['nabla'], self.epsilon)
                    a_numerator += self.calc_inner_product_sum(state['d'], state['nabla'])
                    self.ab_denominator += self.calc_inner_product_sum(state['d'], state['hd'])
                    #self.ab_denominator += numpy.sum(state['d'] * state['hd'])

        if self.ab_denominator == 0:
            print('WARNING numerator', a_numerator, 'denominator', self.ab_denominator)
        a = -a_numerator/(self.ab_denominator + 1e-3)
        for name, param in self.target.namedparams():
            with cuda.get_device(param.data) as dev:
                state = states[name]
                if int(dev) == -1:
                    param.data += (a - self.epsilon) * state['d']
                else:
                    cuda.elementwise(
                        'T a, T d', 'T param',
                        'param += a * d',
                        'update_param_second'
                    )(a - self.epsilon, state['d'], param.data)
#    def update_one(self, param, state):
#        """Updates a parameter based on the corresponding gradient and state.
#
#        This method calls appropriate one from :meth:`update_param_cpu` or
#        :meth:`update_param_gpu`.
#
#        Args:
#            param (~chainer.Variable): Parameter variable.
#            state (dict): State dictionary.
#
#        """
#        if isinstance(param.data, numpy.ndarray):
#            self.update_one_cpu(param, state)
#        else:
#            self.update_one_gpu(param, state)
#
#    def update_one_cpu(self, param, state):
#        """Updates a parameter on CPU.
#
#        Args:
#            param (~chainer.Variable): Parameter variable.
#            state (dict): State dictionary.
#
#        """
#        raise NotImplementedError
#
#    def update_one_gpu(self, param, state):
#        """Updates a parameter on GPU.
#
#        Args:
#            param (~chainer.Variable): Parameter variable.
#            state (dict): State dictionary.
#
#        """
#        raise NotImplementedError

    def use_cleargrads(self, use=True):
        """Enables or disables use of :func:`~chainer.Link.cleargrads` in `update`.

        Args:
            use (bool): If ``True``, this function enables use of
                `cleargrads`. If ``False``, disables use of `cleargrads`
                (`zerograds` is used).

        .. note::
           Note that :meth:`update` calls :meth:`~Link.zerograds` by default
           for backward compatibility. It is recommended to call this method
           before first call of `update` because `cleargrads` is more
           efficient than `zerograds`.

        """
        self._use_cleargrads = use