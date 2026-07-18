import jax.numpy as jnp
import jax.random as random

def get_ell_range():
    # # Predefined list of ell values
    #ell_eval = jnp.array([10., 13.5, 18., 23.5, 30.5, 40., 52.5, 68.5, 89.5, 
    #                     117., 152.5, 198., 257.5, 335.5, 436.5, 567.5, 738., 959.5])

    # ell_eval = jnp.array([
    #     10.0,   13.5,   18.0,   23.5,   30.5,   40.0,   52.5,   68.5,
    #     89.5,  117.0,  152.5,  198.0,  257.5,  335.5,  436.5,  567.5,
    #    738.0,  959.5, 1247.5, 1622.0, 2109.0, 2742.0
    # ])

    # ell_eval = jnp.array([117., 152.5, 198., 257.5, 335.5, 436.5, 567.5, 738., 959.5])
    # ell_eval = jnp.array([10., 13.5, 18., 23.5, 30.5, 40., 52.5, 68.5, 89.5])

    # ell_eval = jnp.array([436.5, 567.5, 738., 959.5])
                         
    # ell_eval = jnp.array([2.00000000e+00, 2.44280552e+00, 2.98364940e+00, 3.64423760e+00,
    #    4.45108186e+00, 5.43656366e+00, 6.64023385e+00, 8.11039993e+00,
    #    9.90606485e+00, 1.20992949e+01, 1.47781122e+01, 1.80500270e+01,
    #    2.20463528e+01, 2.69274761e+01, 3.28892935e+01, 4.01710738e+01,
    #    4.90650604e+01, 5.99282001e+01, 7.31964689e+01, 8.94023690e+01,
    #    1.09196300e+02, 1.33372662e+02, 1.62901737e+02, 1.98968631e+02,
    #    2.43020835e+02, 2.96826318e+02, 3.62544484e+02, 4.42812832e+02,
    #    5.40852815e+02, 6.60599120e+02, 8.06857587e+02, 9.85498082e+02,
    #    1.20369008e+03, 1.47019038e+03, 1.79569458e+03, 2.19326632e+03,
    #    2.67886153e+03, 3.27196886e+03, 3.99639179e+03, 4.88120396e+03,
    #    5.96191597e+03, 7.28190061e+03, 8.89413350e+03, 1.08633192e+04,
    #    1.32684880e+04, 1.62061679e+04, 1.97942581e+04])

    # ell_eval = jnp.array([ 201.33498645,  468.52584512,  758.25099747, 1056.84050866,
    #    1356.86516678, 1653.00918868, 1954.05909259, 2255.02772429,
    #    2552.01284927, 2852.1638449 , 3153.04128834, 3451.99885512,
    #    3752.40263676, 4052.1491112 , 4351.13321412, 4651.64008646,
    #    4952.0937097 , 5250.97380204, 5550.6707366 , 5851.40984543,
    #    6151.32033737, 6451.27061224, 6751.22105627, 7050.68944786,
    #    7350.8687472 , 7651.40650855, 7950.52738028, 8250.15085296,
    #    8551.05597974, 8850.9556685 , 9150.53637033, 9450.81915756,
    #    9750.78785945])
    # ell_eval = jnp.arange(200, 301)  # Creates an array from 10 to 1000 (inclusive)
    # print(ell_eval)
    ell_eval = jnp.geomspace(400, 5000, 18)

    # ell_eval = jnp.arange(0, 1200, 50)
    return ell_eval

def get_ell_binwidth():
    #delta_ell = jnp.array([3., 4., 5., 6., 8., 11., 14., 18., 24., 31., 40., 51., 68., 88., 114., 148.,
    #                       193., 250.])
    # delta_ell = jnp.array([
    #    3.,   4.,   5.,   6.,   8.,  11.,  14.,  18.,  24.,  31.,  40.,
    #   51.,  68.,  88., 114., 148., 193., 250., 325., 424., 550., 716.
    # ])
    
    # ell_eval = get_ell_range()
    # delta_ell = jnp.diff(ell_eval, prepend=ell_eval[0])
    # # Multiply the first and last gap by 2
    # delta_ell = delta_ell.at[0].set(2 * delta_ell[1])
    # delta_ell = delta_ell.at[-1].set(2 * delta_ell[-1])
    # print("delta_ell", delta_ell)
       
    # CENTERED ON THE ELLs :
    ell = get_ell_range()
    delta = jnp.empty_like(ell)
    # bins centrés
    delta = delta.at[1:-1].set(0.5 * (ell[2:] - ell[:-2]))
    delta = delta.at[0].set(ell[1] - ell[0])
    delta = delta.at[-1].set(ell[-1] - ell[-2])
    return delta
    
    #return delta_ell


def simpson(y, *, x=None, dx=1.0, axis=-1):
    y = jnp.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    if x is not None:
        x = jnp.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd
        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)
            h = jnp.asarray([dx, dx], dtype=jnp.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))
                diffs = jnp.float64(jnp.diff(x, axis=axis))
                h = [jnp.squeeze(diffs[hm2], axis=axis),
                     jnp.squeeze(diffs[hm1], axis=axis)]
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = jnp.true_divide(
                num,
                den,
            )
            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = jnp.true_divide(
                num,
                den,
            )
            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = jnp.true_divide(
                num,
                den,
            )
            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]
        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result
def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))
    if x is None:  # Even-spaced Simpson's rule.
        result = jnp.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = jnp.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float)
        h1 = h[sl1].astype(float)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = jnp.true_divide(h0, h1)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - jnp.true_divide(1.0, h0divh1)) +
                          y[slice1] * (hsum *
                                       jnp.true_divide(hsum, hprod)) +
                          y[slice2] * (2.0 - h0divh1))
        result = jnp.sum(tmp, axis=axis)
    return result
def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

def sample_from_uniform(key, x_min, x_max, n=1):
    # Generate a uniform random sample using the provided key.
    sample = random.uniform(key, shape=(n,), minval=x_min, maxval=x_max)
    # If only one sample is requested, return it as a scalar.
    if n == 1:
        sample = sample[0]
    return sample

# --- Helper Functions for Broadcasting ---

def ensure_array(arg):
    """Ensure that the argument is a JAX array."""
    if not isinstance(arg, jnp.ndarray):
        return jnp.array(arg)
    return arg

def broadcast_to_batch(arg, batch_size):
    """If arg is scalar (rank 0), broadcast it to shape (batch_size,)."""
    arg = ensure_array(arg)
    if arg.ndim == 0:
        return jnp.broadcast_to(arg, (batch_size,))
    return arg

def get_batch_size(*args):
    """
    Determine the batch size from the first argument that is batched.
    If none are batched, return None.
    """
    for arg in args:
        arr = ensure_array(arg)
        if arr.ndim > 0:
            return arr.shape[0]
    return None

