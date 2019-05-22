import numpy as np
from scipy import optimize


###################################################
#           Functions for problem 1.1             #                 
###################################################

def consumption(w, h, l, b):
    """ The consumption function dependent on work or not.

    Args:
        w (float): The wage rate
        h (float): Human capital
        l (Boolean): Work dummy
        b (float): Unemployment benefit

    Returns:
        c (float): Consumption
    """

    if l==1:
        c = w*h
    if l==0:
        c = b

    return c

def utility(c,rho):
    """ The consumption of utility.

    Args:
        c (float): consumption
        rho (float): Risk aversion
    
    Returns:
        Utility (float)

    """
    return c**(1-rho)/(1-rho)


def disutility(gamma, l):
    """ The disutility of working
    Args:
        gamma (float): Disutility of working
        l (Boolean): Working dummy

    Returns: 
        The disutility of work
    """
    return gamma*l


def v2(w,h2,l2,b,rho,gamma):
    """ The utility function to be maximized in period 2

    Args: 
        w (float): The wage rate
        h2 (float): Human capital in the second period
        l2 (Boolean): Work dummy in the second period
        b (float): Unemployment benefit
        rho (float): Risk aversion
        gamma (float): Disutility of working

    Returns:
        v2

    """
    c2 = consumption(w,h2,l2,b)
    
    return utility(c2, rho) - disutility(gamma, l2)

def solve_period_2(w,rho,b,gamma):

    # a. grids
    h2_vec = np.linspace(0.1,1.5,100) 
    v2_vec = np.empty(100)
    l2_vec = np.empty(100)

    # b. solve for each h2 in grid
    for i,h2 in enumerate(h2_vec):

        # i. objective
        obj = lambda l2: -v2(w,h2,l2,b,rho,gamma)

        # ii. initial value (half)
        x0 = 1

        # iii. constraint
        const = lambda l2: 
        
        # iv. optimizer
        result = optimize.minimize_scalar(obj,x0)

        # v. save
        v2_vec[i] = -result.fun
        l2_vec[i] = result.x
        
    return h2_vec,v2_vec,l2_vec


###################################################
#           Functions for problem 1.2             #                 
###################################################

def v2_interp(h2_vec, v2_vec)
    """ The interpolator of the v2

    Args:
        h2_vec
        v2_vec

    Returns:

    """
return interpolate.RegularGridInterpolator([h2_vec], v2_vec,bounds_error=False,fill_value=None)

def v1(w,h1,l1,b,rho,gamma, beta, Delta, v2_interp):
    """ The utility function to be maximized in period 2

    Args: 
        w (float): The wage rate
        h1 (float): Human capital in the first period
        l1 (Boolean): Work dummy in the first period
        b (float): Unemployment benefit
        rho (float): Risk aversion
        gamma (float): Disutility of working
        Delta (float): Stochastic experience gain
        beta (float): Discount factor of human capital
        v2_interp 

    Returns:
        v1
    """

    # i. v2 value if low human capital
    h2_low = h1 + l1
    v2_low = v2_interp([h2_low])[0]
    
    # ii. v2 value if high human capital
    h2_high = h1 + l1 + Delta
    v2_high = v2_interp([h2_high])[0]
    
    # iii. expected v2 value
    v2 = 0.5*v2_low + 0.5*v2_high
    
    # iv. consumption in period 1
    c1 = consumption(w,h1,l1,b)
    
    # v. total value
    return utility(c1, rho) - disutility(gamma, l1) + beta*v2


def solve_period_1(w,rho,b,gamma,beta,Delta,v2_interp):
     """ The utility function to be maximized in period 2

    Args: 
        w (float): The wage rate
        h2 (float): Human capital in the second period
        l2 (Boolean): Work dummy in the second period
        b (float): Unemployment benefit
        rho (float): Risk aversion
        gamma (float): Disutility of working
        Delta (float): Stochastic experience gain
        beta (float): Discount factor of human capital
        v2_interp 

    Returns:
        

    """

    # a. grids
    h1_vec = np.linspace(0.1,1.5,100) 
    v1_vec = np.empty(100)
    l1_vec = np.empty(100)

    # b. solve for each h2 in grid
    for i,h1 in enumerate(h1_vec):

        # i. objective
        obj = lambda l1: -v1(w,h1,l1,b,rho,gamma,beta,Delta,v2_interp)

        # ii. initial value (half)
        x0 = 1

        # iii. constraint
        const = lambda l1: 
        
        # iv. optimizer
        result = optimize.minimize_scalar(obj,x0)

        # v. save
        v1_vec[i] = -result.fun
        l1_vec[i] = result.x
        
    return h1_vec,v1_vec,l1_vec