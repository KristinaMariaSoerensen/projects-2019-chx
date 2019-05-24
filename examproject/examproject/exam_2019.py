import numpy as np
from scipy import optimize
from scipy import interpolate

import matplotlib.pyplot as plt
import sympy as sm

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

        # iii. l constraint
        # iv. optimizer
        result = optimize.minimize_scalar(obj,x0)

        # v. save
        v2_vec[i] = -result.fun
        l2_vec[i] = result.x
        
    return h2_vec,v2_vec,l2_vec


###################################################
#           Functions for problem 1.2             #                 
###################################################

def v2_interp(h2_vec, v2_vec):
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
        Something
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

        # iii. l constraint
         
        
        # iv. optimizer
        result = optimize.minimize_scalar(obj,x0)

        # v. save
        v1_vec[i] = -result.fun
        l1_vec[i] = result.x
        
    return h1_vec,v1_vec,l1_vec


####################################################################
####################################################################
#                       Assignment 2                               #
#                                                                  #
####################################################################
####################################################################
####################################################
#           Functions for problem 2.2            #                 
###################################################

def fig_equilibrium(alpha,h,b,phi,gamma,pi_pt,y_pt,s_pt,v_t,s_t,v_shock):
    """ 


    Args:
        alpha (float):
        h (float):
        b (float):
        phi (float):
        gamma (float):
        pi_pt (float):
        y_pt (float):
        s_pt (float):
        v_t (float):
        s_t (float):
        v_shock (float):

    Returns: 
        Result: Figure
    """
    # Value arreys are generated:
    y_arrey = np.linspace(-0.02,0.04)
    ad_pi_arrey = 1/(h*alpha)*(v_t-(1+alpha*b)*y_arrey)
    ad_pi_shock_arrey = 1/(h*alpha)*(v_shock-(1+alpha*b)*y_arrey)
    sras_pi_arrey = pi_pt + gamma*y_arrey - phi*gamma*y_pt + s_t - phi*s_pt

    # The figure is drawn
    fig, ax = plt.subplots()

    ax.plot(y_arrey, ad_pi_arrey, label="AD-curve")
    ax.plot(y_arrey, ad_pi_shock_arrey, label="AD'-curve")
    ax.plot(y_arrey, sras_pi_arrey, label="SRAS-curve")

    ax.set_xlabel('$y_t$')
    ax.set_ylabel('$\pi_t$')

    ax.legend(loc="upper right")
    
    return fig

###################################################
#           Functions for problem 2.3            #                 
###################################################

def persistent_disturbance(T,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                y_neg1,pi_neg1,s_neg1,v_neg1,x0):
    """ Steady state for the Keynesian cross where aggregate expenditure equals total production
    
    Args:
        T (integer): Number of periods
        alpha (float):
        h (float):
        b (float):
        phi (float):
        gamma (float):
        pi_pt (float):
        y_pt (float):
        

    Returns: 
        Result: Figure
    """

    # The initial values:
    y_arrey  = [y_neg1]
    pi_arrey = [pi_neg1]
    s_arrey  = [s_neg1]
    v_arrey  = [v_neg1]
    c_arrey  = np.zeros(T)
    x_arrey  = np.zeros(T)
    
    # The first value of x_arrey is replaced with our shock value:
    x_arrey[1] = x0

    T_arrey = [0]

    # 
    for i in range(1,T):
        T_arrey.append(i)

        v_arrey.append(delta*v_arrey[i-1] + x_arrey[i])
        s_arrey.append(omega*s_arrey[i-1] + c_arrey[i])

        y_arrey.append(sol_func_y(alpha,h,b,phi,gamma, 
                                        pi_arrey[i-1], y_arrey[i-1], s_arrey[i-1], 
                                        v_arrey[i], s_arrey[i]))
        pi_arrey.append(sol_func_pi(alpha,h,b,phi,gamma, 
                                        pi_arrey[i-1], y_arrey[i-1], s_arrey[i-1], 
                                        v_arrey[i], s_arrey[i]))
        
    # The figure is drawn
    fig = plt.figure(figsize = (8,12))
    ax = fig.add_subplot(2,1,1)
    ax.plot(T_arrey, y_arrey, label="$y^*$-curve")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y*$')

    ax = fig.add_subplot(2,1,2)
    ax.plot(T_arrey, pi_arrey, label="$\pi^*$-curve")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\pi^*$')
    
    return

###################################################
#           Functions for problem 2.4            #                 
###################################################

def stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,sigma_x,sigma_c,
                                y_neg1,pi_neg1,s_neg1,v_neg1):
    """ 
    Args:
        T (integer): Number of periods
        alpha (float):
        h (float):
        b (float):
        phi (float):
        gamma (float):
        pi_pt (float):
        y_pt (float):
    Returns: 
        Figure
    """

    # The initial values:
    y_arrey  = [y_neg1]
    pi_arrey = [pi_neg1]
    s_arrey  = [s_neg1]
    v_arrey  = [v_neg1]
   
    # Simulation of shocks
    np.random.seed(seed) # set the seed

    x_arrey = sigma_x*np.random.normal(size=T)
    c_arrey = sigma_c*np.random.normal(size=T)
    
    T_arrey = [0]

    # Loop through genereating the arreys:
    for i in range(1,T):
        T_arrey.append(i)

        v_arrey.append(delta*v_arrey[i-1] + x_arrey[i])
        s_arrey.append(omega*s_arrey[i-1] + c_arrey[i])

        y_arrey.append(sol_func_y(alpha,h,b,phi,gamma, 
                                        pi_arrey[i-1], y_arrey[i-1], s_arrey[i-1], 
                                        v_arrey[i], s_arrey[i]))
        pi_arrey.append(sol_func_pi(alpha,h,b,phi,gamma, 
                                        pi_arrey[i-1], y_arrey[i-1], s_arrey[i-1], 
                                        v_arrey[i], s_arrey[i]))
        
    return y_arrey, pi_arrey, T_arrey

def fig_stochastic_shocks(stochastic_shocks):
    """
    """
    T_arrey = stochastic_shocks[2]
    y_arrey = stochastic_shocks[0]
    pi_arrey = stochastic_shocks[1]
    
    # The figure is drawn
    fig = plt.figure(figsize = (8,12))
    ax = fig.add_subplot(2,1,1)
    ax.plot(T_arrey, y_arrey, label="$y^*$-curve")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y*$')

    ax = fig.add_subplot(2,1,2)
    ax.plot(T_arrey, pi_arrey, label="$\pi^*$-curve")
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\pi^*$')

    return


###################################################
#           Functions for problem 2.5            #                 
###################################################

def corr_of_phi(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                    sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1):
    """
    """
    simul = stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)
        
    y = simul[0]
    pi = simul[1]
        
    return np.corrcoef(y,pi)[1][0]


def plot_corr_phi(T,seed,sol_func_y,sol_func_pi,alpha,h,b,gamma,delta,omega,sigma_x,sigma_c,
                                y_neg1,pi_neg1,s_neg1,v_neg1):
    """
    """
    phi_arrey = np.linspace(0,1)

    corr_arrey = []
    
    for phi in phi_arrey:
        
        simul = stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)
        
        y_arrey = simul[0]
        pi_arrey = simul[1]
        
        correlation = np.corrcoef(y_arrey,pi_arrey)[1][0]
        corr_arrey.append(correlation)

    # The figure is drawn
    fig, ax = plt.subplots()

    ax.plot(phi_arrey, corr_arrey)

    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$corr(y_t, \pi_t)$')
    
    return

def optimize_phi(corr_goal,T,seed,sol_func_y,sol_func_pi,alpha,h,b,gamma,delta,omega,sigma_x,sigma_c,
                        y_neg1,pi_neg1,s_neg1,v_neg1):
    """
    """

    obj = lambda phi_obj: (corr_of_phi(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi_obj,gamma,delta,
                                            omega,sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1) 
                                                - corr_goal)**2
    
    # Initial guess
    x0 = 0
    
    return optimize.minimize_scalar(obj,x0,method='bounded',bounds=[0,1])
