import numpy as np
from scipy import optimize
from scipy import interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sympy as sm

import ipywidgets as widgets

###################################################
#           Functions for problem 1.1             #                 
###################################################

def consumption(w,h,l,b):
    """ The consumption function dependent on to work or not.
    Args:
        w (float): The wage rate
        h (float): Human capital
        l (Int): Work dummy
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

def disutility(gamma,l):
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
        l2 (Int): Work dummy in the second period
        b (float): Unemployment benefit
        rho (float): Risk aversion
        gamma (float): Disutility of working
    Returns:
        Value of the utility function
    """
    # Consumption function of the variables given:
    c2 = consumption(w,h2,l2,b)

    return utility(c2, rho) - disutility(gamma, l2)

def solve_period_2(w,rho,b,gamma,h_vec):
    """ The maximazation of utility and the choice of work at different levels of human capital
    Args:
        w (float): The wage rate
        b (float): Unemployment benefit
        rho (float): Risk aversion
        gamma (float): Disutility of working
        h_vec (arrey): Interval for human capital examined
    Returns:
        v2_vec (arrey): Corresponding utility in period 2 for human capital
        l2_vec (arrey): Corresponding choice of work given accumulated human capital
    """

    # a. grids
    v2_vec = np.empty(100)
    l2_vec = np.empty(100)

    # b. solve for each h2 in grid
    for i,h2 in enumerate(h_vec):
        if v2(w,h2,1,b,rho,gamma) > v2(w,h2,0,b,rho,gamma):
            l2_vec[i] = 1
        else:
            l2_vec[i] = 0

        v2_vec[i] = v2(w,h2,l2_vec[i],b,rho,gamma)
    
    # c. Plot
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(h_vec,l2_vec, color='red')
    
    ax.set_xlabel('$h_2$')
    ax.set_ylabel('$l_2$')
    ax.set_title('To work or not depending on human capital, period 2')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax = fig.add_subplot(1,2,2)
    ax.plot(h_vec,v2_vec, color='darkorange')

    ax.set_xlabel('$h_2$')
    ax.set_ylabel('$v_2$')
    ax.set_title('Value function dependent on human capital, period 2')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    return v2_vec,l2_vec

###################################################
#           Functions for problem 1.2             #                 
###################################################

def v2_interp(h_vec,v2_vec):
    """ The interpolator of the v2
    Args:
        h_vec (arrey): Vector of values for human capital
        v2_vec (arrey): Vector of values for corresponding utility values
    Returns:
        v2_interp (scipy interpolator): The interpolator for the expected value of utility in period 2.
    """
    return interpolate.RegularGridInterpolator([h_vec], v2_vec,bounds_error=False,fill_value=None)

def v1(w,h1,l1,b,rho,gamma,beta,Delta,v2_interp):
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
        v2_interp (scipy interpolator): Expected utility of period 2
    Returns:
        v1 (float): Value of utility in period 1
    """

    # a. v2 value if low human capital
    h2_low = h1 + l1
    v2_low = v2_interp([h2_low])[0]
    
    # b. v2 value if high human capital
    h2_high = h1 + l1 + Delta
    v2_high = v2_interp([h2_high])[0]
    
    # c. expected v2 value
    v2 = 0.5*v2_low + 0.5*v2_high
    
    # d. consumption in period 1
    c1 = consumption(w,h1,l1,b)
    
    # e. total value
    return utility(c1, rho) - disutility(gamma, l1) + beta*v2

def solve_period_1(w,b,rho,gamma,beta,Delta,v2_interp,h_vec):
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
        v1_vec (arrey): Vector of values for v1 for different values of human capital
        l1_vec (arrey): Vector of intergers for work (l=1) or not (l=0)
        Fig (plot): 2 plots illustrating v1 and l1 as a function of human capital
    """
    
    # a. grids
    v1_vec = np.empty(100)
    l1_vec = np.empty(100)

    # b. solve for each h2 in grid
    for i,h1 in enumerate(h_vec):
        if v1(w,h1,1,b,rho,gamma,beta,Delta,v2_interp) > v1(w,h1,0,b,rho,gamma,beta,Delta,v2_interp):
            l1_vec[i] = 1
        else:
            l1_vec[i] = 0

        v1_vec[i] = v1(w,h1,l1_vec[i],b,rho,gamma,beta,Delta,v2_interp)

    # c. illustration
    fig = plt.figure(figsize=(12,4))
    
    ax = fig.add_subplot(1,2,1)
    ax.plot(h_vec,l1_vec, color='red')
    
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$l_1$')
    ax.set_title('To work or not in period 1')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax = fig.add_subplot(1,2,2)
    ax.plot(h_vec,v1_vec, color='darkorange')
    
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$v_1$')
    ax.set_title('Value function in period 1')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()

    return v1_vec,l1_vec

######################################################################
##                       Assignment 2                               ##
######################################################################

####################################################
#           Functions for problem 2.2            #                 
###################################################

def fig_equilibrium(alpha,h,b,phi,gamma,pi_pt,y_pt,s_pt,v_t,s_t,v_shock):
    """ A figure illustrating the AD-curves and SRAS-curve before and after a shock to demand.
    Args:
        alpha (float): Marginal production of the interest rate 
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        phi (float): Marginal expected inflation of past inflation
        gamma (float): Marginal inflation of the output gap
        pi_pt (float): Inflation in period (t-1)
        y_pt (float): Output gap in period (t-1)
        s_pt (float): Supply disturbance in period (t-1)
        v_t (float): Demand disturbance in period t
        s_t (float): Supply disturbance in period t
        v_shock (float): Demand shock in period t
    Returns: 
        fig (plot): Plot of the ad-curves and sras-curve for different values of y.
    """
    # Value arreys are generated:
    y_arrey = np.linspace(-0.01,0.03)
    ad_pi_arrey = 1/(h*alpha)*(v_t-(1+alpha*b)*y_arrey)
    ad_pi_shock_arrey = 1/(h*alpha)*(v_shock-(1+alpha*b)*y_arrey)
    sras_pi_arrey = pi_pt + gamma*y_arrey - phi*gamma*y_pt + s_t - phi*s_pt

    # The figure is drawn
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(y_arrey, ad_pi_arrey, label="AD-curve", color='red',linewidth=2)
    ax.plot(y_arrey, ad_pi_shock_arrey, label="AD'-curve", color='darkorange',linewidth=2)
    ax.plot(y_arrey, sras_pi_arrey, label="SRAS-curve", color='lightblue',linewidth=4)
    
    ax.yaxis.grid(True, which='major')
 
    ax.set_xlabel('$y_t$')
    ax.set_ylabel('$\pi_t$')

    ax.legend(loc="upper right")
    ax.set_title('Figure 2.1: AS-ARAS with demand disturbance')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig

###################################################
#           Functions for problem 2.3            #                 
###################################################

def persistent_disturbance(T,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                y_neg1,pi_neg1,s_neg1,v_neg1,x0):
    """ Draws a figure displaying the persistence of a shock 
    Args:
        T (integer): Number of periods
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        phi (float): Marginal expected inflation of past inflation
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
        x0 (float): Demand shock in period 0    
    Returns: 
        Fig (plot): A plot of inflation and a plot of output over time.
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

    # We loop through to generate the values for the arreys for each period:
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
        
    # The figures are drawn
    fig = plt.figure(figsize = (12,8))
    
    # The figure displaying y*
    ax = fig.add_subplot(2,1,1)
    ax.plot(T_arrey, y_arrey, label="$y^*$-curve", color='red')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y*$')
    ax.set_title('The value of $y^*$ over time')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # The figure displaying pi*
    ax = fig.add_subplot(2,1,2)
    ax.plot(T_arrey, pi_arrey, label="$\pi^*$-curve",color='darkorange')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\pi^*$')
    ax.set_title('The value of $\pi^*$ over time')
    
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # To stop overlapping between subplots
    plt.tight_layout()

    return

###################################################
#           Functions for problem 2.4            #                 
###################################################

def stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,sigma_x,sigma_c,
                                y_neg1,pi_neg1,s_neg1,v_neg1):
    """ The equilibrium values of y and pi over time in an model with stochastic shocks.
    Args:
        T (integer): Number of periods
        seed (integer): Seed number
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        phi (float): Marginal expected inflation of past inflation
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        sigma_x (float): Standard deviation of demand shock
        sigma_c (float): Standard deviation of supply shock
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
    Returns: 
        arreys for y, pi and T
    """

    # a. The initial values:
    y_arrey  = [y_neg1]
    pi_arrey = [pi_neg1]
    s_arrey  = [s_neg1]
    v_arrey  = [v_neg1]
   
    # b. Simulation of shocks
    np.random.seed(seed)

    x_arrey = sigma_x*np.random.normal(size=T)
    c_arrey = sigma_c*np.random.normal(size=T)
    
    T_arrey = [0]

    # c. Loop through genereating the arreys:
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
    """ Illustrates the values of y and pi over time.
    Args:
        stochastic_shocks (Multiple arreys): y_arrey, pi_arrey, t_arrey
    Returns:
        Figure (plot): Plot of output and inflation over T periods.
    """
    T_arrey = stochastic_shocks[2]
    y_arrey = stochastic_shocks[0]
    pi_arrey = stochastic_shocks[1]
    
    # a. The figure is drawn
    fig = plt.figure(figsize = (12,8))
    
    # b. The figure showing y
    ax = fig.add_subplot(2,1,1)
    ax.plot(T_arrey, y_arrey, label="$y^*$-curve",color='red')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$y*$')
    ax.set_title('The value of $y^*$ over time')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # c. The figure showing pi
    ax = fig.add_subplot(2,1,2)
    ax.plot(T_arrey, pi_arrey, label="$\pi^*$-curve",color='darkorange')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\pi^*$')
    ax.set_title('The value of $\pi^*$ over time')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # To stop overlapping between subplots
    plt.tight_layout()
    return

###################################################
#           Functions for problem 2.5            #                 
###################################################

def plot_corr_phi(T,seed,sol_func_y,sol_func_pi,alpha,h,b,gamma,delta,omega,sigma_x,sigma_c,
                                y_neg1,pi_neg1,s_neg1,v_neg1):
    """ A plot of phi and the correlation between y and pi
    Args:
        T (integer): Number of periods
        seed (integer): Seed number
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        sigma_x (float): Standard deviation of demand shock
        sigma_c (float): Standard deviation of supply shock
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
    Returns:
        Fig (plot): A plot of values of the correlation between y and pi for different values of phi.
    """
    # a. The arreys are initilized:
    phi_arrey = np.linspace(0,1)
    corr_arrey = [] # Empty
    
    # b. Loop through the phi_arrey to get the corresponding value of the correlation
    for phi in phi_arrey:
        
        simul = stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)
        
        y_arrey = simul[0]
        pi_arrey = simul[1]
        
        correlation = np.corrcoef(y_arrey,pi_arrey)[1][0]
        corr_arrey.append(correlation)

    # c. The figure is drawn
    fig, ax = plt.subplots(figsize = (10,6))

    ax.plot(phi_arrey,corr_arrey,color='lightblue',linewidth=4)

    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$corr(y_t, \pi_t)$')
    ax.set_title('The value of $corr(y_t, \pi_t)$ as a function of $\phi$')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
 
    return

def correlations(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1):
    """Correlation between y and pi
    Args:
        T (integer): Number of periods
        seed (integer): Seed number
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        phi (float): Marginal expected inflation of past inflation
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        sigma_x (float): Standard deviation of demand shock
        sigma_c (float): Standard deviation of supply shock
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
    Returns:
        corr (float): The correlation between y and pi.
    """
    # The simulation data
    simul = stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)

    # The y and pi arreys    
    y_arrey = simul[0]
    pi_arrey = simul[1]
        
    return np.corrcoef(y_arrey,pi_arrey)[1][0]

def optimize_phi(corr_goal,T,seed,sol_func_y,sol_func_pi,alpha,h,b,gamma,delta,omega,sigma_x,sigma_c,
                        y_neg1,pi_neg1,s_neg1,v_neg1):
    """Optimization of phi so correlation responds to the correlation goal:
    Args:
        corr_goal (float): Correlation which is wished optained
        T (integer): Number of periods
        seed (integer): Seed number
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        sigma_x (float): Standard deviation of demand shock
        sigma_c (float): Standard deviation of supply shock
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
    Returns:
        Optimize results (Scipy optimize result): Characteristics and results of the optimization process.
    """
    # Our objective function
    obj = lambda phi_obj: (correlations(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi_obj,gamma,delta,
                                            omega,sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1) 
                                                - corr_goal)**2
    # Initial guess
    x0 = 0
    
    return optimize.minimize_scalar(obj,x0,method='bounded',bounds=[0,1])

###################################################
#           Functions for problem 2.6            #                 
###################################################

def statistics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                    sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1):
    """Statistics is calculated for y and pi
    Args:
        T (integer): Number of periods
        seed (integer): Seed number
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        phi (float): Marginal expected inflation of past inflation
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        sigma_x (float): Standard deviation of demand shock
        sigma_c (float): Standard deviation of supply shock
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
    Returns:
        var_y,var_pi,corr,autocorr_y,autocorr_pi (float): Statistics
    """
    # The simulated data to be examined
    simul = stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)
        
    # The arreys
    y = simul[0]
    pi = simul[1]

    # The statistics is calculated
    var_y  = np.var(y)
    var_pi = np.var(pi)
    corr   = np.corrcoef(y,pi)[1][0]
    autocorr_y  = np.corrcoef(y[1:],y[:-1])[1][0]
    autocorr_pi = np.corrcoef(pi[1:],pi[:-1])[1][0]

    return var_y,var_pi,corr,autocorr_y,autocorr_pi

def optimize_all_char(T,seed,sol_func_y,sol_func_pi,alpha,h,b,gamma,delta,omega,
                                y_neg1,pi_neg1,s_neg1,v_neg1,
                                var_y,var_pi,corr_y_pi,autocorr_y,autocorr_pi):
    """ Optimizes all statistics to correspond to the values set by: 
        var_y,var_pi,corr_y_pi,autocorr_y,autocorr_pi
    Args:
        T (integer): Number of periods
        seed (integer): Seed number
        sol_func_y (python function): Value in y in equilibrium as a function of parameters
        sol_func_pi (python function): Value in pi in equilibrium as a function of parameters
        alpha (float): Marginal production of the interest rate
        h (float): Inflation aversion parameter
        b (float): Outputgap aversion parameter
        phi (float): Marginal expected inflation of past inflation
        gamma (float): Marginal inflation of the output gap
        delta (float): Demand shock parameter
        omega (float): Supply shock parameter
        sigma_x (float): Standard deviation of demand shock
        sigma_c (float): Standard deviation of supply shock
        y_neg1 (float): Output gap in period (-1)
        pi_neg1 (float): Inflation in period (-1)
        s_neg1 (float): Supply disturbance in period (-1)
        v_neg1 (float): Demand disturbance in period (-1)
        var_y (float): The variation in y
        var_pi (float): The variation in pi
        corr_y_pi (float): The correlation between y and pi
        autocorr_y (float): The autocorrelation in y
        autocorr_pi (float): The autocorrelation in pi
    Returns:
        Optimization values(Scipy.optimizeresult): Characteristics of the optimization process
    """
    # a. A function of phi sigma_x, sigma_c is defined
    def funct(phi,sigma_x,sigma_c):
        return   (
                (statistics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[0] - var_y)**2
              + (statistics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[1] - var_pi)**2
              + (statistics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[2] - corr_y_pi)**2
              + (statistics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[3] - autocorr_y)**2
              + (statistics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[4] - autocorr_pi)**2
                  )
    
    # b. A function of the prior function collecting the parameters to be optemized in a 
    #     combined variable.
    def f(par_size):
        phi,sigma_x,sigma_c = par_size
        return funct(phi,sigma_x,sigma_c)

    # c. Initial guess and bounds for phi, sigma_c, sigma_x
    x0 = 0.5,1,1
    bnds = ((0,1), (1e-8,None), (1e-8,None))

    return optimize.minimize(f,x0,bounds=bnds)

######################################################################
##                       Assignment 3                               ##
######################################################################

##################################################
#           Functions for problem 3.2            #                 
##################################################

def demand_data(b1,b2,b3,e1,e2,e3):
    """ Simulates the excess demand
    Args:
        b1 (arrey): Budget shares for consumers good 1
        b2 (arrey): Budget shares for consumers good 2
        b3 (arrey): Budget shares for consumers good 3
        e1 (arrey): Endowments for good 1
        e2 (arrey): Endowments for good 2
        e3 (arrey): Endowments for good 3
    Returns:
        grids for p1, p2, e1, e2, e3 (arreys)
    """
    # A set of price vectors are defined
    p1_vec = np.linspace(0.1,5,100)
    p2_vec = np.linspace(0.1,5,100)
    p3_vec = 1

    # Now grids for the endowments and prices are constructed
    e1_grid = np.empty((100,100))
    e2_grid = np.empty((100,100))
    e3_grid = np.empty((100,100))
    p1_grid, p2_grid = np.meshgrid(p1_vec,p2_vec)

    # Now we can find the excess demands with a loop
    for i,p1 in enumerate(p1_vec):
        for j,p2 in enumerate(p2_vec):
            e1_grid[i,j] = np.sum(b1*((p1*e1 + p2*e2 + e3)/p1) - e1)
            e2_grid[i,j] = np.sum(b2*((p1*e1 + p2*e2 + e3)/p2) - e2)
            e3_grid[i,j] = np.sum(b3*(p1*e1 + p2*e2 + e3) - e3)

    return p1_grid,p2_grid,e1_grid,e2_grid,e3_grid

def demand_plots_3D(p1_grid,p2_grid,e1_grid,e2_grid,e3_grid):  
    """ Plots 3D excess demand plots for good 1 and 2
    Args:
        p1_grid (ndarrey): grid for p1
        p2_grid (ndarrey): grid for p2
        e1_grid (ndarrey): grid for e1
        e2_grid (ndarrey): grid for e2
        e3_grid (ndarrey): grid for e3
    Returns:
        Plot of excess demand for good 1 and good 2.
    """
    # Collective figure
    fig = plt.figure(figsize=(15,10))

    # Subplot good 1, axes crossing in (0,0)
    ax1 = fig.add_subplot(2,2,1,projection='3d')
    fig1 = ax1.plot_surface(p1_grid, p2_grid, e1_grid, color='red')
    ax1.set_xlabel('$p_1$')
    ax1.set_ylabel('$p_2$')
    ax1.invert_xaxis()
    ax1.set_title('Excess demand of $x_1$')

    # Subplot good 1, axes crossing in (5,5)
    ax1 = fig.add_subplot(2,2,2,projection='3d')
    fig1 = ax1.plot_surface(p1_grid, p2_grid, e1_grid, color='red')
    ax1.set_xlabel('$p_1$')
    ax1.set_ylabel('$p_2$')
    ax1.invert_yaxis()
    ax1.set_title('Excess demand of $x_1$')

    # Subplot good 2, axes crossing in (0,0)
    ax2 = fig.add_subplot(2,2,3,projection='3d')
    fig2 = ax2.plot_surface(p1_grid, p2_grid, e2_grid, color='darkorange')
    ax2.set_xlabel('$p_1$')
    ax2.set_ylabel('$p_2$')
    ax2.invert_xaxis()
    ax2.set_title('Excess demand of $x_2$')

    # Subplot good 2, axes crossing in (5,5)
    ax2 = fig.add_subplot(2,2,4,projection='3d')
    fig2 = ax2.plot_surface(p1_grid, p2_grid, e2_grid, color='darkorange')
    ax2.set_xlabel('$p_1$')
    ax2.set_ylabel('$p_2$')
    ax2.invert_yaxis()
    ax2.set_title('Excess demand of $x_2$')

    plt.tight_layout()
    return

def demand_plot_x3(p1_grid,p2_grid,e1_grid,e2_grid,e3_grid):
    """ Plots 3D excess demand plots for good 3
    Args:
        p1_grid (ndarrey): grid for p1
        p2_grid (ndarrey): grid for p2
        e1_grid (ndarrey): grid for e1
        e2_grid (ndarrey): grid for e2
        e3_grid (ndarrey): grid for e3
    Returns:
        Plot of excess demand for good 3
    """
    # Figure for excess demand, good 3:
    fig3 = plt.figure(figsize=(15,5))
    ax3 = fig3.add_subplot(1,1,1,projection='3d')
    fig3 = ax3.plot_surface(p1_grid, p2_grid, e3_grid, color='lightblue')
    ax3.set_xlabel('$p_1$')
    ax3.set_ylabel('$p_2$')
    ax3.invert_xaxis()
    ax3.set_title('Excess demand of $x_3$')
    return

##################################################
#           Functions for problem 3.3            #                 
##################################################

def find_equilibrium(b1,b2,p1,p2,e1,e2,e3,eps,kappa,N,maxiter=25000):
    """ Finds the Walras equilibrium by a Tatonnement process:
    Args:
        b1 (arrey): Budget shares for consumers good 1
        b2 (arrey): Budget shares for consumers good 2
        b3 (arrey): Budget shares for consumers good 3
        p1 (arrey): Prices for good 1
        p2 (arrey): Prices for good 2
        e1 (arrey): Endowments for good 1
        e2 (arrey): Endowments for good 2
        e3 (arrey): Endowments for good 3
        kappa (float): Parameter
        N (integer): Number of consumers
        maxiter (integer): Maximum number of iterations.
    Returns:
        Walras equilibrium for (p1,p2)
    """
    t = 0
    while True:
        # a. step 2: excess demand
        z1 = np.sum(b1*(p1*e1 + p2*e2 + e3)/p1 - e1)
        z2 = np.sum(b2*(p1*e1 + p2*e2 + e3)/p2 - e2)
            
        # b: step 3: stop?
        if np.abs(z1) < eps and np.abs(z2) < eps or t >= maxiter:
            print(f'{t:3d}: (p1,p2) = ({p1:.2f},{p2:.2f}) -> ({z1:.2f},{z2:.2f})')
            break
             
        # c. step 4: update p1 and p2
        p1 = p1 + kappa*z1/N
        p2 = p2 + kappa*z2/N
                
        # d. Print:
        if t < 5 or t%5000 == 0:
            print(f'{t:3d}: (p1,p2) = ({p1:.2f},{p2:.2f}) -> ({z1:.2f},{z2:.2f})')
        elif t == 5:
            print('   ...')
                
        t += 1    
    return [p1,p2]

##################################################
#           Functions for problem 3.4            #                 
##################################################

def utility_walras(p1,p2,e1,e2,e3,b1,b2,b3,gamma):
    """ The utility function
    Args:    
        b1 (float): Budget shares for consumers good 1
        b2 (float): Budget shares for consumers good 2
        b3 (float): Budget shares for consumers good 3
        p1 (float): Prices for good 1
        p2 (float): Prices for good 2
        e1 (float): Endowments for good 1
        e2 (float): Endowments for good 2
        e3 (float): Endowments for good 3
        gamma (float): Parameter
    Returns:
        Utility
    """
    # The income function
    I = p1*e1 + p2*e2 + e3
    # The goods
    x1 = b1*(I/p1)
    x2 = b2*(I/p2)
    x3 = b3*I
    # The utility
    utility = ((x1**b1)*(x2**b2)*(x3**b3))**gamma

    return utility
    
def utility_hist(p1,p2,e1,e2,e3,b1,b2,b3,gamma):
    """ Density plot of the utility function
    Args:    
        b1 (float): Budget shares for consumers good 1
        b2 (float): Budget shares for consumers good 2
        b3 (float): Budget shares for consumers good 3
        p1 (float): Prices for good 1
        p2 (float): Prices for good 2
        e1 (float): Endowments for good 1
        e2 (float): Endowments for good 2
        e3 (float): Endowments for good 3
        gamma (float): Parameter
    Returns:
        Utility density plot
    """
    
    utility = utility_walras(p1,p2,e1,e2,e3,b1,b2,b3,gamma)
    mean = utility.mean()
    
    # The figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)
    ax.hist(utility,bins=500,color='lightblue')
    plt.axvline(mean, color='red', linestyle='dashed')
   
    ax.set_xlabel('Utility')
    ax.set_ylabel('# consumers')

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return