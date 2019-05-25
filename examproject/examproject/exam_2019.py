import numpy as np
from scipy import optimize
from scipy import interpolate

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sympy as sm

###################################################
#           Functions for problem 1.1             #                 
###################################################

def consumption(w,h,l,b):
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
        l2 (Boolean): Work dummy in the second period
        b (float): Unemployment benefit
        rho (float): Risk aversion
        gamma (float): Disutility of working

    Returns:
        v2

    """
    c2 = consumption(w,h2,l2,b)
    
    return utility(c2, rho) - disutility(gamma, l2)

def solve_period_2(w,rho,b,gamma,h_vec):

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
    
    # illustration
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1,2,1)
    ax.plot(h_vec,l2_vec)
    ax.grid()
    ax.set_xlabel('$h_2$')
    ax.set_ylabel('$l_2$')
    ax.set_title('Work function in period 2')

    ax = fig.add_subplot(1,2,2)
    ax.plot(h_vec,v2_vec)
    ax.grid()
    ax.set_xlabel('$h_2$')
    ax.set_ylabel('$v_2$')
    ax.set_title('Value function in period 2')

    return v2_vec,l2_vec

###################################################
#           Functions for problem 1.2             #                 
###################################################

def v2_interp(h_vec,v2_vec):
    """ The interpolator of the v2

    Args:
        h_vec
        v2_vec

    Returns:

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
        Something
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

    # illustration
    fig = plt.figure(figsize=(10,4))
    
    ax = fig.add_subplot(1,2,1)
    ax.plot(h_vec,l1_vec)
    ax.grid()
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$l_1$')
    ax.set_title('Work function in period 1')

    ax = fig.add_subplot(1,2,2)
    ax.plot(h_vec,v1_vec)
    ax.grid()
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$v_1$')
    ax.set_title('value function in period 1')
    
    return v1_vec,l1_vec

###################################################
#           Functions for problem 1.3             #                 
###################################################

def optimization_period_2():
    """
    """
    def diff_p2(w,h2,b,rho,gamma):
        return v2(w,h2,1,b,rho,gamma) - v2(w,h2,0,b,rho,gamma)

    def f(par_size):
        w,h2,b,rho,gamma = par_size
        return diff_p2(w,h2,b,rho,gamma)

    # Initial guess and bounds for w,h,b,rho,gamma
    x0 = 0.9,1,1,2,0.1
    bnds = ((0,b),(0.1,1.5),(w,None),(0,None),(0,None))

    return optimize.minimize(f,x0,bounds=bnds)
    
######################################################################
##                       Assignment 2                               ##
######################################################################

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

###################################################
#           Functions for problem 2.6            #                 
###################################################

def characteristics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                    sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1):
    """
    """
    simul = stochastic_shocks(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                        sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)
        
    y = simul[0]
    pi = simul[1]

    var_y  = np.var(y)
    var_pi = np.var(pi)
    corr   = np.corrcoef(y,pi)[1][0]

    autocorr_y  = np.corrcoef(y[1:],y[:-1])[1][0]
    autocorr_pi = np.corrcoef(pi[1:],pi[:-1])[1][0]

    return var_y,var_pi,corr,autocorr_y,autocorr_pi


def optimize_all_char(T,seed,sol_func_y,sol_func_pi,alpha,h,b,gamma,delta,omega,
                                y_neg1,pi_neg1,s_neg1,v_neg1,
                                var_y,var_pi,corr_y_pi,autocorr_y,autocorr_pi):
    """
    """
    def funct(phi,sigma_x,sigma_c):
        return   (
                (characteristics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[0] - var_y)**2
              + (characteristics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[1] - var_pi)**2
              + (characteristics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[2] - corr_y_pi)**2
              + (characteristics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[3] - autocorr_y)**2
              + (characteristics(T,seed,sol_func_y,sol_func_pi,alpha,h,b,phi,gamma,delta,omega,
                                sigma_x,sigma_c,y_neg1,pi_neg1,s_neg1,v_neg1)[4] - autocorr_pi)**2
                  )
    
    def f(par_size):
        phi,sigma_x,sigma_c = par_size
        return funct(phi,sigma_x,sigma_c)

    # Initial guess and bounds for phi, sigma_c, sigma_x
    x0 = 0.5,1,1
    bnds = ((0,1), (1e-8,None), (1e-8,None))

    return optimize.minimize(f,x0,bounds=bnds)

######################################################################
##                       Assignment 3                               ##
######################################################################

##################################################
#           Functions for problem 3.2            #                 
##################################################

def demand_plots_3D(b1,b2,b3,e1,e2,e3):
    """
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


    fig = plt.figure(figsize=(10,19))

    ax1 = fig.add_subplot(3,1,1,projection='3d')
    fig1 = ax1.plot_surface(p1_grid, p2_grid, e1_grid, cmap=cm.coolwarm)
    ax1.set_xlabel('$p_1$')
    ax1.set_ylabel('$p_2$')
    ax1.invert_xaxis()
    ax1.set_title('Excess demand of Good 1', fontweight='bold')
    fig.colorbar(fig1, shrink=0.5, aspect=5)

    ax2 = fig.add_subplot(3,1,2,projection='3d')
    fig2 = ax2.plot_surface(p1_grid, p2_grid, e2_grid, cmap=cm.coolwarm)
    ax2.set_xlabel('$p_1$')
    ax2.set_ylabel('$p_2$')
    ax2.invert_xaxis()
    ax2.set_title('Excess demand of Good 2', fontweight='bold')
    fig.colorbar(fig2, shrink=0.5, aspect=5)

    ax3 = fig.add_subplot(3,1,3,projection='3d')
    fig3 = ax3.plot_surface(p1_grid, p2_grid, e3_grid, cmap=cm.coolwarm)
    ax3.set_xlabel('$p_1$')
    ax3.set_ylabel('$p_2$')
    ax3.invert_xaxis()
    ax3.set_title('Excess demand of Good 3', fontweight='bold')
    fig.colorbar(fig3, shrink=0.5, aspect=5)

    return

    ##################################################
    #           Functions for problem 3.3            #                 
    ##################################################

##################################################
#           Functions for problem 3.3            #                 
##################################################

def find_equilibrium(b1,b2,p1,p2,e1,e2,e3,eps,kappa,N,maxiter=25000):
    """
    """
    t = 0
    while True:
        # a. step 1: excess demand
        z1 = np.sum(b1*(p1*e1 + p2*e2 + e3)/p1 - e1)
        z2 = np.sum(b2*(p1*e1 + p2*e2 + e3)/p2 - e2)
            
        # b: step 2: stop?
        if np.abs(z1) < eps and np.abs(z2) < eps or t >= maxiter:
            print(f'{t:3d}: (p1,p2) = ({p1:.2f},{p2:.2f}) -> ({z1:.2f},{z2:.2f})')
            break
             
        # c. step 3: update p1 and p2
        p1 = p1 + kappa*z1/N
        p2 = p2 + kappa*z2/N
                
        # d. step 4: return 
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


    """
    """
    I = p1*e1 + p2*e2 + e3
    
    x1 = b1*(I/p1)
    x2 = b2*(I/p2)
    x3 = b3*I
    
    utility = ((x1**b1)*(x2**b2)*(x3**b3))**gamma
    
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(1,1,1)
    ax.hist(utility,bins=500)
   
    ax.set_xlabel('Utility')
    ax.set_ylabel('# consumers')
    
    return 
