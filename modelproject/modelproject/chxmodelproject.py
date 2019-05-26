import numpy as np
from scipy import optimize
#%matplotlib inline
import matplotlib.pyplot as plt


def keynesian_cross(T, I, G, NX, a, b):
    """ Draws the Keynesian cross with the 45-degree line and 
    the planned total spending as a function of total production.
    
    Args:
        T (float): Taxs
        a (float): Constant consumption, a>0
        b (float): Marginal consumption rate, 0<b<1
        I (float): Investment
        G (float): Public expenditure
        NX (float): Net export

    Return: Figure
    """
    # The data vector to be plotted for production and aggregate expenditure:
    Y_arrey = np.linspace(0,300)
    AD_arrey = (a + b * (Y_arrey - T) + I + G + NX)
    degree = Y_arrey

    # The figure
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(1,1,1)

    ax.plot(Y_arrey, degree, label="45-degree line", color='lightblue',linewidth=3)
    ax.plot(Y_arrey, AD_arrey, label="AD=C+I+G+NX", color='darkorange',linewidth=3)

    ax.set_xlabel("Y")
    ax.set_ylabel("AD")
    ax.legend(loc="upper left")

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return

def cross_equalibrium(T, I, G, NX, a, b):
    """ The equalibrium for the Keynesian cross where aggregate expenditure equals total production
    Args:
        T (float): Tax
        a (float): Constant consumption, a>0
        b (float): Marginal consumption rate, 0<b<1
        I (float): Investment
        G (float): Public expenditure
        NX (float): Net export
    Returns: 
        Result: Production in equalibrium, Y (float)
    """
    return 1/(1-b) * (I + G + NX + a - b*T)

def keynesian_cross_NXshift(T, I, G, NX, a, b, delta_NX):
    """ Steady state for the Keynesian cross where aggregate expenditure equals total production  
    Args:
        AD (float): Aggregate expenditure
        Y (float): Total production
        T (float): Tax
        a (float): Constant consumption, a>0
        b (float): Marginal consumption rate, 0<b<1
        I (float): Investment
        G (float): Public expenditure
        NX (float): Net export
        delta_NX (float): Net export shift in 
    Returns: 
        Result: Figure
    """
    # The equation setup
    NX2 = NX + delta_NX
    Y_arrey = np.linspace(0,300)
    AD_arrey = (a + b * (Y_arrey - T) + I + G + NX)
    AD2_arrey = (a + b * (Y_arrey - T) + I + G + NX2)
    degree = Y_arrey

    # The figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    ax.plot(Y_arrey, degree, label="45-degree line", color='lightblue')
    ax.plot(Y_arrey, AD_arrey, label="AD=C+I+G+NX", color='orange')
    ax.plot(Y_arrey, AD2_arrey, label="AD'=C+I+G+NX'", color='red')

    ax.set_xlabel("Y")
    ax.set_ylabel("AD")
    ax.legend(loc="upper left")

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return

def num_opt(Y_goal,T,I,G,a,b):
    """ Numerical optimazation to calculate value of NX to optain production goal  
    Args:
        Y_goal (float): Production goal
        T (float): Tax
        a (float): Constant consumption, a>0
        b (float): Marginal consumption rate, 0<b<1
        I (float): Investment
        G (float): Public expenditure
    Returns: 
        Result: NX (float)
    """
    # Object function to be optimized:
    obj = lambda NX: (cross_equalibrium(T, I, G, NX, a, b) - Y_goal)**2

    # Initial guess
    x0 = 10
    return optimize.minimize(obj,x0)

def keynesian_cross_NXshift_t(k, t, I, G, NX, a, b, delta_NX):
    """ Steady state for the Keynesian cross where aggregate expenditure equals total production
    Args:
        AD (float): Aggregate expenditure
        Y (float): Total production
        k (float): Base tax
        t (float): Marginal tax rate
        a (float): Constant consumption, a>0
        b (float): Marginal consumption rate, 0<b<1
        I (float): Investment
        G (float): Public expenditure
        NX (float): Net export
        delta_NX (float): Net export shift in 
    Returns: 
        Result: Figure
    """
    # The equation setup and generating of data arreys:
    NX2 = NX + delta_NX
    Y_arrey = np.linspace(0,300)
    AD_arrey = (a + b * (Y_arrey - (k + b*t)) + I + G + NX)
    AD2_arrey = (a + b * (Y_arrey - (k + b*t)) + I + G + NX2)
    degree = Y_arrey

    # The figure:
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    ax.plot(Y_arrey, degree, label="45-degree line", color='lightblue')
    ax.plot(Y_arrey, AD_arrey, label="AD=C+I+G+NX", color='orange')
    ax.plot(Y_arrey, AD2_arrey, label="AD'=C+I+G+NX'", color='red')

    ax.set_xlabel("Y")
    ax.set_ylabel("AD")
    ax.legend(loc="upper left")

    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)  
    return 