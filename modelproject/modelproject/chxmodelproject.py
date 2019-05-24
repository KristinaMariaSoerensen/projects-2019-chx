import numpy as np
from scipy import optimize
#%matplotlib inline
import matplotlib.pyplot as plt

def keynesian_cross(T, I, G, NX, a, b):
    """ Draws the Keynesian cross with the 45-degree line and 
    the planned total spending as a function of total production.
    
    Args:
    AD (float): Aggregate expenditure
    Y (float): Total production
    T (float): Taxs
    a (float): Constant consumption, a>0
    b (float): Marginal consumption rate, 0<b<1
    I (float): Investment
    G (float): Public expenditure
    NX (float): Net export

    Return: Figure
    """

    Y_arrey = np.linspace(0,300)
    AD_arrey = (a + b * (Y_arrey - T) + I + G + NX)
    degree = Y_arrey

    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(Y_arrey, degree, label="45-degree line")
    ax.plot(Y_arrey, AD_arrey, label="AD=C+I+G+NX")

    ax.set_xlabel("Y")
    ax.set_ylabel("AD")

    ax.legend(loc="upper left")

    return fig

def cross_equalibrium(T, I, G, NX, a, b):
    """ The equalibrium for the Keynesian cross where aggregate expenditure equals total production
    
    Args:
    AD (float): Aggregate expenditure
    Y (float): Total production
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
    NX2 = NX + delta_NX
    Y_arrey = np.linspace(0,300)
    AD_arrey = (a + b * (Y_arrey - T) + I + G + NX)
    AD2_arrey = (a + b * (Y_arrey - T) + I + G + NX2)
    degree = Y_arrey

    fig = plt.figure(figsize=(8,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(Y_arrey, degree, label="45-degree line")
    ax.plot(Y_arrey, AD_arrey, label="AD=C+I+G+NX")
    ax.plot(Y_arrey, AD2_arrey, label="AD'=C+I+G+NX'")

    ax.set_xlabel("Y")
    ax.set_ylabel("AD")

    ax.legend(loc="upper left")
    
    return fig

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
    NX2 = NX + delta_NX
    Y_arrey = np.linspace(0,300)
    AD_arrey = (a + b * (Y_arrey - (k + b*t)) + I + G + NX)
    AD2_arrey = (a + b * (Y_arrey - (k + b*t)) + I + G + NX2)
    degree = Y_arrey

    fig = plt.figure(figsize=(8,6), dpi=100)
    ax = fig.add_subplot(1,1,1)

    ax.plot(Y_arrey, degree, label="45-degree line")
    ax.plot(Y_arrey, AD_arrey, label="AD=C+I+G+NX")
    ax.plot(Y_arrey, AD2_arrey, label="AD'=C+I+G+NX'")


    ax.set_xlabel("Y")
    ax.set_ylabel("AD")

    ax.legend(loc="upper left")
    
    return fig