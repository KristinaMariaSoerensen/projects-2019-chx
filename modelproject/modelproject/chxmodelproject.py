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
    return plt.plot(Y_arrey, degree), plt.plot(Y_arrey, AD_arrey)

def ss_cross(T, I, G, NX, a, b):
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

    Returns: 
        Result: Steady state production, Y (float)
    """
    return 1/(1-b) * (I + G + NX + a - b*T)