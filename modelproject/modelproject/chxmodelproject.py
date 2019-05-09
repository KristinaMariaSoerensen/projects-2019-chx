import numpy as np
from scipy import optimize
#%matplotlib inline
import matplotlib.pyplot as plt

def inflation(u, u_n, beta):
    """ the basic Phillips-curve. Inflation as a function of unemployment.

    Args:
    u (float): Unemployment
    u_n (float): Natural employment rate
    beta (float): Parameter


    Returns:
        result (float): Inflation

    """ 
    return -beta * (u - u_n)
    

def simple_phillips_curve(u_n, beta):
    """ Illustrates the simple Phillips curve. The trade-off between low unemployment and low inflation
    Args:
    u (float): Unemployment
    u_n (float): Natural employment rate
    beta (float): Parameter

    Return: Figure
    """

    u_arrey = np.linspace(0, 100)
    pi_arrey = - beta * (u_arrey - u_n)

    return plt.plot(u_arrey, pi_arrey)
    
#def inflation_updated(u, u_n, beta)
