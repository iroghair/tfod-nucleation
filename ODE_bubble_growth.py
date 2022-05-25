from cmath import pi
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# substances
rho_l = 1000 # liquid density [kg/m^3]
rho_g = 10.191 # gas density [kg/m^3]
MW = 44.01 * 10**(-3) # molecular weight of CO2 [kg/mol]
D = 1.97 * 10**(-9) # diffusion coeff CO2 in water [m^2/s]

# k_H = 3.79 * 10**(-7) # Alessandro: Henry constant [mol/kg/Pa]
k_H = 0.035 # [mol/kg/bar] NIST

# Critical radius
sigma = 0.069 # surface tension [N/m] Batistella p.98

#p_0 = 6.4 # initial pressure [bar]
#p_S = 5.5 # pressure after drop [bar]

def get_Rb(p_0=6.4,p_S=5.5,t_max=1801):
    # time vector
    t = np.arange(1,t_max,10) # [s]
    # concentrations: Henry's law
    c_0 = k_H * p_0 # [mol/kg]
    c_S = k_H * p_S # [mol/kg]
    beta = (c_0-c_S) * rho_l/rho_g * MW
    zeta = p_0/p_S - 1 # supersaturation ratio [-]
    # initial condition (Bubble size)
    R_c = (2 * sigma) / (p_S * 10**5 * zeta) # critical radius [m]
    # RHS ODE
    def model(R,t):
        """right hand side of the ODE
        dydt = model(y,t)"""
        dRdt = D * beta * (1/math.sqrt(pi*D*t) + 1/R)
        return dRdt
    # solve ODE
    R_b = odeint(model, R_c, t) # [m]
    return R_b, t

# plot results
if __name__ == '__main__':
    Rb, tb = get_Rb()
    plt.plot(tb,Rb)
    plt.xlabel('Time [s]')
    plt.ylabel('Bubble radius $R_b$(t) [m]')
    plt.savefig('Test_ODE_Rb.png')