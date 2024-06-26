import numpy as np
from   matplotlib import pyplot as plt
import scipy.integrate as integrate
from   scipy.interpolate import CubicSpline
from   scipy.integrate import solve_ivp
import warnings
from mpmath import *
from   Global import const
import BackgroundCosmology

class RecombinationHistory:
  """
  This is a class for solving the recombination (and reionization) history of the Universe.
  It holds recombination parameters and functions relevant for the recombination history.
  
  Input Parameters: 
    cosmo (BackgroundCosmology) : The cosmology we use to solve for the recombination history
    Yp                   (float): Primordial helium fraction
    reionization         (bool) : Include reionization or not
    z_reion              (float): Reionization redshift
    delta_z_reion        (float): Reionization width
    helium_reionization  (bool) : Include helium+ reionization
    z_helium_reion       (float): Reionization redshift for helium+
    delta_z_helium_reion (float): Reionization width for helium+

  Attributes:    
    tau_reion            (float): The optical depth at reionization
    z_star               (float): The redshift for the LSS (defined as peak of visibility function or tau=1)

  Functions:
    tau_of_x             (float->float) : Optical depth as function of x=log(a) 
    dtaudx_of_x          (float->float) : First x-derivative of optical depth as function of x=log(a) 
    ddtauddx_of_x        (float->float) : Second x-derivative of optical depth as function of x=log(a) 
    g_tilde_of_x         (float->float) : Visibility function dexp(-tau)dx as function of x=log(a) 
    dgdx_tilde_of_x      (float->float) : First x-derivative of visibility function as function of x=log(a) 
    ddgddx_tilde_of_x    (float->float) : Second x-derivative of visibility function as function of x=log(a)
    Xe_of_x              (float->float) : Free electron fraction dXedx as function of x=log(a) 
    ne_of_x              (float->float) : Electron number density as function of x=log(a) 
  """

  # Settings for solver
  x_start               = np.log(1e-8)
  x_end                 = np.log(1.0)
  npts                  = 1000
  npts_tau_before_reion = 1000
  npts_tau_during_reion = 1000
  npts_tau_after_reion  = 1000
  Xe_saha_limit         = 0.99
    
  def __init__(self, BackgroundCosmology, Yp = 0.24, 
      reionization = False, z_reion = 11.0, delta_z_reion = 0.5, 
      helium_reionization = False, z_helium_reion = 3.5, delta_z_helium_reion = 0.5):
    self.cosmo            = BackgroundCosmology
    
    self.Yp               = Yp
    
    self.reionization     = reionization
    self.z_reion          = z_reion
    self.delta_z_reion    = delta_z_reion
    
    self.helium_reionization  = helium_reionization
    self.z_helium_reion       = z_helium_reion
    self.delta_z_helium_reion = delta_z_helium_reion
  
  #=========================================================================
  # Methods availiable after solving
  #=========================================================================
  
  def tau_of_x(self,x):
    if not hasattr(self, 'tau_of_x_spline'):
      raise NameError('The spline tau_of_x_spline has not been created') 
    return self.tau_of_x_spline(x)
  def dtaudx_of_x(self,x):
    if not hasattr(self, 'dtaudx_of_x_spline'):
      raise NameError('The spline dtaudx_of_x_spline has not been created') 
    return self.dtaudx_of_x_spline(x)
  def ddtauddx_of_x(self,x):
    if not hasattr(self, 'ddtauddx_of_x_spline'):
      raise NameError('The spline ddtauddx_of_x_spline has not been created') 
    return self.ddtauddx_of_x_spline(x)
  def g_tilde_of_x(self,x):
    if not hasattr(self, 'g_tilde_of_x_spline'):
      raise NameError('The spline g_tilde_of_x_spline has not been created') 
    return self.g_tilde_of_x_spline(x)
  def dgdx_tilde_of_x(self,x):
    if not hasattr(self, 'dgdx_tilde_of_x_spline'):
      raise NameError('The spline dgdx_tilde_of_x_spline has not been created') 
    return self.dgdx_tilde_of_x_spline(x)
  def ddgddx_tilde_of_x(self,x):
    if not hasattr(self, 'ddgddx_tilde_of_x_spline'):
      raise NameError('The spline ddgddx_tilde_of_x_spline has not been created') 
    return self.ddgddx_tilde_of_x_spline(x)
  def Xe_of_x(self,x):
    if not hasattr(self, 'log_Xe_of_x_spline'):
      raise NameError('The spline log_Xe_of_x_spline has not been created') 
    return self.log_Xe_of_x_spline(x)
  def ne_of_x(self,x):
    if not hasattr(self, 'log_ne_of_x_spline'):
      raise NameError('The spline log_ne_of_x_spline has not been created') 
    return np.exp(self.log_ne_of_x_spline(x))

  #=========================================================================
  #=========================================================================
  #=========================================================================
  
  def info(self):
    print("")
    print("Recombination History:")
    print("Yp:                   %8.7f" % self.Yp)
    print("reionization:         %8.7f" % self.reionization)
    print("z_reion:              %8.7f" % self.z_reion)
    print("delta_z_reion:        %8.7f" % self.delta_z_reion)
    print("helium_reionization:  %8.7f" % self.helium_reionization)
    print("z_helium_reion:       %8.7f" % self.z_helium_reion)
    print("delta_z_helium_reion: %8.7f" % self.delta_z_helium_reion)

  def solve(self):
    """
    Main driver for doing all the solving
    We first compute Xe(x) and ne(x)
    Then we compute the optical depth tau(x) and the visibility function g(x)
    """
    self.solve_number_density_electrons()
    
    self.solve_for_optical_depth_tau()
 
    # Compute z_star (peak of visibility function or tau = 1)
    # XXX TODO XXX
    
    # PhD: compute optical depth at reionization
    # XXX TODO XXX

  def plot(self):
    """
    Make some useful plots
    """
    
    npts   = 10000
    xarr   = np.linspace(self.x_start, self.x_end, num = npts)
    Xe     = [self.Xe_of_x(xarr[i]) for i in range(npts)]
    # ne     = [self.ne_of_x(xarr[i]) for i in range(npts)]
    # tau    = [self.tau_of_x(xarr[i]) for i in range(npts)]
    # dtaudx = [-self.dtaudx_of_x(xarr[i]) for i in range(npts)]
    # ddtauddx = [self.ddtauddx_of_x(xarr[i]) for i in range(npts)]
    # g_tilde      = self.g_tilde_of_x(xarr)
    # dgdx_tilde   = self.dgdx_tilde_of_x(xarr)
    # ddgddx_tilde = self.ddgddx_tilde_of_x(xarr)
    
    # Recombination g_tilde
    # plt.xlim(-7.5,-6.5)
    # plt.ylim(-4,6)
    # plt.title('Visibility function and derivatives close to recombination')
    # plt.plot(xarr, g_tilde, xarr, dgdx_tilde/15., xarr, ddgddx_tilde/300.)
    # plt.show()
    
    # # Reionization g_tilde
    # plt.xlim(-2.7,-2.0)
    # plt.ylim(-0.15,0.15)
    # plt.title('Visibility function and derivatives close to reionization')
    # plt.plot(xarr, g_tilde, xarr, dgdx_tilde/15., xarr, ddgddx_tilde/300.)
    # plt.show()
    
    # Xe(x) of x
    plt.title('Free electron fraction')
    plt.plot(xarr, Xe)
    plt.xlim(-12.5, 0)
    plt.yscale('log')
    plt.show()

    # # ne of x
    # plt.yscale('log')
    # plt.title('Electron numberdensity')
    # plt.plot(xarr, ne)
    # plt.show()
    
    # tau
    # plt.yscale('log')
    # plt.title('Tau and derivatives')
    # plt.ylim(1e-8,1e8)
    # plt.plot(xarr, tau, xarr, dtaudx, xarr, ddtauddx)
    # plt.show()
    
  #=========================================================================
  #=========================================================================
  #=========================================================================

  def solve_number_density_electrons(self):
    """
    Solve for the evolution of the electron number density by solving
    the Saha and Peebles equations
    """
     
    # Settings for the arrays we use below
    npts    = self.npts
    x_start = self.x_start
    x_end   = self.x_end
    
    # Set up arrays to compute X_e and n_e on
    x_array = np.linspace(x_start, x_end, num=npts)
    Xe_arr  = np.zeros(npts)
    ne_arr  = np.zeros(npts)

    Xe_saha_limit = 0.99

    # Calculate recombination history
    for i in range(npts):  
      # Current scale factor
      x = x_array[i]
      a = np.exp(x)

      # Physical constants
      k_b         = const.k_b;
      G           = const.G;
      c           = const.c;
      m_e         = const.m_e;
      hbar        = const.hbar;
      m_H         = const.m_H;
      epsilon_0   = const.epsilon_0;
      xhi0        = const.xhi0;
      xhi1        = const.xhi1;
      # Cosmological parameters and variables 
      Yp          = self.Yp
      OmegaB      = self.cosmo.OmegaB
      TCMB        = self.cosmo.TCMB
      H0          = self.cosmo.H0
      H           = self.cosmo.H_of_x(x)

      rho_b = (3*H0**2)/(8*np.pi*G) * (OmegaB / (a**3))
      n_b = rho_b / m_H

      #===========================================
      # Get f_e from solving the Saha equation
      #==============================================================
      Xe_current, ne_current = self.electron_fraction_from_saha_equation(x)
      
      # Two regimes: Saha and Peebles regime
      # if(Xe_current > Xe_saha_limit):
      if False:

        # Store the results from the Saha equation
        Xe_arr[i] = Xe_current
        ne_arr[i] = ne_current 

      else:

        #==============================================================
        # We need to solve the Peebles equation for the rest of the time
        #==============================================================
    
        # Make x-array for Peebles system from current time till the end
        x_interval = (x_array[i], x_end)
        x_arraynew = x_array[i:]
        # print(len(x_arraynew))
        # Set initial conditions
        y = [Xe_current]
       
        # Solve the Peebles ODE 
        peebles = solve_ivp(self.rhs_peebles_ode, x_interval, y, t_eval= x_arraynew)

        # Fill up array with the result
        for j in range(i, npts - 1):
          # print(i, j, npts)
          # print(peebles.y[0][j-i])
          Xe_arr[j] = peebles.y[0][j-i]
          ne_arr[j] = peebles.y[0][j-i]*n_b

          
        # We are done so exit for loop
        break
      
    # Xe_arr = [Xe_arr[i] if Xe_arr[i] != 0  else 1E-6 for i in range(len(Xe_arr)) ]

    # print(Xe_arr)
    # Make splines of log(Xe) and log(ne) as function of x = log(a)
    # XXX TODO XXX
    self.log_Xe_of_x_spline = CubicSpline(x_array, Xe_arr)
    # self.log_ne_of_x_spline = CubicSpline(x_array, np.log(ne_arr))

    return
 
  def solve_for_optical_depth_tau(self):
    """
    Solve for the optical depth tau(x) by integrating up
    dtaudx = -c sigmaT ne/H
    (PhD: Include the effects of reionization if z_reion > 0)
    """

    # Set up x_array
    # XXX TODO XXX

    # Set initial conditions for tau
    # XXX TODO XXX

    # Solve the tau ODE and normalize it such that tau(0) = 0.0
    # XXX TODO XXX

    # Spline it up
    # XXX TODO XXX
    # self.tau_of_x_spline = CubicSpline(x_array, tau)
    
    # Compute and spline the derivatives of tau
    # XXX TODO XXX
    # self.dtaudx_of_x_spline = CubicSpline(x_array, dtaudx)
    # self.ddtauddx_of_x_spline = CubicSpline(x_array, ddtauddx)

    # Compute and spline visibility function and it derivatives
    # XXX TODO XXX
    # self.g_tilde_of_x_spline  = CubicSpline(x_array, g_tilde_of_x)
    # self.dgdx_tilde_of_x_spline = CubicSpline(x_array, dgdx_tilde_of_x)
    # self.ddgddx_tilde_of_x_spline = CubicSpline(x_array, ddgddx_tilde_of_x)

    return

  def electron_fraction_from_saha_equation(self,x):
    """
    Solve the Saha equations for hydrogen and helium recombination
    Returns: Xe, ne with Xe = ne/nH beging the free electron fraction
    and ne the electon number density
    """

    # Physical constants
    k_b         = const.k_b;
    G           = const.G;
    c           = const.c;
    m_e         = const.m_e;
    hbar        = const.hbar;
    m_H         = const.m_H;
    epsilon_0   = const.epsilon_0;
    xhi0        = const.xhi0;
    xhi1        = const.xhi1;
    # Cosmological parameters and variables 
    a           = np.exp(x)
    Yp          = self.Yp
    OmegaB      = self.cosmo.OmegaB
    TCMB        = self.cosmo.TCMB
    H0          = self.cosmo.H0
    H           = self.cosmo.H_of_x(x)

    rho_b = (3*H0**2)/(8*np.pi*G) * (OmegaB / (a**3))
    n_b = rho_b / m_H

    # Solve Saha equation for Xe
   
    RSE = (1 / n_b) * ((k_b*(TCMB/a)*m_e)/(2*np.pi*hbar**2))**(3/2) * np.exp(-epsilon_0 / (k_b * (TCMB/a)))
    delta = RSE**2 - 4*RSE
    Xe = (- RSE + delta)/(2*RSE)

    ne = Xe * n_b

    # Return Xe and ne
    return Xe, ne
  
  def rhs_tau_ode(self, x, y):
    """
    Right hand side of the optical depth ODE dtaudx = RHS
    """
   
    # Set the right hand side
    # XXX TODO XXX
    dtaudx = 1.0
    return dtaudx

  def rhs_peebles_ode(self, x, y):
    """
    Right hand side of Peebles ODE for the free electron fraction dXedx = RHS
    """

    # Solver variables
    X_e         = mpf(y[0])
    a           = mpf(np.exp(x))

    # Physical constants 
    k_b         = mpf(const.k_b)
    G           = mpf(const.G)
    c           = mpf(const.c)
    m_e         = mpf(const.m_e)
    hbar        = mpf(const.hbar)
    m_H         = mpf(const.m_H)
    sigma_T     = mpf(const.sigma_T)
    lambda_2s1s = mpf(const.lambda_2s1s)
    epsilon_0   = mpf(const.epsilon_0)

    # Cosmological parameters 
    Yp          = mpf(self.Yp)
    OmegaB      = mpf(self.cosmo.OmegaB)
    TCMB        = mpf(self.cosmo.TCMB)
    H0          = mpf(self.cosmo.H0)
    H           = mpf(self.cosmo.H_of_x(x))

    # Set right hand side of the Peebles equation

    phi_Tb = mpf(0.448)*ln(epsilon_0/(k_b*(TCMB/a))) # dimensionless
    # print(f'Phi_tb = {phi_Tb}')

    fine_struct = mpf(1) / mpf(137.0359992) 

    Sigma_T = (mpf(8)*pi/mpf(3)) * ((fine_struct*hbar*c) / (m_e * c**2))**2
    # print(f'Sigma_T = {sigma_T}')

    alpha_Tb = (mpf(8)/sqrt(mpf(3)*pi))*c*sigma_T*sqrt(epsilon_0/(k_b*TCMB/a)) * phi_Tb # dimension: mˆ3/s
    # print(f'Alpha_tb = {alpha_Tb}')

    beta_Tb = alpha_Tb * ((m_e*k_b*TCMB/a)/(mpf(2)*pi*hbar**2))**(3/2) * exp(-epsilon_0/(k_b*TCMB/a)) # dimension: 1/s
    # print(f'Beta_tb = {beta_Tb}')

    beta2_Tb = beta_Tb*exp((mpf(3)*epsilon_0)/(mpf(4)*k_b*TCMB/a)) # dimension: 1/s
    # print(f'Beta2_tb = {beta2_Tb}')

    n_H = (mpf(3)*OmegaB*H0**2*(1-Yp))/(mpf(8)*np.pi*G*a**mpf(3)*m_H)
    # print(f'n_H = {n_H}')

    n1s = (mpf(1) - X_e) * n_H
    # print(f'n1s = {n1s}')

    lambda_a = H * ((mpf(3)*epsilon_0)**3) / ((mpf(8)*np.pi)**2 * c**3 * hbar**3 * n1s) # dimension 1/s
    # print(f'Lambda_a = {lambda_a}')

    Cr_Tb = (lambda_2s1s + lambda_a)/(lambda_2s1s+lambda_a+beta2_Tb) # dimensionless
    # print(f'Cr_Tb = {Cr_Tb}')

    dXedx = (Cr_Tb/(H))*(beta_Tb*(mpf(1)-X_e) - n_H*alpha_Tb*X_e**2)

    # print(f'dXe_de = {dXedx}')
    # if np.isnan(dXedx):
    #   print(f'Phi_tb = {phi_Tb}')
    #   print(f'sigma_T = {sigma_T}')
    #   print(f'Sigma_T = {Sigma_T}')
    #   print(f'Alpha_tb = {alpha_Tb}')
    #   print(f'Beta_tb = {beta_Tb}')
    #   print(f'Beta2_tb = {beta2_Tb}')
    #   print(f'n_H = {n_H}')
    #   print(f'n1s = {n1s}')
    #   print(f'Lambda_a = {lambda_a}')
    #   print(f'Cr_Tb = {Cr_Tb}')
    #   print('')


    # print(dXedx)

    return dXedx