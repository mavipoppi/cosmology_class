import numpy as np
from   matplotlib import pyplot as plt
import numdifftools as nd
from   scipy.interpolate import CubicSpline
import scipy.integrate as integrate

from   Global import const

class BackgroundCosmology:
  """
  This is a class for the cosmology at the background level.
  It holds cosmological parameters and functions relevant for the background.
  
  Input Parameters: 
    h           (float): The little Hubble parameter h in H0 = 100h km/s/Mpc
    OmegaB      (float): Baryonic matter density parameter at z = 0
    OmegaCDM    (float): Cold dark matter density parameter at z = 0
    OmegaK      (float,optional): Curvative density parameter at z = 0
    name        (float,optional): A name for describing the cosmology
    TCMB        (float,optional): The temperature of the CMB today in Kelvin. Fiducial value is 2.725K
    Neff        (float,optional): The effective number of relativistic neutrinos

  Attributes:    
    OmegaR      (float): Radiation matter density parameter at z = 0
    OmegaNu     (float): Massless neutrino density parameter at z = 0
    OmegaM      (float): Total matter (CDM+b+mnu) density parameter at z = 0
    OmegaK      (float): Curvature density parameter at z = 0
  
  Functions:
    eta_of_x             (float->float) : Conformal time times c (units of length) as function of x=log(a) 
    H_of_x               (float->float) : Hubble parameter as function of x=log(a) 
    dHdx_of_x            (float->float) : First derivative of hubble parameter as function of x=log(a) 
    Hp_of_x              (float->float) : Conformal hubble parameter H*a as function of x=log(a) 
    dHpdx_of_x           (float->float) : First derivative of conformal hubble parameter as function of x=log(a) 
  """
  
  # Settings for integration and splines of eta
  x_start = np.log(1e-10)
  x_end = np.log(100)
  n_pts_splines = 10000

  def __init__(self, h = 0.67, OmegaB = 0.046, OmegaCDM = 0.224, OmegaK = 0.0, 
      name = "FiducialCosmology", TCMB_in_K = 2.725, Neff = 3.046):
    self.h           = h
    self.OmegaB      = OmegaB
    self.OmegaCDM    = OmegaCDM
    self.OmegaK      = OmegaK
    self.H0          = const.H0_over_h * h
    self.name        = name
    self.TCMB        = TCMB_in_K * const.K
    self.Neff        = Neff
 
    # Set the constants
    self.rhoc0       = (3*self.H0)/(8*np.pi*const.G) # Critical density today
    # self.OmegaR      = 2*(np.pi**2/30) * ((const.k_b * TCMB_in_K)**4/(const.hbar**3 * const.c**5)) * ((8*np.pi*const.G)/3*(self.H0/const.H0_over_h)**2) # Radiation
    # self.OmegaNu     = self.Neff * (7/8) * (4/11)**(4/3) * self.OmegaR # Neutrino radiation
    self.OmegaR = 5.44377e-5
    self.OmegaNu = 3.76583e-5
    self.OmegaM      = self.OmegaB + self.OmegaCDM # Total matter
    self.OmegaRtot   = self.OmegaR + self.OmegaNu # Total radiation
    self.OmegaLambda = 1 - self.OmegaM - self.OmegaRtot - self.OmegaK # Dark energy (from Sum Omega_i = 1)
  
  #=========================================================================
  # Methods availiable after solving
  #=========================================================================
  
  def eta_of_x(self,x):
    if not hasattr(self, 'eta_of_x_spline'):
      raise NameError('The spline eta_of_x_spline has not been created') 
    return self.eta_of_x_spline(x)
  def time_of_x(self, x):
    if not hasattr(self, 'time_of_x_spline'):
      raise NameError('The spline time_of_x_spline has not been created') 
    return self.time_of_x_spline(x)
  def H_of_x(self,x):
    return self.H0 * np.sqrt((self.OmegaM)*np.exp(-3*x) + (self.OmegaRtot)*np.exp(-4*x) + self.OmegaK*np.exp(-2*x) + self.OmegaLambda)
  def Hp_of_x(self,x):
    return np.exp(x)* (self.H_of_x(x))
  
  # derivatives 
  def dHdx_of_x(self,x):
    dH = nd.Derivative(self.H_of_x)
    return dH(x)
  def dHpdx_of_x(self,x):
    dHp = nd.Derivative(self.Hp_of_x)
    return dHp(x)
  def dHp2dx2_of_x(self,x):
    dHp2 = nd.Derivative(self.Hp_of_x, n=2)
    return dHp2(x)
  
  # defining the density parameters

  def omega_k(self, x):
    return self.OmegaK / (np.exp(2*x)*self.H_of_x(x)**2 / self.H0**2)

  def omega_totalmatter(self, x):
    return self.OmegaCDM / (np.exp(3*x)*self.H_of_x(x)**2 / self.H0**2) + self.OmegaB / (np.exp(3*x)*self.H_of_x(x)**2 / self.H0**2)
  
  def omega_totalrad(self, x):
    return self.OmegaR / (np.exp(4*x) * self.H_of_x(x)**2 / self.H0**2) + self.OmegaNu / (np.exp(4*x) * self.H_of_x(x)**2 / self.H0**2)
  
  def omega_darkenergy(self, x):
    return self.OmegaLambda / (self.H_of_x(x)**2 / self.H0**2)
  
  # defining new methods
  
  def comoving_distance(self, x):
    return self.eta_zero - self.eta_of_x(x)
  
  def angular_distance(self, x):
    return np.exp(x)*self.comoving_distance(x)
  
  def luminosity_distance(self, x):
    return self.angular_distance(x) / np.exp(2*x)
  
  def redshift(self, x):
    return (1/np.exp(x)) - 1
  
  def hubble_distance(self, x):
    return (self.H0/const.H0_over_h) * self.redshift(x) 
  
  #=========================================================================
  #=========================================================================
  #=========================================================================
  
  def info(self):
    """
    Print some useful info about the class
    """
    print("")
    print("Background Cosmology:")
    print("OmegaB:        %8.7f" % self.OmegaB)
    print("OmegaCDM:      %8.7f" % self.OmegaCDM)
    print("OmegaLambda:   %8.7f" % self.OmegaLambda)
    print("OmegaR:        %8.7e" % self.OmegaR)
    print("OmegaNu:       %8.7e" % self.OmegaNu)
    print("OmegaK:        %8.7f" % self.OmegaK)
    print("TCMB (K):      %8.7f" % (self.TCMB / const.K))
    print("h:             %8.7f" % self.h)
    print("H0:            %8.7e" % self.H0)
    print("H0 (km/s/Mpc): %8.7f" % (self.H0 / (const.km / const.s / const.Mpc)))
    print("Neff:          %8.7f" % self.Neff)
    print("OmegaM:        %8.7f" % self.OmegaM)
    print("OmegaRtot:     %8.7e" % self.OmegaRtot)
  
  def solve(self):
    """
    Main driver for all the solving.
    For LCDM we only need to solve for the conformal time eta(x)
    """

    # Make scale factor array (logspaced)
    x_array  = np.linspace(self.x_start, self.x_end, num = self.n_pts_splines)
    eta_array = np.zeros(self.n_pts_splines)
    time_array = np.zeros(self.n_pts_splines)

    # Compute and spline conformal time eta = Int_0^t dt/a = Int da/(a^2 H(a)) =  Int dx/[ exp(x) * H(exp(x)) ] where x = log a
    for i in range(len(x_array)):
      x = x_array[i]
      eta = integrate.quad(lambda x_p: const.c / (self.Hp_of_x(x_p)), self.x_start, x)
      np.put(eta_array, i, eta)

    # Spline up result
    self.eta_of_x_spline = CubicSpline(x_array, eta_array)
    self.eta_zero = self.eta_of_x_spline(0)

    # Computing and splining cosmic time t = int_(-inf)^x dx/H(x)

    for i in range(len(x_array)):
      x = x_array[i]
      cosmic_time = integrate.quad(lambda x_p: 1 / (self.H_of_x(x_p)), self.x_start, x)
      np.put(time_array, i, cosmic_time)

    # cosmic = time_array*const._c_SI*const.Mpc
    self.time_of_x_spline = CubicSpline(x_array, time_array)



  def plot(self):
    """
    Plot some useful quantities
    """    
    npts = 2000
    xarr = np.linspace(np.log(1e-8), np.log(10), num = npts)
    z_array = np.linspace(0.01, 1000, npts)
    x_array = [-np.log(1+z) for z in z_array]

   # Plotting H(x)

    H_x = [self.H_of_x(xarr[i]) / self.H0 for i in range(npts)]

    # Plotting Hp(x)

    Hp_x = [self.Hp_of_x(xarr[i]) / const.H0_over_h for i in range(npts)]

    # Plotting eta(x)

    eta = [self.eta_of_x(xarr[i]) / const.Mpc for i in range(npts)] 

    # Plotting dHp/dx / Hp(x)

    dHp_dx = [self.dHpdx_of_x(xarr[i]) / self.Hp_of_x(xarr[i]) for i in range(npts)]

    # Plotting DHp2/dx2 / Hp(x)

    dHp2_dx2 = [self.dHp2dx2_of_x(xarr[i]) / self.Hp_of_x(xarr[i]) for i in range(npts)]

    # Plotting the density parameters

    omega_k = [self.omega_k(xarr[i]) for i in range(npts)]
    omega_totalrad = [self.omega_totalrad(xarr[i]) for i in range(npts)]
    omega_darkenergy = [self.omega_darkenergy(xarr[i]) for i in range(npts)]
    omega_totalmatter = [self.omega_totalmatter(xarr[i]) for i in range(npts)]

    # Plotting evolution of cosmological distance measures

    hubble_distance = [self.hubble_distance(x_array[i]) for i in range(npts)]
    luminosity = [self.luminosity_distance(x_array[i]) for i in range(npts)]
    comoving_distance = [self.comoving_distance(x_array[i]) for i in range(npts)]
    angular_distance = [self.angular_distance(x_array[i]) for i in range(npts)]

    # Plotting cosmic time

    cosmic_time = [self.time_of_x(xarr[i]) for i in range(npts)]

    # plt.plot(xarr, cosmic_time)
    # plt.savefig('cosmic.png')
    # Making plots

    # Plotting eta*H_p/c

    eta_h_p = [self.eta_of_x(xarr[i])*self.Hp_of_x(xarr[i])/ const.c for i in range(npts)]

    # plt.plot(xarr, eta_h_p)
    # plt.yticks(np.arange(0.75, 3, 0.25))
    # plt.xlim(-14,0)
    # plt.ylim(0.75, 3)
    # plt.savefig('eta_H.png')
    
    fig, axs = plt.subplots(2,3, figsize=(18,12), dpi=500)

    axs[0][0].set_title('Evolution of Hubble factor')
    axs[0][1].set_title(r'Evolution of scaled Hubble factor $\mathcal{H} = aH$')
    axs[0][2].set_title('Evolution of density parameters')
    axs[1][0].set_title('Evolution of conformal time')
    axs[1][1].set_title(r'Evolution of Hubble factors (derivatives of $\mathcal{H} = aH$)')
    axs[1][2].set_title('Evolution of cosmological distance measures')

    axs[0][0].plot(xarr, H_x, color='blue') # H(x)
    axs[0][1].plot(xarr, Hp_x, color='blue') # Hp(x)
    axs[0][2].plot(xarr, omega_totalrad, color='orange', label=r'$\Omega_{TotalRadiation}$')
    axs[0][2].plot(xarr, omega_darkenergy, color='blue', label=r'$\Omega_{DarkEnergy}$')
    axs[0][2].plot(xarr, omega_k, color='red', label=r'$\Omega_{Curvature}$')
    axs[0][2].plot(xarr, omega_totalmatter, color='green', label=r'$\Omega_{TotalMatter}$')
    axs[1][0].plot(xarr, eta, color='blue')
    axs[1][1].plot(xarr, dHp_dx, color='blue', label=r'$\frac{d\mathcal{H}}{dx} \cdot \frac{1}{\mathcal{H}}$')
    axs[1][1].plot(xarr, dHp2_dx2, color='orange', label=r'$\frac{d\mathcal{H}^2}{dx^2} \cdot \frac{1}{\mathcal{H}}$')
    axs[1][2].plot(z_array, hubble_distance, color='black', label=r'Naive Hubble distance $(d = H_0 z)$', linestyle='-')
    axs[1][2].plot(z_array, luminosity, color='blue', label='Luminosity distance')
    axs[1][2].plot(z_array, comoving_distance, color='orange', label='Comoving distance')
    axs[1][2].plot(z_array, angular_distance, color='red', label='Angular diameter distance')

    axs[0][0].set_ylabel(r'$H(x)$')
    axs[0][0].set_xlabel(r'$x = log(a)$')
    axs[0][1].set_ylabel(r'$\mathcal{H}(x)$')
    axs[0][1].set_xlabel(r'$x = log(a)$')
    axs[0][2].set_ylabel(r'$\Omega$')
    axs[0][2].set_xlabel(r'$x=log(a)$')
    axs[1][0].set_ylabel(r'$\eta(x)$')
    axs[1][0].set_xlabel(r'$x = log(a)$')
    axs[1][1].set_ylabel(r'$d\mathcal{h}/dx 1/\mathcal{H}$, $d\mathcal{H}^2/dx^2 1/\mathcal{H}$')
    axs[1][1].set_xlabel(r'$x = log(a)$')
    axs[1][2].set_ylabel('Distance (Mpc)')
    axs[1][2].set_xlabel('Redshift z')

    axs[0][0].axvline(x=0, color= 'black', ls=':')
    axs[0][2].axvline(x=0, color='black', ls=':')
    axs[0][2].axhline(y=0, color='black', ls=':')
    axs[1][0].axvline(x=0, color= 'black', ls=':')
    axs[1][1].axvline(x=0, color= 'black', ls=':')
    axs[1][1].axhline(y=0, color= 'black', ls=':')

    axs[0][0].set_xlim(-18, 2)
    axs[0][1].set_xlim(-12, 0)
    axs[0][1].set_ylim(1e-1, 1e4)
    axs[0][2].set_xlim(-18,2)
    axs[1][0].set_xlim(-18,2)
    axs[1][1].set_xlim(-18,2)
    axs[1][2].set_xlim(1e-2, 1e3)

    axs[0][0].set_yscale('log')
    axs[0][1].set_yscale('log')
    axs[1][0].set_yscale('log')
    axs[1][2].set_yscale('log')
    axs[1][2].set_xscale('log')

    axs[0][2].legend()
    axs[1][1].legend()
    axs[1][2].legend()
    plt.tight_layout()
    plt.savefig('H0_cosmology.png')  


  # =========================================================================
  # =========================================================================
  # =========================================================================
  

