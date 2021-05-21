# Ab initio Elasticity and  Thermodynamics of Minerals
#
# Version 2.4.0 18/02/2021
#

# Comment the following four lines to produce the documentation 
# with readthedocs


# from IPython import get_ipython
# get_ipython().magic('cls')
# get_ipython().magic('reset -sf')

import datetime
import os  
import sys  
import scipy
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import pandas as pd
import sympy as sym
import parame as pr

from scipy.optimize import curve_fit, fmin, minimize_scalar, minimize
from scipy.interpolate import UnivariateSpline, Rbf
from scipy import integrate

from mineral_data import mineral, load_database, equilib, reaction,\
     pressure_react, export, field, import_database, name_list
from mineral_data import ens, cor, py, coe, q, fo, ky, sill, andal, per, sp, \
     mao, fmao, stv
    
import_database()

mpl.rcParams['figure.dpi']= 80

class flag:
    def __init__(self,value):
        self.value=value
        self.jwar=0
    def on(self):
        self.value=True
    def off(self):
        self.value=False
    def inc(self):
        self.jwar += 1
    def reset(self):
        self.jwar=0
        
class verbose_class():
    def __init__(self,value):
        self.flag=value
    def on(self):
        self.flag=True
        print("Verbose mode on")
    def off(self):
        self.flag=False
        print("Verbose mode off")
        
class BM3_error(Exception):
      pass
        
class vol_corr_class:
      def __init__(self):
          self.flag=False 
          self.v0_init=None
      def on(self):
          self.flag=True
      def off(self):
          self.flag=False
      def set_volume(self,vv):
          self.v0_init=vv
                
class data_info():
    """
    Stores information about the current settings
    """
    def __init__(self):
        self.min_static_vol=None
        self.max_static_vol=None
        self.static_points=None
        self.min_freq_vol=None
        self.max_freq_vol=None
        self.freq_points=None
        self.min_select_vol=None
        self.max_select_vol=None
        self.select_points=None
        self.freq_sets=None
        self.fit_type='No fit'
        self.min_vol_fit=None
        self.max_vol_fit=None
        self.fit_points=None
        self.fit_degree=None
        self.fit_smooth=None
        self.k0=None
        self.kp=None
        self.v0=None
        self.temp=None
        self.k0_static=None
        self.kp_static=None
        self.v0_static=None
        self.min_names=name_list.mineral_names
        self.title=None
    
    def show(self):
        """
        Prints information about the current settings stored in the classes
        """
        if self.title !=None:
            print(self.title)
        print("\nCurrent settings and results\n")
        if self.min_static_vol != None:
           print("Static data            ** min, max volumes: %8.4f, %8.4f; points: %d"\
             % (self.min_static_vol, self.max_static_vol, self.static_points))
        if self.min_freq_vol != None:
           print("Frequency volume range ** min, max volumes: %8.4f, %8.4f; points: %d"\
             % (self.min_freq_vol, self.max_freq_vol, self.freq_points))     
        if self.min_select_vol != None:
           print("Selected freq. sets    ** min, max volumes: %8.4f, %8.4f; points: %d"\
             % (self.min_select_vol, self.max_select_vol, self.select_points))
           print("Frequency sets:           %s" % str(self.freq_sets))
        if self.fit_type != 'No fit':
           if self.fit_type=='poly':
              print("\nFit of frequencies     ** type: %s, degree: %d" \
                    % (self.fit_type, self.fit_degree))
           else:
              print("\nFit of frequencies     ** type: %s, degree: %d, smooth: %2.1f" \
                    % (self.fit_type, self.fit_degree, self.fit_smooth)) 
              
           print("                          min, max volumes: %8.4f, %8.4f; points %d" %\
                 (self.min_vol_fit, self.max_vol_fit, self.fit_points))
        else:
           print("No fit of frequencies")
           
        if supercell.flag:
           print("\n*** This is a computation performed on SUPERCELL data")
           print("    (SCELPHONO and QHA keywords in CRYSTAL). Number of cells: %3i" % supercell.number)
            
        if self.k0_static != None:
           print("\n** Static EoS (BM3) **")
           print("K0: %6.2f GPa, Kp: %4.2f, V0: %8.4f A^3" %\
                 (self.k0_static, self.kp_static, self.v0_static))
        if static_range.flag:
            print("\n** Static EoS is from a restricted volume range:")
            print("Minimum volume: %8.3f" % static_range.vmin)
            print("Maximum volume: %8.3f" % static_range.vmax)
            
        if self.k0 != None:
            print("\n** BM3 EoS from the last computation, at the temperature of %5.2f K **" % self.temp)
            print("K0: %6.2f GPa, Kp: %4.2f, V0: %8.4f A^3" %\
                 (self.k0, self.kp, self.v0))
            if not f_fix.flag:
                print("Kp not fixed")
            else:
                print("Kp fixed")
                
        if exclude.ex_mode != []:
            print("\nZone center excluded modes: %s" % str(exclude.ex_mode))
        else:
            print("\nNo zone center excluded modes")
            
        if disp.ex_flag:
            print("Off center excluded modes: %s" % str(disp.excluded_list))
        else:
            print("No off center excluded modes")
        
        if kieffer.flag==True:
           print("\nKieffer model on; frequencies %5.2f %5.2f %5.2f cm^-1" %\
                (kieffer.kief_freq_inp[0], kieffer.kief_freq_inp[1], \
                 kieffer.kief_freq_inp[2]))
        else:
           print("\nKieffer model off")
           
        if anharm.flag:
            print("\nAnharmonic correction for mode(s) N. %s" % str(anharm.mode).strip('[]'))
            print("Brillouin flag(s): %s" % str(anharm.brill).strip('[]'))
            
        if disp.flag:
            print("\nDispersion correction activated for the computation of entropy and")
            print("specific heat:")
            print("Number of frequency sets: %3i" % disp.nset)
            if disp.nset > 1:
               if disp.fit_type == 0:
                  print("Polynomial fit of the frequencies; degree: %3i " % disp.fit_degree)
               else:
                  print("Spline fit of the frequencies; degree: %3i,  smooth: %3.1f"\
                        % (disp.fit_degree, disp.fit_type)) 
            print("Number of off-centered modes: %5i" % disp.f_size)
        
        if disp.flag:
           if disp.eos_flag:
               print("\nThe phonon dispersion is used for the computation of the bulk modulus")
               print("if the bulk_dir or the bulk_modulus_p functions are used, the latter")
               print("in connection with the noeos option.")    
               if disp.fit_vt_flag:
                   print("The required V,T-fit of the free energy contribution from")
                   print("the off-centered modes is ready. Fit power: %3i" % disp.fit_vt_deg)
               else:
                   print("The required V,T-fit of the free energy contribution from")
                   print("the off-centered mode is NOT ready.")
           else:
               print("\nThe phonon dispersion correction is not used for the computation")
               print("of the bulk modulus")
            
        if lo.flag:
            out_lo=(lo.mode, lo.split)
            df_out=pd.DataFrame(out_lo, index=['Mode', 'Split'])
            df_out=df_out.T
            df_out['Mode']=np.array([int(x) for x in df_out['Mode']], dtype=object)
            print("\nFrequencies corrected for LO-TO splitting.\n")
            if verbose.flag:
               print(df_out.to_string(index=False)) 
               print("---------------------------------------------")
               
        print("\n**** Volume driver for volume_dir function ****")
        print("Delta: %3.1f; degree: %2i; left: %3.1f; right: %3.1f, Kp_fix: %s; t_max: %5.2f"\
               % (volume_ctrl.delta, volume_ctrl.degree, volume_ctrl.left, volume_ctrl.right,\
                  volume_ctrl.kp_fix, volume_ctrl.t_max))
        print("EoS shift: %3.1f; Quad_shrink: %2i; T_dump: %3.1f; Dump fact.: %2.1f, T_last %4.1f" % \
              (volume_ctrl.shift, volume_ctrl.quad_shrink, volume_ctrl.t_dump, volume_ctrl.dump,\
               volume_ctrl.t_last))
        
        if verbose.flag:
           print("\n--------- Database section ---------")    
           print("Loaded phases:")
           print(self.min_names)
            
        
class exclude_class():
    """
    Contains the list of modes to be excluded from the
    calculation of the Helmholtz free energy.
    It can be constructed by using the keyword EXCLUDE
    in the input.txt file.
    """
    def __init__(self):
        self.ex_mode=[]
        self.ex_mode_keep=[]
        self.flag=False
    def __str__(self):
        return "Excluded modes:" + str(self.ex_mode)
    def add(self,modes):
        """
        Args:
            n : can be a scalar or a list of modes to be excluded
        """
        if type(modes) is list:
           self.ex_mode.extend(modes)
           self.flag=True
        elif type(modes) is int: 
           self.ex_mode.append(modes)
           self.flag=True
        else:
            print("** Warning ** exclude.add(): invalid input type")
            return
    def restore(self):
        """
        Restores all the excluded modes
        """
        if self.flag:
           self.ex_mode_keep=self.ex_mode
        self.ex_mode=[]
        self.flag=False
    def on(self):
        self.ex_mode=self.ex_mode_keep
        self.flag=True        
                       
class fix_flag:
    def __init__(self,value=0.):
        self.value=value
        self.flag=False
    def on(self,value=4):
        self.value=value
        self.flag=True
    def off(self):
        self.value=0.
        self.flag=False
        
class fit_flag:
    def __init__(self):
        pass
    def on(self):
        self.flag=True
    def off(self):
        self.flag=False
        
class spline_flag(fit_flag):
    """
    Sets up the spline fit of the frequencies as functions of
    the volume of the unit cell.
    
    Several variables are defined:
        1. flag: (boolean); if True, frequencies are fitted with splines
        2. degree: degree of the spline
        3. smooth: *smoothness* of the spline
        4. flag_stack: (boolean) signals the presence of the spline stack
        5. pol_stack: it is the stack containing parameters for the spline fit
        
    Note:
        The spline stack can be set up and initialized by using the keyword\
        SPLINE under the keyword FITVOL in the *input.txt* file
        
    Methods:
        
    """
        
    def __init__(self,flag=False,degree=3,smooth=0):
        super().__init__()
        self.flag=False
        self.flag_stack=False
        self.degree=degree
        self.smooth=smooth
        self.pol_stack=np.array([])
    def on(self):
        super().on()
    def off(self):
        super().off()
    def set_degree(self,degree):
        self.degree=int(degree)
    def set_smooth(self,smooth):
        self.smooth=smooth
    def stack(self):
        self.pol_stack=freq_stack_spline()
        self.flag_stack=True
    def vol_range(self,v_ini, v_fin, npoint):
        self.fit_vol=np.linspace(v_ini, v_fin, npoint)
            
class poly_flag(fit_flag):
    def __init__(self,flag=False,degree=2):
        super().__init__()
        self.flag=flag
        self.flag_stack=False
        self.degree=degree
        self.pol_stack=np.array([])
    def on(self):
        super().on()
    def off(self):
        super().off() 
    def set_degree(self,degree):
        self.degree=int(degree)
    def stack(self):
        self.pol_stack=freq_stack_fit()
        self.flag_stack=True
    def vol_range(self,v_ini, v_fin, npoint):
        self.fit_vol=np.linspace(v_ini, v_fin, npoint)
        
class kieffer_class():
    def __str__(self):
        return "Application of the Kieffer model for acoustic phonons"
    def __init__(self,flag=False):
        self.flag=False
        self.stack_flag=False
        self.kief_freq=None
        self.kief_freq_inp=None
        self.t_range=None
        self.f_list=None
        self.input=False
    def stack(self, t_range, f_list):
        self.t_range=t_range
        self.f_list=f_list
    def get_value(self,temperature):
        free=scipy.interpolate.interp1d(self.t_range, self.f_list, kind='quadratic')
        return free(temperature)
    def on(self):
        self.flag=True
        print("Kieffer correction on")
        if disp.flag:
           disp.flag=False
           print("Phonon dispersion is deactivated")
        if not self.stack_flag:
           free_stack_t(pr.kt_init,pr.kt_fin,pr.kt_points) 
    def off(self):
        self.flag=False
        print("Kieffer correction off")
    def freq(self,f1,f2,f3):
        self.kief_freq_inp=np.array([f1, f2, f3])
        self.kief_freq=self.kief_freq_inp*csl*h/kb
        free_stack_t(pr.kt_init,pr.kt_fin,pr.kt_points)
    def plot(self):
        plt.figure()
        plt.plot(self.t_range, self.f_list, "k-")
        plt.xlabel("Temperature (K)")
        plt.ylabel("F free energy (J/mol apfu)")
        plt.title("Free energy from acustic modes (Kieffer model)")
        plt.show()
        
               
class bm4_class():
    """
    Set up and information for a 4^ order Birch-Murnaghan EoS (BM4)
    
    It provides:
        1.  energy:      function; Volume integrated BM4 (V-BM4)
        2.  pressure:    function; BM4        
        3.  bm4_static_eos: BM4 parameters for the static energy
            calculation as a function of V
        4.  en_ini:     initial values for the BM4 fit
        5.  bm4_store:  BM4 parameters from a fitting at a given
            temperature
                        
    methods:                    
    """
    def __init__(self):
        self.flag=False
        self.start=True
        self.energy=None
        self.pressure=None
        self.en_ini=None
        self.bm4_static_eos=None
        self.bm4_store=None
    def __str__(self):
        return "BM4 setting: " + str(self.flag)
    def on(self):
        """
        Switches on the BM4 calculation
        """
        self.flag=True
        if self.start:
           self.energy, self.pressure=bm4_def()
           self.start=False
    def estimates(self,v4,e4):
        """
        Estimates initial values of BM4 parameters for the fit
        """
        ini=init_bm4(v4,e4,4.0)
        new_ini,dum=curve_fit(v_bm3, v4, e4, \
             p0=ini,ftol=1e-15,xtol=1e-15)
        kpp=(-1/new_ini[1])*((3.-new_ini[2])*\
                  (4.-new_ini[2])+35./9.)*1e-21/conv
        self.en_ini=[new_ini[0], new_ini[1],\
                   new_ini[2], kpp, new_ini[3]]
        k0_ini=new_ini[1]*conv/1e-21
        print("\nBM4-EoS initial estimate:")
        print("\nV0:  %6.4f" % self.en_ini[0])
        print("K0:  %6.2f" % k0_ini)
        print("Kp:  %6.2f" % self.en_ini[2])
        print("Kpp: %6.2f" % self.en_ini[3])
        print("E0:   %8.6e" % self.en_ini[4])
    def store(self,bm4st):
        """
        Stores BM4 parameters from a fit a given temperature
        """        
        self.bm4_store=bm4st
    def upload(self,bm4_eos):
        """
        Loads the parameters from the static calculation
        (that are then stored in bm4_static_eos) 
        """
        self.bm4_static_eos=bm4_eos
    def upgrade(self):
        """
        Uses the stored values of parameters [from the application of 
        store()] to upgrade the initial estimation done with estimates()
        """
        self.en_ini=self.bm4_store
    def off(self):
        """
        Switches off the BM4 calculation
        """
        self.flag=False   
    def status(self):
        """
        Informs on the status of BM4 (on, or off) 
        """
        print("\nBM4 setting: %s " % self.flag)
        
class gamma_class():
    """
    Store coefficients of a gamma(T) fit
    """
    def __init__(self):
        self.flag=False
        self.degree=1
        self.pol=np.array([])
    def upload(self,deg,pcoef):
        self.flag=True
        self.degree=deg
        self.pol=pcoef    
        
class super_class():
    """
    Store supercell data: number of cells on which the frequencies
    computation was done. To be used in connection with CRYSTAL
    calculations performed with SCELPHONO and QHA keywords.
    Default value: 1
    """
    
    def __init__(self):
        self.number=1
        self.flag=False
        
    def set(self,snum):
        self.flag=True
        self.number=snum
        print("\n*** Supercell *** Number of cells: %3i" % snum)
        
    def reset(self):
        self.flag=False
        self.number=1
        print("\n*** Supercell deactivated *** Number of cells set to 1")
                
class lo_class():
    """
    LO/TO splitting correction. 
    
    The class stores a copy of the original TO frequencies, the modes
    affected by LO/TO splitting and the splitting values.
    Modes are identified by their progressive number (starting from 0) stored
    in the *mode* attribute. 
    When the correction is activated, new values of frequencies (*f_eff*) 
    are computed for the relevant modes, according to the formula:

        f_eff = 2/3 f_TO + 1/3 f_LO
        
    where f_LO = f_TO + split.
 
    Correction is activated by the keyword LO in the input.txt file,
    followed by the name of the file containing the splitting data (two
    columns: mode number and the corresponding split in cm^-1).

    Internally, the methods *on* and *off* switch respectively on and off
    the correction. The method *apply* does the computation of the frequencies 
    *f_eff*.               
    """
    def __init__(self):
        self.flag=False
        self.mode=np.array([])
        self.split=np.array([])
        self.data_freq_orig=np.array([])
        self.data_freq=np.array([])
    def on(self):
        self.apply()
        if flag_spline.flag:
            flag_spline.stack()
        elif flag_poly.flag:
            flag_poly.stack()
        self.flag=True
        print("Frequencies corrected for LO-TO splitting")
    def off(self):
        self.flag=False
        self.data_freq=np.copy(self.data_freq_orig)
        if flag_spline.flag:
            flag_spline.stack()
        elif flag_poly.flag:
            flag_poly.stack()
        print("LO-TO splitting not taken into account")
    def apply(self):
        for ifr in np.arange(lo.mode.size):
            im=lo.mode[ifr]
            for iv in int_set:
                freq_lo=self.data_freq_orig[im,iv+1]+self.split[ifr]
                self.data_freq[im,iv+1]=(2./3.)*self.data_freq_orig[im,iv+1]\
                     +(1./3.)*freq_lo

class anh_class():
    def __init__(self):
        self.flag=False
    def off(self):
        self.flag=False
        exclude.restore()
        disp.free_exclude_restore()
        print("Anharmonic correction is turned off")
        print("Warning: all the excluded modes are restored")
    def on(self):
        self.flag=True
        self.flag_brill=False
        for im, ib in zip(anharm.mode, anharm.brill):
            if ib == 0:
               exclude.add([im])
            else:
               disp.free_exclude([im])
               self.flag_brill=True
               
        if self.flag_brill:
            disp.free_fit_vt()
            
        print("Anharmonic correction is turned on")
        
class  static_class():
    """
    Defines the volume range for the fit of the static EoS
    If not specified (default) such range is defined from the
    volumes found in the static energies file.
    """
    def __init__(self):
        self.flag=False
    def set(self, vmin, vmax):
        """
        Sets the minimum and maximum volumes for the V-range
        
        Args:
            vmin: minimum volume
            vmax: maximum volume
        """
        self.vmin=vmin
        self.vmax=vmax
    def off(self):
        """
        Restores the original V-range (actually, it switches off the volume 
        selection for the fit of the static EoS)
        """
        self.flag=False 
    def on(self):
        """
        It switches on the volume selection for the fit of the static EoS
        
        Note:
            The minimum and maximum V-values are set by the 'set' method
            of the class
        """
        self.flag=True
        
class volume_control_class():
    """
    Defines suitable parameters for the volume_dir function
    """
    def __init__(self):
        self.degree=2
        self.delta=2.
        self.t_max=500.
        self.shift=0.
        self.t_dump=0.
        self.dump=1.
        self.quad_shrink=4
        self.kp_fix=False
        self.debug=False
        self.upgrade_shift=False
        self.skew=1.
        self.t_last=0.
        self.t_last_flag=False
        self.v_last=None
    def set_degree(self, degree):
        """
        Sets the degree of polynomial used to fit the (P(V)-P0)^2 data. 
        The fitted curve is the minimized to get the equilibrium volume
        at each T and P. 
        
        For each of the single parameter revelant in this class, there exist
        a specific method to set its value. The method set_all can be used to
        set the values of a number of that, at the same time, by using appropriate
        keywords as argument. The arguments to set_all are:
        
        Args:
            degree: degree of the fitting polynomial (default=2)
            delta:  volume range where the minimum of the fitting function
                    is to be searched (default=2.)
            skew:   the Volume range is centered around the equilibrium
                    volume approximated by the EoS-based new_volume function
                    The symmetry around such point can be controlled by
                    the skew parameter (default=1.: symmetric interval)
            shift:  Systematic shift from the new_volume estimation (default=0.) 
            t_max:  In the initial estimation of the volume at P/T with the EoS-based
                    new_volume function, the Kp is refined if T < t_max. 
                    If T > t_max and kp_fix=True, Kp is fixed at the value
                    refined at t_max (default=500K)
            kp_fix: See t_max (default=True)
            quad_shrink: if degree=2, it restricts the volume range around the 
                         approximated volume found. The new range is 
                         delta/quad_shrink (default=4)
            upgrade_shift: at the end of the computation, the difference between
                           the volume found and the initial one (from the EoS-
                           based new_volume function) is calculated. The shift
                           attribute is then upgraded if upgrade_shift is True
                           (default=False)
            debug:  if True, the (P(V)-P0)**2 function is plotted as a function
                    of V (default=False)   
            t_dump: temperature over which a dumping on the shift parameter is
                    applied (default=0.)
            dump: dumping on the shift parameter (shift=shift/dump; default=1.)
            t_last: if t_last > 10., the last volume computed is used as the 
                    initial guess value (vini) for the next computation at a
                    new temperature.       
        """
        self.degree=degree
    def set_delta(self, delta):
        self.delta=delta
    def set_tmax(self,tmax):
        self.t_max=tmax
    def set_skew(self, skew):
        self.left=skew+1
        self.right=(skew+1)/skew
    def kp_on(self):
        self.kp_fix=True
    def kp_off(self):
        self.kp_fix=False
    def debug_on(self):
        self.debug=True
    def debug_off(self):
        self.debug=False
    def set_shift(self, shift):
        self.shift=shift
    def upgrade_shift_on(self):
        self.upgrade_shift=True
    def upgrade_shift_off(self):
        self.ugrade_shift=False
    def set_shrink(self, shrink):
        self.quad_shrink=shrink   
    def shift_reset(self):
        self.shift=0.    
    def set_t_dump(self,t_dump=0., dump=1.0):
        self.t_dump=t_dump
        self.dump=dump  
    def set_t_last(self, t_last):
        self.t_last=t_last
    def set_all(self,degree=2, delta=2., skew=1., shift=0., t_max=500.,\
                quad_shrink=4, kp_fix=True, upgrade_shift=False, debug=False,\
                t_dump=0., dump=1., t_last=0.):
        
        self.degree=degree
        self.delta=delta
        self.t_max=t_max
        self.kp_fix=kp_fix
        self.debug=debug
        self.left=skew+1
        self.right=(skew+1)/skew        
        self.shift=shift
        self.quad_shrink=quad_shrink
        self.upgrade_shift=upgrade_shift
        self.skew=skew
        self.t_last=t_last
        
class disp_class():
    """
    Sets up the computation for the inclusion of phonon dispersion effects
    the EoS computation or for the calculation of all the thermodynamic
    properties. 
    
    The class is relevant and activated if the DISP keyword is contained
    in the input.txt input file.
    
    Dispersion effects can be switched on or off by using the on() and off()
    methods.
    
    Note:
        To apply the phonon dispersion correction to computation of an equation
        of state, the method eos_on() must be invoked [the method eos_off() switches
        it off].   

    Note:
        The method free_fit_vt() must be used to get the F(V,T) function for
        off-center phonon modes.                                             
    """
    def __init__(self):
        self.flag=False
        self.eos_flag=False
        self.freq=None
        self.deg=None    
        self.fit_type=None
        self.input=False
        self.fit_vt_flag=False
        self.fit_vt=None
        self.temp=None
        self.error_flag=False
        self.ex_flag=False
        self.free_min_t=10.
        self.fit_vt_deg=4
        self.free_nt=24
        self.free_disp=True
    def on(self):
        self.flag=True
        print("Dispersion correction activated")
        if kieffer.flag:
           kieffer.flag=False
           print("Kieffer correction is deactivated")
    def off(self):
        self.flag=False
        print("Dispersion correction off")
    def eos_on(self):
        if self.flag :
           if not self.error_flag:
              self.eos_flag=True
              print("Phonon dispersion correction for bulk_dir computation")
           else:
              print("Only 1 volume found in the 'disp' files; NO disp_eos possible")
        else:
           print("Phonon dispersion is not on; use disp.on() to activate")
    def eos_off(self):
        self.eos_flag=False
        print("No phonon dispersion correction for bulk_dir computation")
        
    def freq_spline_fit(self):
        """
        It requests and makes spline fits of the frequencies of the off
        center modes as function of volumes. 
        
        Relevant parameters for the fit (degree and smooth parameters) are 
        specified in the appropriate input file. 
        """
 
        self.spline=np.array([])
        ord_vol=list(np.argsort(self.vol))
        vol = [self.vol[iv] for iv in ord_vol]        
        
        for ifr in np.arange(self.f_size):
            freq=self.freq[:,ifr]
            freq=[freq[iv] for iv in ord_vol]
            ifit=UnivariateSpline(vol, freq, k=self.fit_degree, s=self.fit_type)
            self.spline=np.append(self.spline, ifit)
        
    def freq_fit(self):
        """
        It requests and makes polynomial fits of the frequencies of the off
        center modes as function of volumes. 
        
        The relevant parameter for the fit (degree) is specified in the  
        appropriate input file. 
        """        
        self.poly=np.array([])
        
        for ifr in np.arange(self.f_size):
             if self.nset > 1:
               freq=self.freq[:,ifr]
               ifit=np.polyfit(self.vol, freq, self.fit_degree)
               self.poly=np.append(self.poly,ifit)               
             else:
               self.poly=np.append(self.poly, (0, self.freq[:,ifr][0]))
        
        if self.nset == 1:
           self.poly=self.poly.reshape(self.f_size,2)
        else:
           self.poly=self.poly.reshape(self.f_size,self.fit_degree+1) 
           
    def freq_func(self,ifr,vv):     
        fit=self.poly[ifr]
        return np.polyval(fit,vv)
    
    def freq_spline_func(self,ifr,vv):
        fit=self.spline[ifr](vv)
        return fit.item(0)
    
    def check(self,ifr):
        """
        Check of the frequencies fit quality for a specified mode
       
        Args:
            ifr: sequence number of the mode to be checked
        """
        
        v_list=np.linspace(np.min(disp.vol), np.max(disp.vol),40)
        
        if self.fit_type == 0:
          f_list=[self.freq_func(ifr,iv) for iv in v_list]  
        else:
          f_list=[self.freq_spline_func(ifr,iv) for iv in v_list]
        
        tlt="Check fit for mode N. "+ str(ifr)
        plt.figure()
        plt.plot(v_list,f_list, "k-")
        plt.plot(disp.vol, disp.freq[:,ifr],"b*")
        plt.xlabel("Volume (A^3)")
        plt.ylabel("Frequency (cm^-1)")
        plt.title(tlt)
        plt.show()
        
    def check_multi(self, fr_l):
        """
        Check of the frequencies fit quality for a list of modes
        
        Args:
            fr_l: list of sequence numbers of the various modes to be checked
        
        Example:
            >>> disp.check_multi([0, 1, 2, 3])
            >>> disp.check_multi(np.arange(10))
        """
        for ifr in fr_l:
            self.check(ifr)
            
    def free_exclude(self,ex_list):
        """
        Excludes the indicated off-center modes from the computation of the 
        free energy
        
        Args:
            ex_list: list of modes to be excluded
        
        Note:
            Even a single excluded mode must be specified as a list; for instance
            disp.free_exclude([0])
            
        Note: 
            after the exclusion of some modes, the F(V,T) function has
            to be recomputed by the free_fit_vt method
        """
      
        self.ex_flag=True
        self.excluded_list=ex_list
        print("Off center modes excluded: ", self.excluded_list)
        print("Compute a new disp.free_fit_vt surface")
        
    def free_exclude_restore(self):
        """
        The excluded modes are restored
        """       
        self.ex_flag=False   
        print("All off centered mode restored")
        print("Compute a new disp.free_fit_vt surface")
        
    def free(self,temp,vv):
        
        nf_list=np.arange(self.f_size)
        
        if self.fit_type == 0:
           freq=(self.freq_func(ifr,vv) for ifr in nf_list)
        else:
           freq=(self.freq_spline_func(ifr,vv) for ifr in nf_list) 
        d_deg=self.deg
        wgh=self.w_list
        
        enz=0.
        fth=0.
        idx=0
        nfreq=0
        for ifr in freq:
            if not self.ex_flag:
               nfreq=nfreq+1
               fth=fth+d_deg[idx]*np.log(1-np.e**(ifr*e_fact/temp))*wgh[idx]
               enz=enz+d_deg[idx]*ifr*ez_fact*wgh[idx]
            else:
               if not (idx in self.excluded_list):
                  nfreq=nfreq+1
                  fth=fth+d_deg[idx]*np.log(1-np.e**(ifr*e_fact/temp))*wgh[idx]
                  enz=enz+d_deg[idx]*ifr*ez_fact*wgh[idx]
            idx=idx+1 
          
        return enz+fth*kb*temp/conv
    
    def free_fit(self,mxt,vv,disp=True):
        
        fit_deg=4
        nt=16
        nt_plot=50
        tl=np.linspace(10,mxt,nt)
        
        free=np.array([])
        
        for it in tl:
            ifree=self.free(it,vv)
            free=np.append(free,ifree)
            
        fit=np.polyfit(tl,free,fit_deg)
        self.fit=fit
        
        if disp:
           tl_plot=np.linspace(10,mxt,nt_plot)
           free_plot=self.free_func(tl_plot)
        
           print("Phonon dispersion correction activated by the DISP keyword")
           print("in the input file; the contribution to the entropy and to the")
           print("specific heat is taken into account.\n")
             
           if verbose.flag:
              plt.figure()
              plt.plot(tl,free,"b*",label="Actual values")
              plt.plot(tl_plot, free_plot,"k-",label="Fitted curve")
              plt.legend(frameon=False)
              plt.xlabel("T (K)")
              plt.ylabel("F (a.u.)")
              plt.title("Helmholtz free energy from off-centered modes")
              plt.show()    
  
    def free_fit_ctrl(self, min_t=10., degree=4, nt=24, disp=True):
        """
        Free fit driver: sets the relevant parameters for the fit computation
        of the F(V,T) function, on the values of F calculated on a grid
        of V and T points.
        
        Args:
            min_t: minimum temperature for the construction of the 
                   VT grid (default=10.)
            degree: degree of the surface (default=4)
            nt: number of points along the T axis for the definition of the 
                (default=24) grid
            disp: it True, a plot of the surface is shown (default=True)
            
        Note:
            The method does not execute the fit, but it defines the most
            important parameters. The fit is done by the free_fit_vt() method.
            
        Note: 
            the volumes used for the construction of the VT grid are those
            provided in the appropriate input file. They are available
            in the disp.vol variable.
        """    
        self.free_min_t=min_t
        self.fit_vt_deg=degree
        self.free_nt=nt
        self.free_disp=disp
        
        self.free_fit_vt()
        
    def set_tmin(self,tmin):
        self.min_t=tmin
        
    def set_nt(self,nt):
        self.nt=nt   
        
    def free_fit_vt(self):
        
        self.fit_vt_flag=True
        min_t=self.free_min_t
        nt=self.free_nt
        disp=self.free_disp
        deg=self.fit_vt_deg
        
        max_t=self.temp
        
        pl=np.arange(deg+1)
        p_list=np.array([],dtype=int)
    
        for ip1 in pl:
          for ip2 in pl:
            i1=ip2
            i2=ip1-ip2
            if i2 < 0:
                break
            ic=(i1, i2)
            p_list=np.append(p_list,ic)
    
        psize=p_list.size
        pterm=int(psize/2)
    
        self.p_list=p_list.reshape(pterm,2)
        
        x0=np.ones(pterm)
        
        t_list=np.linspace(min_t,max_t,nt)
        v_list=self.vol
        nv=len(v_list)
        
        if nv == 1:
           print("\n**** WARNING ****\nOnly one volume found in the 'disp' data files;") 
           print("NO V,T-fit of F is possible")
           self.eos_off()
           self.error_flag=True
           return
        
        free_val=np.array([])
        for it in t_list:
            for iv in v_list:
                ifree=self.free(it,iv)
                free_val=np.append(free_val,ifree)
                
        free_val=free_val.reshape(nt,nv)
        
        vl,tl=np.meshgrid(v_list,t_list)
        vl=vl.flatten()
        tl=tl.flatten()
        free_val=free_val.flatten()
    
        fit, pcov = curve_fit(self.free_vt_func, [vl, tl], free_val, p0 = x0)
        
        self.fit_vt=fit
        
        error=np.array([])
        for it in t_list:
            for iv in v_list:
                f_calc=self.free_vt(it,iv)
                f_obs=self.free(it,iv)
                ierr=(f_calc-f_obs)**2
                error=np.append(error,ierr)
                
        mean_error=np.sqrt(np.mean(error))
        max_error=np.sqrt(np.max(error))
        print("V,T-fit of the Helmholtz free energy contribution from the off-centered modes")
        print("Power of the fit: %3i" % self.fit_vt_deg)
        print("Mean error: %5.2e" % mean_error)
        print("Maximum error: %5.2e" % max_error)
        
        if self.ex_flag:
            print("Excluded modes: ", self.excluded_list)
        
        if disp:
            t_plot=np.linspace(min_t,max_t,40)
            v_plot=np.linspace(np.min(vl),np.max(vl),40)
        
            v_plot,t_plot=np.meshgrid(v_plot,t_plot)
            v_plot=v_plot.flatten()
            t_plot=t_plot.flatten()
            h_plot=self.free_vt_func([v_plot, t_plot], *fit)
        
            h_plot=h_plot.reshape(40,40)
            v_plot=v_plot.reshape(40,40)
            t_plot=t_plot.reshape(40,40)
        
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111,projection='3d', )
        
            ax.scatter(tl,vl,free_val,c='r')
            ax.plot_surface(t_plot, v_plot, h_plot)
            ax.set_xlabel("Temperature", labelpad=7)
            ax.set_ylabel("Volume", labelpad=7)
            ax.set_zlabel('F(T,V)', labelpad=8)
            plt.show()
        
    def free_vt_func(self,data,*par):
            
        vv=data[0]
        tt=data[1]

        nterm=self.p_list.shape[0]
        func=0.   
        for it in np.arange(nterm):
            pv=self.p_list[it][0]
            pt=self.p_list[it][1]
            func=func+par[it]*(vv**pv)*(tt**pt)
            
        return func  
    
    def free_vt(self,temp,volume):
        return self.free_vt_func([volume,temp],*self.fit_vt)
        
    def free_func(self,temp):
        free_disp=np.polyval(self.fit,temp)
        return free_disp

# reads in data file. It requires a pathname to the folder
# containing data

def read_file(data_path):
   global volume, energy, deg, data_vol_freq, num_set_freq
   global num_mode, ini, int_set, int_mode, data_vol_freq_orig
   global temperature_list, popt, pcov, data_freq, path, data_file
   global data, zu, apfu, power, lpow, power_a, lpow_a, mass
   global flag_eos, flag_cp, flag_alpha, flag_err, flag_exp, flag_mass
   global data_cp_exp

   flag_eos=False
   flag_cp=False
   flag_alpha=False
   flag_err=False
   flag_exp=False
   flag_fit=False
   flag_mass=False
   flag_super=False
    
   flag_static, flag_volume, flag_freq, flag_ini, flag_fu, flag_set \
                 = False, False, False, False, False, False
   path=data_path
   input_file=data_path+'/'+'input.txt'
   line_limit=100
   with open(input_file) as fi:
      jc=0
      l0=['']
      while (l0 !='END') and (jc < line_limit): 
         str=fi.readline()
         lstr=str.split()
         l0=''
         if lstr !=[]:
            l0=lstr[0].rstrip()
         if l0 !='#':
            if l0=='STATIC':
               data_file=data_path+'/'+fi.readline()
               data_file=data_file.rstrip()
               flag_static=os.path.isfile(data_file)
            elif l0=='VOLUME': 
               data_file_vol_freq=data_path+'/'+fi.readline()
               data_file_vol_freq=data_file_vol_freq.rstrip()
               flag_volume=os.path.isfile(data_file_vol_freq)
            elif l0=='FREQ':
               data_file_freq=data_path+'/'+fi.readline()
               data_file_freq=data_file_freq.rstrip()
               flag_freq=os.path.isfile(data_file_freq)
            elif l0=='EXP':
                data_file_exp=data_path+'/'+fi.readline()
                data_file_exp=data_file_exp.rstrip()
                flag_exp=os.path.isfile(data_file_exp)
            elif l0=='LO':
                lo_freq_file=data_path+'/'+fi.readline()
                lo_freq_file=lo_freq_file.rstrip()
                lo.flag=True
            elif l0=='FITVOL':
                fit_type=fi.readline()
                fit_vol=fi.readline()
                flag_fit=True
            elif l0=='FU':
               zu=fi.readline()
               flag_fu=True
            elif l0=='MASS':
                mass=fi.readline()
                flag_mass=True
            elif l0=='SET':
                istr=fi.readline()
                while istr.split()[0] =='#':
                    istr=fi.readline()
                int_set=istr    
                flag_set=True
            elif l0=='TEMP':
               temperature_list=fi.readline()
               flag_eos=True
            elif l0=='TITLE':
                title=fi.readline().rstrip()
                info.title=title
            elif l0=='INI':               
               ini=fi.readline()
               flag_ini=True
            elif l0=='CP':
               power=fi.readline()
               flag_cp=True
            elif l0=='ALPHA':
               power_a=fi.readline()
               flag_alpha=True
            elif l0=='EXCLUDE':
               exclude.restore()
               ex_mode=fi.readline()
               ex_mode=list(map(int, ex_mode.split()))
               exclude.add(ex_mode)
            elif l0=='KIEFFER':
               kieffer.input=True
               kieffer.flag=True
               kief_freq=fi.readline()
               kief_freq_inp=list(map(float, kief_freq.split()))
               kief_freq=np.array(kief_freq_inp)*csl*h/kb
               kieffer.kief_freq=kief_freq 
               kieffer.kief_freq_inp=kief_freq_inp
            elif l0=='ANH':
               anharm.nmode=int(fi.readline().rstrip())
               anharm.mode=np.array([],dtype=int)
               anharm.wgt=np.array([],dtype=int)
               anharm.brill=np.array([],dtype=int)
               for im in np.arange(anharm.nmode):
                   line=fi.readline().rstrip()
                   mw=list(map(int, line.split()))                
                   mode=int(mw[0])
                   brill=int(mw[1])
                   wgt=int(mw[2])
                   anharm.mode=np.append(anharm.mode, mode)
                   anharm.wgt=np.append(anharm.wgt, wgt) 
                   anharm.brill=np.append(anharm.brill, brill)
               anharm.flag=True
            elif l0=='SUPER':
                line=fi.readline().rstrip()
                line_val=list(map(int, line.split()))
                snum=line_val[0]
                static_vol=line_val[1]
                flag_static_vol=False
                if static_vol == 0:
                    flag_static_vol=True                   
                flag_super=True
            elif l0=='DISP':
                disp.flag=True
                disp.input=True
                disp_file=data_path+'/'+fi.readline()
                disp_info=data_path+'/'+fi.readline()
                disp_file=disp_file.rstrip()
                disp_info=disp_info.rstrip()
                
                fd=open(disp_info)
                
                line=fd.readline().rstrip().split()
                disp.molt=int(line[0])
                disp.fit_degree=int(line[1])
                disp.fit_type=float(line[2])
                disp.temp=float(line[3])
                
                line=fd.readline().rstrip().split()
                disp.numf=list(map(int, line))
                
                line=fd.readline().rstrip().split()
                disp.wgh=list(map(int, line))
                
                line=fd.readline().rstrip().split()
                disp.vol=list(map(float, line))
                
                fd.close()
                
                w_list=np.array([],dtype=int)
                for iw in np.arange(disp.molt):
                    wl=np.repeat(disp.wgh[iw],disp.numf[iw])
                    w_list=np.append(w_list,wl)
                disp.w_list=w_list
                
                disp.f_size=disp.w_list.size
                
         jc=jc+1
   if jc>=line_limit:
       print("\nWarning: END keyword not found")
   if not flag_volume or not flag_freq or not flag_static:
       print("\nError: one or more data file not found, or not assigned"
             " in input")
       flag_err=True
       return
   if not flag_fu:
       print("\nError: mandatory FU keyword not found")
       flag_err=True
       return
   if not flag_set:
       print("\nError: mandatory SET keyword not found")
       flag_err=True
       return
   
   fi.close()
   if flag_view_input.value:
      view_input(input_file)
      print("\n-------- End of input file -------\n")
      flag_view_input.off()
   
   int_set=int_set.rstrip()
   int_set=list(map(int, int_set.split()))
   info.freq_sets=int_set
   
   if flag_eos:
      temperature_list=temperature_list.rstrip()
      temperature_list=list(map(float,temperature_list.split()))
   if flag_ini:
      ini=ini.rstrip()
      ini=list(map(float, ini.split()))
      ini[1]=ini[1]*1e-21/conv
   zus=list(map(int,zu.rstrip().split()))
   zu=zus[0]
   apfu=zus[1]
   if flag_fit:
      fit_type=fit_type.rstrip()
      fit_vol=fit_vol.rstrip()
      fit_vol=list(map(float, fit_vol.split()))
      v_ini=fit_vol[0]
      v_fin=fit_vol[1]
      nv=int(fit_vol[2])        
      if fit_type=='SPLINE': 
         flag_spline.on()
         flag_spline.set_degree(fit_vol[3])
         flag_spline.set_smooth(fit_vol[4])
         flag_spline.vol_range(v_ini, v_fin, nv)
         info.fit_type='spline'
         info.fit_degree=flag_spline.degree
         info.fit_smooth=flag_spline.smooth
         info.min_vol_fit=v_ini
         info.max_vol_fit=v_fin
         info.fit_points=nv
      elif fit_type=='POLY':
         flag_poly.on()
         flag_poly.set_degree(fit_vol[3])
         flag_poly.vol_range(v_ini, v_fin, nv)
         info.fit_type='poly'
         info.fit_degree=flag_poly.degree
         info.min_vol_fit=v_ini
         info.max_vol_fit=v_fin
         info.fit_points=nv
         
   if flag_super:
      supercell.set(snum)
         
   if flag_cp:
      power=power.rstrip()
      power=list(map(float, power.split()))
      lpow=len(power)
      test_cp=[ipw in cp_power_list for ipw in power]        
      if not all(test_cp): 
             print("WARNING: the power list for the Cp fit is not consistent")
             print("         with the Perplex database")
             print("Allowed powers:", cp_power_list)
             print("Given powers:", power)
             print("")
      
   if flag_alpha:
      power_a=power_a.rstrip()
      power_a=list(map(float, power_a.split()))
      lpow_a=len(power_a)
      test_al=[ipw in al_power_list for ipw in power_a]
      if not all(test_al):
          print("WARNING: the power list for the alpha fit is not consistent")
          print("         with the Perplex database")
          print("Allowed powers:", al_power_list)
          print("Given powers:", power_a)
          print("")
      
   if flag_mass:
       mass=float(mass.rstrip())
   
   b_flag=False
   if anharm.flag:
       anharm_setup()
       
       for im,ib in zip(anharm.mode, anharm.brill):
           if ib == 0:
              exclude.add([im])
           else:
              disp.free_exclude([im])
              b_flag=True
                         
   if disp.flag:
       disp.freq=np.array([])
       disp_data=np.loadtxt(disp_file)
       disp.deg=disp_data[:,0]
       nset=len(disp.vol)
       disp.nset=nset
       for iv in np.arange(nset):
           disp.freq=np.append(disp.freq, disp_data[:,iv+1])
           
       disp.freq=disp.freq.reshape(nset,disp.f_size)
       
       if disp.fit_type == 0:
          disp.freq_fit()
       else:
          disp.freq_spline_fit()
       disp.free_fit(disp.temp,disp.vol[0])
       
   data=np.loadtxt(data_file)
   data_vol_freq_orig=np.loadtxt(data_file_vol_freq)
       
   lo.data_freq=np.loadtxt(data_file_freq)
   lo.data_freq_orig=np.copy(lo.data_freq)
   
   info.min_freq_vol=min(data_vol_freq_orig)
   info.max_freq_vol=max(data_vol_freq_orig)
   info.freq_points=len(data_vol_freq_orig)
   
   if flag_exp:
       data_cp_exp=np.loadtxt(data_file_exp)

   volume=data[:,0]
   energy=data[:,1]
   if flag_super:
      if flag_static_vol:
         volume=volume*snum
         energy=energy*snum
    
   info.min_static_vol=min(volume)
   info.max_static_vol=max(volume)
   info.static_points=len(volume)
    
   deg=lo.data_freq[:,0]
   num_set_freq=lo.data_freq.shape[1]-1
   num_mode=lo.data_freq.shape[0]-1
   int_mode=np.arange(num_mode+1)
   if flag_super:
       deg=deg/supercell.number
   
   if not flag_ini:
       ini=init_bm3(volume,energy)
   data_vol_freq=[]
   for iv in int_set:
      data_vol_freq=np.append(data_vol_freq, data_vol_freq_orig[iv])
      
   int_set_new=np.array([],dtype='int32')
   ind=data_vol_freq.argsort()
   for ind_i in ind:
       int_set_new=np.append(int_set_new, int_set[ind_i])
   if not np.array_equal(int_set, int_set_new):
       print("\nWarning ** Volume and frequencies lists have been sorted")
       print("           indexing: ", ind)
       print("")
   int_set=int_set_new
   data_vol_freq.sort()
      
   info.min_select_vol=min(data_vol_freq)
   info.max_select_vol=max(data_vol_freq)
   info.select_points=len(data_vol_freq)
   
   volume_ctrl.set_all()
      
   if flag_fit:
       if flag_spline.flag:
          flag_spline.stack()
       elif flag_poly.flag:
          flag_poly.stack()
          
   if lo.flag:
       lo_data=np.loadtxt(lo_freq_file)
       lo.mode=lo_data[:,0].astype(int)
       lo.split=lo_data[:,1].astype(float)
       lo.on()
       
   if disp.input and kieffer.input:
       kieffer.flag=False
       print("\nBoth Kieffer and phonon dispersion data were found in the input file")
       print("The Kieffer model is therefore deactivated")
       
   if b_flag:
       print("")
       disp.free_fit_vt()
          
def view():
    """
    View input file (input.txt)
    """
    input_file=path+"/input.txt"
    view_input(input_file)
    
def view_input(input_file):
    line_limit=1000
    print("\nInput file\n")
    with open(input_file) as fi:
      jc=0
      l0=['']
      while (l0 !='END') and (jc < line_limit):
         str=fi.readline()
         lstr=str.split()
         if lstr !=[]:
            l0=lstr[0].rstrip()
         if l0 !='#':
            print(str.rstrip())
            jc=jc+1
               
def reload_input(path):
    
    reset_flag()
    read_file(path)
    static()

def set_fix(fix=4.):
    """
    Sets Kp to a value and keeps it fixed during fitting of EoS
    
    Args:
        fix (optional): Kp value. Default 4.
        if fix=0, Kp if fixed to the last computed value stored in info.kp
        
    The flag f_fit.flag is set to True
    """
    if fix == 0:
        fix=info.kp
        
    f_fix.on(fix)

def reset_fix():
    """
    Resets the fix Kp option: f_fit.flag=False
    """
    f_fix.off()
    
def fix_status():
    """
    Inquires about the setting concerning Kp
    """
    print("Fix status: %r" % f_fix.flag)
    if f_fix.flag:
        print("Kp fixed at %4.2f" % f_fix.value )
        
def set_spline(degree=3,smooth=5, npoint=16):
    """
    Sets spline fits of the frequencies as function of volume
    
    Args:
        degree (optional): degree of the spline (default: 3)
        smooth (optional): smoothness of the spline (default: 5)
        npoint (optional): number of points of the spline function
                           (default: 16)
    """
    dv=0.2
    flag_spline.on()
    flag_poly.off()
    flag_spline.set_degree(degree)
    flag_spline.set_smooth(smooth)
    fit_vol_exists=True
    try: 
       flag_spline.fit_vol      
    except AttributeError: 
       fit_vol_exists=False
    if not fit_vol_exists:  
        set_volume_range(min(data_vol_freq)-dv,max(data_vol_freq)+dv,npoint,\
                         prt=True)
    else:
        set_volume_range(min(flag_spline.fit_vol),max(flag_spline.fit_vol),npoint)
    flag_spline.stack()  
    
    info.fit_type='spline'
    info.fit_degree=degree
    info.fit_smooth=smooth
    info.fit_points=npoint
    info.min_vol_fit=min(flag_spline.fit_vol)
    info.max_vol_fit=max(flag_spline.fit_vol)    
        
def set_poly(degree=4,npoint=16):
    """
    Sets polynomial fits of the frequencies as function of volume
    
    Args:
        degree (optional): degree of the spline (default: 4)
        npoint (optional): number of points of the polynomial function
                           (default: 16)
    """
    dv=0.2
    flag_poly.on()
    flag_spline.off()
    flag_poly.set_degree(degree)
    fit_vol_exists=True
    try:
       flag_poly.fit_vol
    except AttributeError: 
       fit_vol_exists=False  
    if not fit_vol_exists:  
        set_volume_range(min(data_vol_freq)-dv,max(data_vol_freq)+dv,npoint, \
                         prt=True)
    else:
        set_volume_range(min(flag_poly.fit_vol),max(flag_poly.fit_vol),npoint) 
    flag_poly.stack()
    
    info.fit_type='poly'
    info.fit_degree=degree
    info.fit_points=npoint
    info.min_vol_fit=min(flag_poly.fit_vol)
    info.max_vol_fit=max(flag_poly.fit_vol)
       
def set_volume_range(vini,vfin,npoint=16,prt=False):
    """
    Defines a volume range for the fitting of frequencies and EoS
    in the case that SPLINE or POLY fits have been chosen
    
    Args:
        vini: minimum volume
        vfin: maximum volume
        npoint (optional): number of points in the volume range
    """
    
    if flag_poly.flag:
       flag_poly.vol_range(vini,vfin,npoint)
       flag_poly.stack()
       info.fit_points=npoint
       info.min_vol_fit=min(flag_poly.fit_vol)
       info.max_vol_fit=max(flag_poly.fit_vol)
       if prt:
          print("Volume range %8.4f - %8.4f defined for 'POLY' fit" %\
             (vini, vfin))
    elif flag_spline.flag:      
       flag_spline.vol_range(vini,vfin,npoint)
       flag_spline.stack()
       info.fit_points=npoint
       info.min_vol_fit=min(flag_spline.fit_vol)
       info.max_vol_fit=max(flag_spline.fit_vol)
       if prt: 
          print("Volume range %8.4f - %8.4f defined for 'SPLINE' fit" %\
             (vini, vfin))
    else:
       print("No fit of frequencies active\nUse set_poly or set_spline\n")
       
       
          
def fit_status():
    if flag_poly.flag or flag_spline.flag:
       print("Fit of frequencies is active")
       if flag_spline.flag:
          print("Spline fit: degree %2d, smooth: %3.1f" \
                  % (flag_spline.degree, flag_spline.smooth))
          print("Volume range: %5.2f - %5.2f, points=%d" % \
            (min(flag_spline.fit_vol), max(flag_spline.fit_vol), \
             flag_spline.fit_vol.size))
       else:
          print("Polynomial fit: degree %2d"  % flag_poly.degree)
          print("Volume range: %5.2f - %5.2f, points=%d" % \
              (min(flag_poly.fit_vol), max(flag_poly.fit_vol), \
               flag_poly.fit_vol.size))
    else:
       print("Fitting is off") 
        
def fit_off():
    flag_poly.off()
    flag_spline.off()
    info.fit_type='No fit'

   
def quick_start(path):
    """
    Quick start of the program.
    Reads the input files found under the folder 'path'
    whose name is written in the 'quick_start.txt' file
    (found in the master folder).
    Executes read_file; static (static equation of state)
    and stacks data for the application of the Kieffer model,
    if required with the optional 'KIEFFER' keyword in input.txt
    """
    read_file(path)
    static(plot=False)
    if kieffer.flag:
        free_stack_t(pr.kt_init, pr.kt_fin, pr.kt_points)
        if verbose.flag:
           print("Results from the Kieffer model for acoustic branches:")
           print("plot of the Helmholtz free energy as a function of T.")
           print("Temperature limits and number of points defined in parame.py")
           kieffer.plot()
        else:
           print("Kieffer model for the acoustic branches activated")
            

def v_bm3(vv,v0,k0,kp,c):
    """
    Volume integrated Birch-Murnaghan equation (3^rd order)
    
    Args:
        vv: volume 
        v0: volume at the minimum of the energy
        k0: bulk modulus
        kp: derivative of k0 with respect to P
        c:  energy at the minimum
        
    Returns:
        the energy at the volume vv
    """
    v0v=(np.abs(v0/vv))**(2/3)
    f1=kp*(np.power((v0v-1.),3))
    f2=np.power((v0v-1.),2)
    f3=6.-4*v0v
    return c+(9.*v0*k0/16.)*(f1+f2*f3)


def bm3(vv,v0,k0,kp):
    """
    Birch-Murnaghan equation (3^rd order)
    
    Args:
        vv: volume 
        v0: volume at the minimum of the energy
        k0: bulk modulus
        kp: derivative of k0 with respect to P
        
    Returns:
        the pressure at the volume vv    
    """
    v0v7=np.abs((v0/vv))**(7/3)
    v0v5=np.abs((v0/vv))**(5/3)
    v0v2=np.abs((v0/vv))**(2/3)
    f1=v0v7-v0v5
    f2=(3/4)*(kp-4)*(v0v2-1)
    return (3*k0/2)*f1*(1+f2)
 
def bmx_tem(tt,**kwargs):
    """
    V-BMx (volume integrated) fit at the selected temperature
    
    Args:
        tt: temperature
        
    Keyword Args:
          fix: if fix > 0.1, kp is fixed to the value 'fix'
               during the optimization of the EoS.
               (this is a valid option only for the BM3 fit,
               but it is ignored for a BM4 EoS)
             
    Returns:              
           1. free energy values at the volumes used for the fit
           2. optimized v0, k0, kp, (kpp), and c
           3. covariance matrix
             
    Note: 
        bmx_tem optimizes the EoS according to several 
        possible options specified elsewhere:            
           1. kp fixed or free
           2. frequencies not fitted, or fitted by
              polynomials or splines
           3. 3^rd or 4^th order BM EoS          
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    flag_x=False
    volb=data_vol_freq
    if flag_poly.flag:
       volb=flag_poly.fit_vol
    elif flag_spline.flag:    
       volb=flag_spline.fit_vol
    
    if f_fix.flag:
       fix=f_fix.value
       flag_x=True
       p0_f=[ini[0],ini[1],ini[3]]
       
    if fixpar:
       if fix_value < 0.1:
          flag_x=False
       else:
          fix=fix_value
          flag_x=True
          p0_f=[ini[0],ini[1],ini[3]]
        
    if flag_poly.flag or flag_spline.flag:
        free_energy=free_fit(tt)
    else:
        free_energy=free(tt) 
    if (flag_x) and (not bm4.flag):
           pterm, pcov_term = curve_fit(lambda volb, v0, k0, c: \
                        v_bm3(volb, v0, k0, fix, c), \
                        volb, free_energy, p0=p0_f, \
                        ftol=1e-15, xtol=1e-15)
           pterm=np.append(pterm,pterm[2])
           pterm[2]=fix
    else:   
        if bm4.flag:
           if f_fix.flag: 
               reset_fix()
               fix_status()
           with warnings.catch_warnings():
              warnings.simplefilter("ignore")
              pterm, pcov_term= curve_fit(bm4.energy, volb, free_energy,\
                method='dogbox',p0=bm4.en_ini, ftol=1e-18, xtol=3.e-16,gtol=1e-18)
           bm4.store(pterm)
        else:
                pterm, pcov_term = curve_fit(v_bm3, volb, free_energy, \
                                     p0=ini, ftol=1e-15, xtol=1e-15)
            
    return [free_energy, pterm, pcov_term]

def stop():
   """
   used to exit from the program in case of fatal exceptions
   """
    
   while True:
        print("Program will be terminated due to errors in processing data")
        answ=input('Press enter to quit')
        sys.exit(1)
          

def bm4_def():
   V0=sym.Symbol('V0',real=True,positive=True)
   V=sym.Symbol('V',real=True,positive=True)
   f=sym.Symbol('f',real=True)
   kp=sym.Symbol('kp',real=True)
   ks=sym.Symbol('ks',real=True)
   k0=sym.Symbol('k0',real=True)   
   P=sym.Symbol('P',real=True,positive=True)
   E0=sym.Symbol('E0',real=True)
   c=sym.Symbol('c',real=True)
   

   f=((V0/V)**sym.Rational(2,3)-1)/2
   P=3*k0*f*((1+2*f)**sym.Rational(5,2))*(1+sym.Rational(3,2)*(kp-4.)*f +\
             sym.Rational(3,2)*(k0*ks+(kp-4.)*(kp-3.)+sym.Rational(35,9))*(f**2))
   E=sym.integrate(P,V)
   E0=E.subs(V,V0)
   E=E0-E+c

   bm4_energy=sym.lambdify((V,V0,k0,kp,ks,c),E,'numpy')
   bm4_pressure=sym.lambdify((V,V0,k0,kp,ks),P,'numpy')
   
   return bm4_energy, bm4_pressure

def init_bm4(vv,en,kp):
    
    """
    Function used to estimate the initial parameters of a V-integrated BM4 
    EoS. The function is used by the method "estimates" of the bm4 class.
    The estimation is done on the basis of a previous BM3 optimization 
    whose initial parameters are provided by the current function.
    
    Args:
        vv (list): volumes
        en (list): static energies at the corresponding volumes vv
        kp:initail value assigned to kp
        
    Returns:
        "ini" list of V-integrated EoS parameters (for a BM3) estimated by a 
        polynomial fit: v_ini, k0_ini, kp, e0_ini. 
        
    Note: such parameters are used as initial guesses for the BM3 optimization
    performed by the method "estimates" of the class bm4 that, in turn, 
    outputs the "ini" list for the BM4 EoS optimization. 
    """
    
    pol=np.polyfit(vv,en,4)
    pder1=np.polyder(pol,1)
    pder2=np.polyder(pol,2)
    v_r=np.roots(pder1)
    vs=v_r*np.conj(v_r)
    min_r=np.argmin(vs)
    v_ini=np.real(v_r[min_r])
    e0_ini=np.polyval(pol, v_ini)
    k0_ini=np.polyval(pder2, v_ini)
    k0_ini=k0_ini*v_ini
    ini=[v_ini, k0_ini, kp, e0_ini]
    return ini

def init_bm3(vv,en):
    """
    Estimates initial parameters for the V-integrated BM3 EoS in case
    the INI keyword is not present in "input.txt"
    
    Args:
        vv (list): volumes
        en (list): static energies at the corresponding volumes vv
        
    Returns:
        "ini" list of V-integrated EoS parameters estimated by a 
        polynomial fit: v_ini, k0_ini, kp, e0_ini. kp is set to 4.
        
    Note: 
        such parameters are used as initial guesses for the bm3 optimization.
    """
    kp_ini=4.
    pol=np.polyfit(vv,en,3)
    pder1=np.polyder(pol,1)
    pder2=np.polyder(pol,2)
    v_r=np.roots(pder1)
    vs=v_r*np.conj(v_r)
    min_r=np.argmin(vs)
    v_ini=np.real(v_r[min_r])
    e0_ini=np.polyval(pol, v_ini)
    k0_ini=np.polyval(pder2, v_ini)
    k0_ini=k0_ini*v_ini
    ini=[v_ini, k0_ini, kp_ini, e0_ini]
    return ini


# Output the pressure at a given temperature (tt) and volume (vv).
# Kp can be kept fixed (by setting fix=Kp > 0.1)
def pressure(tt,vv,**kwargs):
    """
    Computes the pressure at a temperature and volume
    
    Args:
        tt:  temperature
        vv:  unit cell volume
        fix (optional): optimizes Kp if fix=0., or keeps Kp 
                        fixed if fix=Kp > 0.1  
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    if fixpar:      
       [ff,veos,err]=bmx_tem(tt,fix=fix_value)
    else:
       [ff,veos,err]=bmx_tem(tt)  
    if bm4.flag:
        eos=veos[0:4]
        return round(bm4.pressure(vv,*eos)*conv/1e-21,3)
    else:     
       eos=veos[0:3]
       return round(bm3(vv,*eos)*conv/1e-21,3)

def pressure_dir(tt,vv):
    
    deg=pr.degree_v
    
    vmin=vv-pr.delta_v/2.
    vmax=vv+pr.delta_v/2.
    
    v_range=np.linspace(vmin,vmax,pr.nump_v)
    f_list=np.array([])
    for iv in v_range:
        fi=free_fit_vt(tt,iv)
        f_list=np.append(f_list,fi)
         
    vfit=np.polyfit(v_range,f_list,deg)
    vfitder=np.polyder(vfit,1)
    press=-1*np.polyval(vfitder,vv)
    return press*conv/1e-21


    
def volume_dir(tt,pp,alpha_flag_1=False, alpha_flag_2=False):
    
    """
    Computes the equilibrium volume at a given temperature and pressure
    without using an equation of state.
    
    An initial estimation of the volume is however obtained by using
    a BM3 EoS, by calling the eos_temp function; such volume is stored 
    in the v_new variable. 
    
    A list of volumes around the v_new value is then built and, for each 
    value in the list, a pressure is computed by using the pressure_dir
    function, and compared to the input pressure to find the volume
    at which the two pressures are equal.
    
    A number of parameters are used to control the computation. They are
    all defined by the volume-control driver (volume-ctrl). Convenient
    values are already set by default, but can be changed by using
    the method volume_ctrl.set_all. Use the info.show method to get such
    values under the 'volume driver section'.

    """  
    vol_opt.on()
    
    if volume_ctrl.kp_fix:
       reset_fix()
       if tt < volume_ctrl.t_max:           
          eos_temp(tt,kp_only=True)
       else:
          eos_temp(volume_ctrl.t_max,kp_only=True)
       set_fix(0)
       
    if (alpha_flag_1) and (not alpha_flag_2):
        reset_fix()
        eos_temp(tt,kp_only=True)
        set_fix(0)
             
    vini=new_volume(tt,pp)    
    v_new=vini[0]                # Initial volume from EoS
    
    if volume_ctrl.t_last_flag:
       vini=volume_ctrl.v_last
       
    if (tt > volume_ctrl.t_last) & (volume_ctrl.t_last > 10.): 
       volume_ctrl.t_last_flag=True
       volume_ctrl.shift=0.
       volume_ctrl.upgrade_shift=False
       
    if not flag_poly.flag:
        if flag_fit_warning.value:
           print("It is advised to use polynomial fits for 'dir' calculations\n")
           fit_status()
           print("")
           flag_fit_warning.value=False
    
    if flag_poly.flag:
        volume_max=max(flag_poly.fit_vol)
        volume_min=min(flag_poly.fit_vol)
    if flag_spline.flag:
        volume_max=max(flag_spline.fit_vol)
        volume_min=min(flag_spline.fit_vol)    
    
    if flag_poly.flag:
        if vini > volume_max:
           flag_volume_max.value=True
           if flag_volume_warning.value:
              flag_volume_warning.value=False
              print("Warning: volume exceeds the maximum value set in volume_range")
              print("Volume: %8.4f" % vini)
              fit_status()
              print("")
#           return vini
       
    if flag_spline.flag:
        if vini > volume_max:
           flag_volume_max.value=True
           if flag_volume_warning.value:
              flag_volume_warning.value=False 
              print("Warning: volume exceeds the maximum value set in volume_range")
              print("Volume: %8.4f" % vini)
              fit_status()
              print("")
#           return vini

    if volume_ctrl.t_last_flag:
       if (tt > volume_ctrl.t_last) & (volume_ctrl.t_last > 10.):
          vvi=volume_ctrl.v_last
          vplot=vvi
          v_list=np.linspace(vvi - volume_ctrl.delta/volume_ctrl.left,\
                       vvi + volume_ctrl.delta/volume_ctrl.right, 24) 
    else:       
       if tt > volume_ctrl.t_dump:
          volume_ctrl.shift=volume_ctrl.shift/volume_ctrl.dump         
       v_list=np.linspace(vini[0]-volume_ctrl.shift - volume_ctrl.delta/volume_ctrl.left,\
                       vini[0]-volume_ctrl.shift + volume_ctrl.delta/volume_ctrl.right, 24)
       vplot=vini[0]
       
    p_list=np.array([])
    for iv in v_list:
        pi=(pressure_dir(tt,iv)-pp)**2
        p_list=np.append(p_list,pi)   
      
    fitv=np.polyfit(v_list,p_list,volume_ctrl.degree)
    pressure=lambda vv: np.polyval(fitv,vv)
    
    min_p=np.argmin(p_list)
    vini=[v_list[min_p]]
    
    if volume_ctrl.degree > 2:
       bound=[(volume_min, volume_max)]
       vmin=minimize(pressure,vini,method='L-BFGS-B', bounds=bound, tol=1e-10,
                     options={'gtol':1e-10, 'maxiter':500})
       shift=v_new-vmin.x[0]
    else:
       shrink=volume_ctrl.quad_shrink
       new_v=np.linspace(vini[0]-volume_ctrl.delta/shrink, vini[0]+volume_ctrl.delta/shrink,8)
       new_p=np.array([])
       for iv in new_v:
           pi=(pressure_dir(tt,iv)-pp)**2
           new_p=np.append(new_p,pi)
           
       fit_new=np.polyfit(new_v, new_p,2)
       der_new=np.polyder(fit_new,1)
       vmin=-1*der_new[1]/der_new[0] 
       shift=v_new-vmin

    if volume_ctrl.upgrade_shift:
       volume_ctrl.shift=shift     
 
    if volume_ctrl.degree > 2:
       if volume_ctrl.debug:
          x=vini[0]
          y=0.95*np.max(p_list)
          y2=0.88*np.max(p_list)
          y3=0.81*np.max(p_list)
          y4=0.74*np.max(p_list)
          plt.figure()
          title="Temperature: "+str(round(tt,2))+" K"
          plt.plot(v_list,p_list)
          plt.title(title)
          v_opt="Opt volume:    "+str(vmin.x[0].round(4))
          v_min="Approx volume: "+str(vini[0].round(4))
          v_new="EoS volume:    "+str(v_new.round(4))
          v_ini="V_ini volume:  "+str(vplot.round(4))
          plt.text(x,y,v_opt,fontfamily='monospace')
          plt.text(x,y2,v_min, fontfamily='monospace')
          plt.text(x,y3,v_new,fontfamily='monospace')
          plt.text(x,y4,v_ini,fontfamily='monospace')         
          plt.show()
    else:
       if volume_ctrl.debug:
          x=vmin
          y=0.95*np.max(p_list)
          y2=0.88*np.max(p_list)
          y3=0.81*np.max(p_list)
          y4=0.74*np.max(p_list)
          plt.figure()
          title="Temperature: "+str(round(tt,2))+" K"
          plt.plot(v_list,p_list)
          plt.plot(new_v, new_p,"*")
          plt.title(title)
          v_opt="Opt. volume:   "+str(round(vmin,4))
          v_min="Approx volume: "+str(vini[0].round(4))
          v_new="EoS Volume:    "+str(v_new.round(4))
          v_ini="V_ini volume:  "+str(vplot.round(4))
          plt.text(x,y,v_opt,fontfamily='monospace')
          plt.text(x,y2,v_min, fontfamily='monospace')
          plt.text(x,y3,v_new,fontfamily='monospace')
          plt.text(x,y4,v_ini,fontfamily='monospace')
          plt.show()
           
    if volume_ctrl.degree > 2:   
       test=vmin.success
       if not test:
          print("\n**** WARNING ****")
          print("Optimization in volume_dir not converged; approx. volume returned")
          print("temperature: %5.2f, Volume: %6.3f" % (tt, vini[0]))
          volume_ctrl.v_last=vini[0]
          vol_opt.off()
                   
          return vini[0]
       else:
         volume_ctrl.v_last=vini[0]  
         return vmin.x[0]
    else:  
       volume_ctrl.v_last=vmin
       return vmin 
    
def find_temperature_vp(vv,pp, tmin=100., tmax=1000., prt=True):
    
    nt=50
    t_list=np.linspace(tmin,tmax,nt)     
    v_list=list(volume_dir(it,pp) for it in t_list)   
    diff_l=list((v_list[idx]-vv)**2 for idx in np.arange(len(v_list)))
    
    min_diff=np.argmin(diff_l)
    t_0=t_list[min_diff]
    
    delta=20.
    t_min=t_0-delta
    t_max=t_0+delta
    t_list=np.linspace(t_min,t_max,nt)
    v_list=list(volume_dir(it,pp) for it in t_list)
    diff_l=list((v_list[idx]-vv)**2 for idx in np.arange(len(v_list)))  
    min_diff=np.argmin(diff_l)
    t_0f=t_list[min_diff]  
    
    if prt:
       print("Temperature found:")
       print("First guess %5.2f; result: %5.2f K" % (t_0, t_0f))
    else:
      return t_0f  

def find_pressure_vt(vv,tt, pmin, pmax, prt=True):
    
    npp=50
    p_list=np.linspace(pmin,pmax,npp)     
    v_list=list(volume_dir(tt,ip) for ip in p_list)   
    diff_l=list((v_list[idx]-vv)**2 for idx in np.arange(len(v_list)))
    
    min_diff=np.argmin(diff_l)
    p_0=p_list[min_diff]
    
    delta=0.5
    p_min=p_0-delta
    p_max=p_0+delta
    p_list=np.linspace(p_min,p_max,npp)
    v_list=list(volume_dir(tt,ip) for ip in p_list)
    diff_l=list((v_list[idx]-vv)**2 for idx in np.arange(len(v_list)))  
    min_diff=np.argmin(diff_l)
    p_0f=p_list[min_diff]  
    
    if prt:
       print("Pressure found:")
       print("First guess %5.2f; result: %5.2f GPa" % (p_0, p_0f))
    else:
      return p_0f  
    

def bulk_dir(tt,prt=False,**kwargs):
    """
    Optimizes a BM3 EoS from volumes and total pressures at a given 
    temperature. In turn, phonon pressures are directly computed as volume
    derivatives of the Helmholtz function; static pressures are from a V-BM3
    fit of E(V) static data.
       
    Negative pressures are excluded from the computation.
    
    Args:
        tt: temperature
        prt (optional): if True, prints a P(V) list; default: False
        
    Keyword Args:
        fix: Kp fixed, if fix=Kp > 0.1 
        
    """
    
    flag_volume_max.value=False
    
    
    l_arg=list(kwargs.items())
    fixpar=False
    flag_serie=False
    vol_flag=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
       if 'serie' == karg_i[0]:
          flag_serie=karg_i[1]
       if 'volume' == karg_i[0]:
          vol_flag=karg_i[1]
          
    [dum,pterm,dum]=bmx_tem(tt)
    ini=pterm[0:3]
    
    flag_x=False
    if f_fix.flag:
       fix=f_fix.value
       flag_x=True
       p0_f=[ini[0],ini[1]]
       
    if fixpar:
       if fix_value < 0.1:
          flag_x=False
       else:
          fix=fix_value
          flag_x=True
          p0_f=[ini[0],ini[1]]
    
    if flag_spline.flag:
        v_list=flag_spline.fit_vol
    elif flag_poly.flag:
        v_list=flag_poly.fit_vol
    else:
        war1="Warning: frequency fit is off; use of poly or spline fits"
        war2=" is mandatory for bulk_dir"
        print(war1+war2)
        return
    
    f_fix_orig=f_fix.flag   
    volmax=volume_dir(tt,0.)
    if flag_volume_max.value:
       print("Computation stop. Use set_volume_range to fix the problem")
       stop()
        
    volnew=np.append(v_list,volmax)
    
    p_list=np.array([])    
    for vi in volnew:
        pi=pressure_dir(tt,vi)
        p_list=np.append(p_list,pi)
        
    v_new=np.array([])
    p_new=np.array([])    
    for iv in zip(volnew,p_list):
        if iv[1]>=-0.01:
            v_new=np.append(v_new,iv[0])
            p_new=np.append(p_new,iv[1])
    try:
       if flag_x:    
           pdir, pcov_dir = curve_fit(lambda v_new, v0, k0: \
                                  bm3(v_new, v0, k0, fix), \
                                  v_new, p_new, p0=p0_f, method='dogbox',\
                                  ftol=1e-15, xtol=1e-15)        
       else:    
           pdir, pcov_dir =  curve_fit(bm3, v_new, p_new, \
                        method='dogbox', p0=ini[0:3], ftol=1e-15, xtol=1e-15)
       
       perr_t=np.sqrt(np.diag(pcov_dir))
    except RuntimeError:
        print("EoS optimization did not succeeded for t = %5.2f" % tt)
        flag_dir.on()
        if flag_serie:
           return 0,0
        else:
           return
           
    if flag_x:
       pdir=np.append(pdir,fix)
       perr_t=np.append(perr_t,0.00)
       
    if flag_serie and vol_flag:
        return pdir[0],pdir[1],pdir[2]
    
    if flag_serie:
        return pdir[1],pdir[2]
              
       
    print("\nBM3 EoS from P(V) fit\n")
    print("K0:  %8.2f   (%4.2f) GPa"  % (pdir[1],perr_t[1]))
    print("Kp:  %8.2f   (%4.2f)    "  % (pdir[2],perr_t[2]))
    print("V0:  %8.4f   (%4.2f) A^3"    % (pdir[0],perr_t[0]))
    
    info.temp=tt
    info.k0=pdir[1]
    info.kp=pdir[2]
    info.v0=pdir[0]
    
    vol=np.linspace(min(v_new),max(v_new),16)
    press=bm3(vol,*pdir)
        
    plt.figure()
    plt.title("BM3 fit at T = %5.1f K\n" % tt)
    plt.plot(v_new,p_new,"k*")
    plt.plot(vol,press,"k-")
    plt.xlabel("Volume (A^3)")
    plt.ylabel("Pressure (GPa)")
    plt.show()
    
    if not f_fix_orig:
        reset_fix()
        
    if prt:
        print("\nVolume-Pressure list at %5.2f K\n" % tt)
        for vp_i in zip(v_new,p_new):
            print(" %5.3f   %5.2f" % (vp_i[0], vp_i[1])) 
            
def bulk_dir_serie(tini, tfin, npoints, degree=2, update=False, **kwargs):
     
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
        if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    
    t_serie=np.linspace(tini, tfin, npoints)
    tx_serie=np.array([])
    b_serie=np.array([])
    for ti in t_serie:
        flag_dir.off()
        if not fixpar:
               bi,kpi=bulk_dir(ti,serie=True)
        else:
               bi,kpi=bulk_dir(ti, serie=True, fix=fix_value)
        if not flag_dir.value:
           b_serie=np.append(b_serie,bi)
           tx_serie=np.append(tx_serie,ti)
        else:
            pass
    
    t_serie=tx_serie
    
    plt.figure()
    plt.plot(t_serie,b_serie,"k*")
    plt.title("Bulk modulus (K0)")
    plt.xlabel("T(K)")
    plt.ylabel("K (GPa)")
    plt.title("Bulk modulus as a function of T")
    
    fit_b=np.polyfit(t_serie,b_serie,degree)
    b_fit=np.polyval(fit_b,t_serie)
    plt.plot(t_serie,b_fit,"k-")
    print("\nResults from the fit (from high to low order)")
    np.set_printoptions(formatter={'float': '{: 4.2e}'.format})
    print(fit_b)
    np.set_printoptions(formatter=None)
    plt.show()
    
    if update:
       return fit_b       
    
    volume_ctrl.shift=0.    
    
def bm4_dir(tt,prt=True):
    """
    Optimizes a BM4 EoS from volumes and total pressures at a given 
    temperature. Negative pressures are excluded from the computation.
    
    Args:
        tt: temperature
        prt (optional): if True, prints a P(V) list; default: False
    """
    
    flag_volume_max.value=False
    
    start_bm4()
    
    if flag_spline.flag:
        v_list=flag_spline.fit_vol
    elif flag_poly.flag:
        v_list=flag_poly.fit_vol
    else:
        war1="Warning: frequency fit is off; use of poly or spline fits"
        war2=" is mandatory for bulk_dir"
        print(war1+war2)
        return
    
    volmax=volume_dir(tt,0.)
    if flag_volume_max.value:
        print("Computation stop. Use set_volume_range to fix the problem")
        stop()
        
    volnew=np.append(v_list,volmax)
    
    p_list=np.array([])    
    for vi in volnew:
        pi=pressure_dir(tt,vi)
        p_list=np.append(p_list,pi)
        
    v_new=np.array([])
    p_new=np.array([])    
    for iv in zip(volnew,p_list):
        if iv[1]>=-0.01:
            v_new=np.append(v_new,iv[0])
            p_new=np.append(p_new,iv[1])
    
    ini=np.copy(bm4.en_ini[0:4])
    ini[1]=ini[1]*conv*1e21
    
    pdir, pcov_dir =  curve_fit(bm4.pressure, v_new, p_new, \
                      p0=ini, ftol=1e-15, xtol=1e-15)
       
    perr_t=np.sqrt(np.diag(pcov_dir)) 
          
    print("\nBM4 EoS from P(V) fit\n")
    print("K0:  %8.2f   (%4.2f) GPa"  % (pdir[1],perr_t[1]))
    print("Kp:  %8.2f   (%4.2f)    "  % (pdir[2],perr_t[2]))
    print("Kpp: %8.2f   (%4.2f)    "  % (pdir[3], perr_t[3]))
    print("V0:  %8.4f   (%4.2f) A^3"  % (pdir[0],perr_t[0]))
        
    vol=np.linspace(min(v_new),max(v_new),16)
    press=bm4.pressure(vol,*pdir)
    
    plt.figure()
    plt.title("BM4 fit at T = %5.1f K\n" % tt)
    plt.plot(v_new,p_new,"k*")
    plt.plot(vol,press,"k-")
    plt.xlabel("Volume (A^3)")
    plt.ylabel("Pressure (GPa)")
    plt.show()
    
    if prt:
        print("\nVolume-Pressure list at %5.2f K\n" % tt)
        for vp_i in zip(v_new,p_new):
            print(" %5.3f   %5.2f" % (vp_i[0], vp_i[1]))         

    
def bulk_modulus_p(tt,pp,noeos=False,prt=False,**kwargs):
    """
    Bulk modulus at a temperature and pressure
    
    Args:
        tt: temperature
        pp: pressure
        noeos: to compute pressures, the bm3 EoS is used if 
               noeos=False (default); otherwise the EoS is
               used only for the static part, and vibrational
               pressures are obtained from the derivative
               of the F function (pressure_dir function) 
        prt: if True, results are printed
        fix (optional): optimizes Kp if fix=0., or keeps Kp 
                        fixed if fix=Kp > 0.1. This is relevant
                        if noeos=False
                        
    The values are computed through the direct derivative -V(dP/dV)_T. 
    Since the computation of pressure requires the bm3_tem function
    (if noeos=False) Kp can be kept fixed by setting fix=Kp > 0.1
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if not noeos:      
       if fixpar:
          vol=new_volume(tt,pp,fix=fix_value)[0]
       else:
          vol=new_volume(tt,pp)[0] 
    else:
       vol=volume_dir(tt,pp)
         
    delta=pr.delta_v
    numv=pr.nump_v
    degree=pr.degree_v
    v_range=np.linspace(vol-delta/2.,vol+delta/2.,numv)
    press_range=[]
    for iv in v_range:
        if not noeos:
           if fixpar:
              p_i=pressure(tt,iv,fix=fix_value)
           else:
              p_i=pressure(tt,iv)
        else:
           p_i=pressure_dir(tt,iv)
           
        press_range=np.append(press_range,p_i)
    press_fit=np.polyfit(v_range,press_range,degree)
    b_poly=np.polyder(press_fit,1)
    b_val=np.polyval(b_poly,vol)
     
    b_val=(-1*b_val*vol)
    
    if prt:
        eos=str(noeos)
        print("Bulk Modulus at T = %5.1f K and P = %3.1f GPa, noeos = %s: %6.3f GPa, V = %6.3f " %\
              (tt,pp,eos,b_val, vol))
    else:
        b_val=round(b_val,3)
        return b_val, vol

def bulk_modulus_p_serie(tini, tfin, nt, pres, noeos=False, fit=False, type='poly', \
                         deg=2, smooth=5, out=False, **kwargs):
    
    """
    Computes the bulk modulus from the definition K=-V(dP/dV)_T in a range
    of temperature values.
    
    Args:
        tini:   lower temperature in the range
        tfin:   higher temperature in the range
        nt:     number of points in the [tini, tfin] range
        pres:   pressure (GPa)
        noeos:  see note below
        fit:    if True, a fit of the computed K(T) values is performed
        type:   type of the fit ('poly', or 'spline')
        deg:    degree of the fit
        smooth: smooth parameter for the fit; relevant if type='spline'
        out:    if True, the parameters of the K(T) and V(T) fits are printed
        
   Keyword Args:
        fix:    if fix is provided, Kp is kept fixed at the fix value
                Relevant if noeos=False
                
   Note:
        if noeos=False, the pressure at any given volume is calculated 
        from the equation of state. If noeos=True, the pressure is computed
        as the first derivative of the Helmholtz function (at constant
        temperature)    
    """

    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    
    t_list=np.linspace(tini, tfin, nt)
    
    b_l=np.array([])    
    t_l=np.array([])
    v_l=np.array([])
    
    if fixpar:
       for it in t_list:
           ib, v_val=bulk_modulus_p(it,pres,noeos=noeos,fix=fix_value)
           if vol_opt.flag:
              b_l=np.append(b_l,ib)
              t_l=np.append(t_l,it)
              v_l=np.append(v_l,v_val)
    else:
       for it in t_list:
           ib,v_val=bulk_modulus_p(it,pres,noeos=noeos)
           if vol_opt.flag:
              t_l=np.append(t_l,it)
              b_l=np.append(b_l,ib) 
              v_l=np.append(v_l,v_val)
           
    if fit:
        t_fit=np.linspace(tini,tfin,50)
        if type=='poly':
           fit_par=np.polyfit(t_l,b_l,deg)        
           b_fit=np.polyval(fit_par,t_fit)
           
           fit_par_v=np.polyfit(t_l,v_l,deg)
           v_fit=np.polyval(fit_par_v,t_fit)
        elif type=='spline':
           fit_par=UnivariateSpline(t_l,b_l,k=deg,s=smooth)
           b_fit=fit_par(t_fit)
           
           fit_par_v=UnivariateSpline(t_l,v_l,k=deg,s=0.1)
           v_fit=fit_par_v(t_fit)
           
    method='poly'
    if type=='spline':
        method='spline'
        
    lbl=method+' fit'    
    plt.figure()
    plt.plot(t_l,b_l,"k*",label='Actual values')
    if fit:
        plt.plot(t_fit, b_fit,"k-",label=lbl)
    plt.xlabel("Temperature (K)")
    plt.ylabel("K (GPa)")
    tlt="Bulk modulus at pressure "+str(pres)
    plt.title(tlt)
    plt.legend(frameon=False)
    plt.show()
    
    reset_fix()
    
    if out & fit:
        return fit_par, fit_par_v

def bulk_modulus_adiabat(tt,pp,noeos=False, prt=True,**kwargs):
    """
    Adiabatic bulk modulus at a temperature and pressure
    
    Args:
        tt: temperature
        pp: pressure
        fix (optional): optimizes Kp if fix=0., or keeps Kp 
                        fixed if fix=Kp > 0.1
                        
    The values are computed through the direct derivative -V(dP/dV)_T. 
    Since the computation of pressure requires the bm3_tem function, 
    Kp can be kept fixed by setting fix=Kp > 0.1  
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
     
    if fixpar:    
       vol=new_volume(tt,pp,fix=fix_value)
       alpha,kt_dum,pr=thermal_exp_v(tt,vol,False,fix=fix_value)
       kt=bulk_modulus_p(tt,pp,noeos=noeos,fix=fix_value)
       ent,cv=entropy_v(tt,vol,False,False,fix=fix_value)
    else:
       vol=new_volume(tt,pp) 
       alpha,kt_dum,pr=thermal_exp_v(tt,vol,False)
       kt=bulk_modulus_p(tt,pp,noeos=noeos)
       ent,cv=entropy_v(tt,vol,False,False)
       
    volm=(vol*avo*1e-30)/zu
    ks=kt*(1+volm*(tt*1e9*kt*alpha**2)/cv)
    if prt:
       print("\nAdiabatic bulk modulus Ks: %5.2f GPa" % ks)
       print("Isoth. Kt: %5.2f GPa, alpha: %5.2e K^-1, sp. heat Cv: %6.2f J/mol K"\
             % (kt, alpha, cv))         
       print("Cell volume: %6.2f A^3, molar volume %6.2f cm^3" % (vol, 1e6*volm))
    else:
       return ks
        

def static(plot=False, vmnx=[0., 0.]):
    """
    Static EoS
    
    Args: 
        plot: plot of the E(V) curve
        vmnx: array of two reals [vmin and vmax]; vmin is the
              minimum volume and vmax is the maximum volume.
              If vmin and vmax are both 0., the whole V range
              is used (as specified in the static energies file).
              Default=[0., 0.]
      
    Note:            
        The volume range can also be modified by using the methods
        of the static_volume class
        
    Examples:
        >>> static_volume.set(100., 120.)
        >>> static_volume.on()
        >>> static(plt=True)
        
        Computes the static EoS in the [100., 120.] volume range. The same
        is obtained with
        
        >>> static(plt=True, vmnx=[100., 120.])
        
        However, with the first method the defined volume range is recorded for
        future computations; by using the second method, the volume range is reset
        to the original one, once the fit is performed.  
    """
    global popt, pcov
        
    if flag_err:
        return None
    
    vol_flag=False
    if static_range.flag:
       vol_min=static_range.vmin
       vol_max=static_range.vmax
       vol_flag=True
    else:    
       if (vmnx[0] > 0.1) or (vmnx[1] > 0.1):
          vol_flag=True
          vol_min=vmnx[0]
          vol_max=vmnx[1]
          
    if vol_flag:      
       vol_select=(volume >= vol_min) & (volume <= vol_max)
       vol_selected=volume[vol_select]
       energy_selected=energy[vol_select]
       
    if not vol_flag:
       popt, pcov = curve_fit(v_bm3, volume, energy, p0=ini,ftol=1e-15,xtol=1e-15)
    else:
       popt, pcov = curve_fit(v_bm3, vol_selected, energy_selected, p0=ini,ftol=1e-15,xtol=1e-15)
               
    k_gpa=popt[1]*conv/1e-21
    kp=popt[2]
    v0=popt[0]
    perr=np.sqrt(np.diag(pcov))
    ke=perr[1]*conv/1e-21

    print("\nStatic BM3 EoS")
    print("\nBulk Modulus: %5.2f (%4.2f) GPa" % (k_gpa, ke))
    print("Kp:            %5.2f (%4.2f)" % (kp, perr[2]))
    print("V0:           %5.4f (%4.2f) A^3" % (v0, perr[0]))
    print("E0:            %5.8e (%4.2e) hartree" % (popt[3], perr[3]))
    
    if vol_flag:
        print("\nStatic EoS computed in a restricted volume range:")
        print(vol_selected)
    
    print("\n")

    info.k0_static=k_gpa
    info.kp_static=kp
    info.v0_static=v0
    
    vol_min=np.min(volume)
    vol_max=np.max(volume)
    nvol=50
    vol_range=np.linspace(vol_min,vol_max,nvol)
    if not plot:
       return pcov
    else:        
       plt.figure(0)
       plt.title("E(V) static BM3 curve")
       plt.plot(volume,energy,"*")
       plt.plot(vol_range, v_bm3(vol_range, *popt), 'b-')
       plt.ylabel("Static energy (a.u.)")
       plt.xlabel("V (A^3)")
       plt.show()
       
def static_pressure_bm3(vv):
    """
    Outputs the static pressure (in GPa) at the volume (vv)
    
    Args:
        vv: volume
    """
    static(plot=False)
    k0=popt[1]
    kp=popt[2]
    v0=popt[0]
    p_static_bm3=bm3(vv,v0, k0,kp)
    ps=p_static_bm3*conv/1e-21
    print("Static pressure at the volume: %4.2f" % ps)
           
def start_bm4():
    bm4.on()
    bm4.estimates(volume,energy)
    with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       bm4p, bm4c = curve_fit(bm4.energy, volume, energy, \
                  method='dogbox', p0=bm4.en_ini,ftol=1e-15,xtol=1e-15,gtol=1e-15)
    bm4.store(bm4p)
    bm4.upgrade()
    bm4.upload(bm4p)
    bm4_k=bm4p[1]*conv/1e-21
    kp=bm4p[2]
    kpp=bm4p[3]
    v0=bm4p[0]
    print("\nStatic BM4-EoS")
    print("\nBulk Modulus:  %5.2f GPa" % bm4_k)
    print("Kp:             %5.2f " % kp)
    print("Kpp:            %5.2f " % kpp)
    print("V0:            %8.4f A^3" % v0)
    print("\n")
    plt.figure()
#    bm4e=np.array([])
    vbm4=np.linspace(min(volume),max(volume),50)
    bm4e=bm4.energy(vbm4,*bm4.bm4_static_eos)
    plt.plot(vbm4,bm4e,"k-")
    plt.plot(volume,energy,"k*")
    plt.title("Static Energy: BM4 fit")
    plt.xlabel("Static energy (a.u.)")
    plt.ylabel("V (A^3)")
    plt.show()
        
def free(temperature):
    """
    Computes the Helmholtz free energy (hartree) at a given temperature
    
    Args:
        temperature: temperature (in K) at which the computation is done
        
    Note: 
       1. ei is the static energy
       2. enz_i is the zero point energy
       3. fth_i is thermal contribution to the Helmholtz free energy 
       4. tot_i is the total Helmholtz free energy
    
    Note:
       This is a direct calculation that avoids the fit of a polynomium
       to the frequencies. No FITVOL in input.txt
    
    Note: 
      If kieffer.flag is True, the contribution from acoustic branches
      is taken into account, by following the Kieffer model.
    """
    energy_tot=[]
    for ivol in int_set:
        vol_i=data_vol_freq_orig[ivol]
        if bm4.flag:
           ei=bm4.energy(vol_i,*bm4.bm4_static_eos) 
        else: 
           ei=v_bm3(vol_i, *popt)
        enz_i=0.
        fth_i=0.
        eianh=0.
        
        if anharm.flag:
           eianh=0.
           for im in np.arange(anharm.nmode):
               eianh=eianh+helm_anharm_func(im,ivol,temperature)*anharm.wgt[im]
          
        for ifreq in int_mode:
            if ifreq in exclude.ex_mode:
                pass
            else:
                freq_i=lo.data_freq[ifreq,ivol+1]
                if freq_i >= 0.:
                   fth_i=fth_i+deg[ifreq]*np.log(1-np.e**(freq_i*e_fact/temperature))
                else:
                   print("Negative frequency found: mode n. %d" % ifreq)
                   stop()
                   
                enz_i=enz_i+deg[ifreq]*freq_i*ez_fact
                evib_i=enz_i+fth_i*kb*temperature/conv+eianh
                
                tot_i=ei+evib_i
        energy_tot=np.append(energy_tot,tot_i) 
                
    if kieffer.flag:  
           free_k=kieffer.get_value(temperature)
           free_k=free_k*zu*apfu/(avo*conv)
           energy_tot=energy_tot+free_k
          
    return energy_tot 


def free_fit(temperature):
    """
    Computes the Helmholtz free energy (in hartree) at a given temperature
    
    Args:
        temperature: temperature (in K)
        
    Note: 
       1. ei is the static energy
       2. enz_i is the zero point energy
       3. fth_i is thermal contribution to the Helmholtz free energy 
       4. tot_i is the total Helmholtz free energy
       
    Note:
       This computation makes use of polynomia fitted
       to the frequencies of each vibrational mode, as 
       functions of volume. It is activated by the keyword
       FITVOL in the input.txt file
    """
    energy_tot=[]
    eianh=0.
    
    if flag_spline.flag:
        fit_vol=flag_spline.fit_vol
    elif flag_poly.flag:
        fit_vol=flag_poly.fit_vol
           
    for ivol in fit_vol:
        if bm4.flag:
           ei=bm4.energy(ivol,*bm4.bm4_static_eos) 
           if anharm.flag:
               eianh=0.
               for im in np.arange(anharm.nmode):
                   eianh=eianh+helm_anharm_func(im,ivol,temperature)*anharm.wgt[im]
        else: 
           ei=v_bm3(ivol,*popt)
           if anharm.flag:
               eianh=0.
               for im in np.arange(anharm.nmode):
                 eianh=eianh+helm_anharm_func(im,ivol,temperature)*anharm.wgt[im]
        enz_i=0.
        fth_i=0.
        for ifreq in int_mode:
            if ifreq in exclude.ex_mode:
                pass
            else:
               if not flag_spline.flag:
                  freq_i=freq_v_fun(ifreq,ivol)
               else:
                  freq_i=freq_spline_v(ifreq,ivol)
               if freq_i >= 0.:
                  fth_i=fth_i+deg[ifreq]*np.log(1-np.e**(freq_i*e_fact/temperature))
               else:
                  print("Negative frequency found: mode n. %d" % ifreq)
                  stop()
                
               enz_i=enz_i+deg[ifreq]*freq_i*ez_fact
               evib_i=enz_i+fth_i*kb*temperature/conv+eianh
                                 
               tot_i=ei+evib_i
        energy_tot=np.append(energy_tot,tot_i)  
        

    if kieffer.flag:        
        free_k=kieffer.get_value(temperature)
        free_k=free_k*zu*apfu/(avo*conv)
        energy_tot=energy_tot+free_k
          
    return energy_tot 

def free_fit_vt(tt,vv):
    e_static=v_bm3(vv,*popt)
    enz=0
    fth=0
    eianh=0.
    
    if anharm.flag:
       eianh=0.
       for im in np.arange(anharm.nmode):
           eianh=eianh+helm_anharm_func(im,vv,tt)*anharm.wgt[im]
           
    for ifreq in int_mode:
        if ifreq in exclude.ex_mode:
           pass
        else:
           if not flag_spline.flag:
              freq_i=freq_v_fun(ifreq,vv)
           else:
              freq_i=freq_spline_v(ifreq,vv)
           if freq_i >= 0.:
              fth=fth+deg[ifreq]*np.log(1-np.e**(freq_i*e_fact/tt))
           else:
              print("Negative frequency found: mode n. %d" % ifreq)
              stop()
              
           enz=enz+deg[ifreq]*freq_i*ez_fact
        
    tot_no_static=enz+fth*kb*tt/conv+eianh       
    tot=e_static+tot_no_static 
        
    if kieffer.flag:        
        free_k=kieffer.get_value(tt)
        free_k=free_k*zu*apfu/(avo*conv)
        tot=tot+free_k
  
    if disp.flag and disp.eos_flag:
        
        if not disp.fit_vt_flag:
            disp.free_fit_vt()
            print("\n**** INFORMATION ****")
            print("The V,T-fit of the phonon dispersion surface was not done")
            print("it has been perfomed with default values of the relevant parameters")
            print("Use the disp.free_fit_vt function to redo with new parameters\n")
        
        disp_l=disp.free_vt(tt,vv)
        free_f=(tot_no_static+disp_l)/(disp.molt+1)   
        tot=e_static+free_f  
        
    return tot

def eos_temp_range(vmin_list, vmax_list, npp, temp):
    """
    EoS computed for different volumes ranges
    
    Args:
       vmin_list: list of minimum volumes
       vmax_list: list of maximum volumes
       npp: number of points in each V-range
       temp: temperature
    
    Note:
       vmin_list and vmax_list must be lists of same length
    """
    final=np.array([])
    size=len(vmin_list)
    for vmin, vmax in zip(vmin_list,vmax_list):
        v_list=np.linspace(vmin,vmax,npp)
        free_list=np.array([])
        for iv in v_list:
            ifree=free_fit_vt(temp, iv)
            free_list=np.append(free_list,ifree)
        
        pterm, pcov_term = curve_fit(v_bm3, v_list, free_list, \
                                   p0=ini, ftol=1e-15, xtol=1e-15) 
    
        k_gpa=pterm[1]*conv/1e-21
        k_gpa_err=pcov_term[1]*conv/1e-21
        pmax=pressure(temp,vmin)
        pmin=pressure(temp,vmax)
        
        final=np.append(final, [vmin, vmax, round(pmax,1), round(pmin,1), round(pterm[0],4), round(k_gpa,2), \
                                round(pterm[2],2)])
          
    final=final.reshape(size,7)
    final=final.T    
    pd.set_option('colheader_justify', 'center')
    df=pd.DataFrame(final, index=['Vmin','Vmax','Pmax','Pmin','V0','K0','Kp'])
    df=df.T
    print("\nBM3-EoS computed for different volume ranges")
    print("Temperature: %6.1f K" % temp)
    print("")
    print(df.to_string(index=False))  
             
def g_vt_dir(tt,pp,**kwargs):
    
    flag_volume_max.value=False
    
    l_arg=list(kwargs.items())
    v0_flag=False
    g0_flag=False
    for karg_i in l_arg:
      if 'g0' == karg_i[0]:
          g0_flag=True
          gexp=karg_i[1]
      elif 'v0' == karg_i[0]:
          v0_flag=True
          v0_value=karg_i[1]
    
    vol0=volume_dir(298.15,0.0001)
       
    fact=1.
    if v0_flag:
       fact=(1e25*v0_value*zu/avo)/vol0
       
    gref=free_fit_vt(298.15,vol0)*conv*avo/zu + 0.0001*vol0*fact*avo*1e-21/zu

    if g0_flag:
       gref=gref-gexp
     
    vv=volume_dir(tt,pp) 
    if flag_volume_max.value:
       flag_volume_max.inc()
       if flag_volume_max.jwar < 2:
           print("Warning g_vt_dir: volume exceeds maximum set in volume_range") 
    
    free_f=free_fit_vt(tt,vv)

    gtv=(avo/zu)*(free_f*conv) + (avo/zu)*pp*vv*fact*1e-21
    
    
    return gtv-gref  


def entropy_v(tt,vv, plot=False, prt=False, **kwargs):
    """
    Entropy and specific heat at constant volume 
    
    Args:
        tt: temperature
        vv: volume
        plot (optional): (default False) plots free energy vs T for checking
                         possible numerical instabilities
        prt (optional):  (default False) prints formatted output
         
    Keyword Args: 
        fix: if fix is provided, it controls (and overrides the setting 
             possibly chosen by set_fix) the optimization of kp in BM3; 
             if fix > 0.1, kp = fix and it is not optimized. 
             
    Returns:
        if prt=False (default) outputs the entropy and the specific heat
        at constant volume (unit: J/mol K). if prt=True, a formatted
        output is printed and the function provides no output
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    delta=pr.delta
    nump=pr.nump
    degree=pr.degree
    maxv=max(data_vol_freq)
    free_f=[]
    min_t=tt-delta/2.
    max_t=tt+delta/2.
    if min_t < 0.1:
        min_t=0.1
    t_range=np.linspace(min_t,max_t,nump)
    for i_t in t_range: 
       if fixpar: 
          [free_energy, pterm, pcov_term]=bmx_tem(i_t,fix=fix_value)
       else:
          [free_energy, pterm, pcov_term]=bmx_tem(i_t) 
       if (pterm[0]>maxv): 
           if flag_warning.value:
            print("\nWarning: volume out of range; reduce temperature")
            flag_warning.off()
            flag_warning.inc()
       if bm4.flag:
          f1=bm4.energy(vv,*pterm) 
       else:
          f1=v_bm3(vv,*pterm)
       free_f=np.append(free_f,f1)
       
    if disp.flag:
        disp_l=[]
        disp.free_fit(disp.temp,vv,disp=False)
        for i_t in t_range:
            idf=disp.free_func(i_t)
            disp_l=np.append(disp_l,idf)
        free_f=(free_f+disp_l)/(disp.molt+1)
            
    if plot:
       plt.figure(4)
       plt.plot(t_range,free_f,"*")
       plt.title("F free energy (a.u.)")
       plt.show()
    fit=np.polyfit(t_range,free_f,degree)
    der1=np.polyder(fit,1)
    der2=np.polyder(fit,2)     
    entropy=-1*np.polyval(der1,tt)*conv*avo/zu
    cv=-1*np.polyval(der2,tt)*tt*conv*avo/zu
        
    if prt:
        print("\nEntropy: %7.2f J/mol K" % entropy)
        print("Specific heat (at constant volume): %7.2f J/mol K" % cv)

        return None
    else:
        return entropy, cv 


def entropy_p(tt,pp,plot=False,prt=True,**kwargs):
    """
    Entropy and specific heat at constant volume at selected temperature
    and pressure
    
    Args:
        tt: temperature
        pp: pressure
        plot (optional): (default False) plots free energy vs T for checking
                         possible numerical instabilities
        prt (optional):  (default True) prints formatted output
         
    Keyword Args: 
        fix: if fix is provided, it controls (and overrides the setting 
             possibly chosen by set_fix) the optimization of kp in BM3; 
             if fix > 0.1, kp = fix and it is not optimized. 
             
    Returns:
        if prt=False outputs the entropy (J/mol K); if prt=True (default), 
        a formatted output is printed and the function returns None
    """ 
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if fixpar:      
       vol=new_volume(tt,pp,fix=fix_value)
       ent_v=entropy_v(tt,vol,plot,prt,fix=fix_value)
    else:
       vol=new_volume(tt,pp)
       ent_v=entropy_v(tt,vol,plot,prt)
    if prt:
       print("Pressure: %5.2f GPa; Volume %8.4f A^3" % (pp, vol))
       return None
    else:
       return ent_v
   
    
def thermal_exp_v(tt,vv,plot=False,**kwargs):

    """
    Thermal expansion at a given temperature and volume
    
    Args:
        tt:              temperature
        vv:              volume
        plot (optional): (default False) plots pressure vs T for checking
                         possible numerical instabilities  
    
    Keyword Args: 
        fix: if fix is provided, it controls (and overrides the setting 
             possibly chosen by set_fix) the optimization of kp in BM3; 
             if fix > 0.1, kp = fix and it is not optimized. 
  
    Returns: 
          thermal expansion (K^-1), bulk modulus (GPa) and pressure (GPa)
          at given temperature=tt and volume=vv
            
    Notes: 
        The value is obtained by calculating (dP/dT)_V divided by K
        where K=K0+K'*P; P is obtained by the BM3 EoS's whose parameters 
        (at temperatures in the range "t_range") are refined by fitting
        the free energy F(V,T) curves. The different pressures calculated 
        (at constant vv) for different T in t_range, are then fitted by a
        polynomial of suitable degree  ("degree" variable) which is then 
        derived analytically at the temperature tt, to get (dP/dT)_V
           
        If "fix" > 0.1, the BM3 fitting is done by keeping kp fixed at the 
        value "fix". 
           
        The function outputs the thermal expansion (in K^-1), the bulk 
        modulus [at the pressure P(vv,tt)] and the pressure (in GPa) 
        if the boolean "plot" is True (default) a plot of P as a 
        function of T is plotted, in the range t_range         
    """
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    delta=pr.delta_alpha
    nump=pr.nump_alpha
    degree=pr.degree_alpha
    maxv=max(data_vol_freq)
    pressure=[]
    min_t=tt-delta/2.
    max_t=tt+delta/2.
    if min_t < 0.1:
        min_t=0.1
    t_range=np.linspace(min_t,max_t,nump)
    for ict in t_range:
        if fixpar:        
          [free_energy, pterm, pcov_term]=bmx_tem(ict,fix=fix_value)
        else:
          [free_energy, pterm, pcov_term]=bmx_tem(ict)   
        if bm4.flag:
           f1=bm4.pressure(vv,pterm[0],pterm[1],pterm[2],pterm[3])*\
              conv/1e-21
        else:
           f1=bm3(vv,pterm[0],pterm[1],pterm[2])*conv/1e-21
        pressure=np.append(pressure,f1)
        if (pterm[0]>maxv):
           if flag_warning.value:
               print("\nWarning: volume out of range; reduce temperature")
               flag_warning.off()
           flag_warning.inc()
    if plot:
       plt.figure(5)
       plt.plot(t_range,pressure,"*")
       plt.title("Pressure (GPa)")
       plt.show()
    fit=np.polyfit(t_range,pressure,degree)
    der1=np.polyder(fit,1)
    if fixpar:
       [free_energy, pterm, pcov_term]=bmx_tem(tt,fix=fix_value)
    else:
       [free_energy, pterm, pcov_term]=bmx_tem(tt)  
    if bm4.flag:
        pressure=bm4.pressure(vv,pterm[0],pterm[1],pterm[2],pterm[3])*\
              conv/1e-21
    else:
        pressure=bm3(vv,pterm[0],pterm[1],pterm[2])*conv/1e-21
    k=(pterm[1]*conv/1e-21)+pterm[2]*pressure
    return np.polyval(der1,tt)/k,k,pressure


def thermal_exp_p(tt,pp,plot=False,exit=False,**kwargs):
    """
    Thermal expansion at given temperature and pressure
    
    Args:
        tt:               temperature
        pp:               pressure
        plot (optional):  plots pressure vs T values (see help to
                          the thermal_exp_v function)
    Keyword Args: 
        fix: if fix is provided, it controls (and overrides the setting 
             possibly chosen by set_fix) the optimization of kp in BM3; 
             if fix > 0.1, kp = fix and it is not optimized.
             
    Note: 
        see help for the thermal_exp_v function  
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    if fixpar:
       vol=new_volume(tt,pp,fix=fix_value)
       [alpha,k,pressure]=thermal_exp_v(tt,vol,plot,fix=fix_value)
    else:
       vol=new_volume(tt,pp)
       [alpha,k,pressure]=thermal_exp_v(tt,vol,plot)
    if exit:
       return alpha
    else:
       print("\nThermal expansion: %6.2e K^-1" % alpha)
       print("Bulk modulus:        %6.2f GPa" % k)
       print("Pressure:            %6.2f GPa" % pressure)
       print("Volume:              %8.4f A^3\n" % vol)


def alpha_serie(tini,tfin,npoint,pp,plot=False,prt=True, fit=True,\
                HTlim=0.,degree=1, save='',dpi=300, g_deg=1, **kwargs):
    
    """
    Thermal expansion in a temperature range, at a given pressure (pp), 
    and (optional) fit with a polynomium whose powers are specified 
    in the input.txt file    
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if HTlim > 0.:
          alpha_limit=grun_therm_serie(tini,tfin,npoint=12,HTlim=HTlim,degree=degree,\
                                     g_deg=g_deg, ex=True)

    t_range=np.linspace(tini,tfin,npoint)
    alpha_serie=[]
    for ict in t_range:
        if fixpar:
           vol=new_volume(ict,pp,fix=fix_value)
           [alpha_i,k,pressure]=thermal_exp_v(ict,vol,plot,fix=fix_value)
        else:
           vol=new_volume(ict,pp)
           [alpha_i,k,pressure]=thermal_exp_v(ict,vol,plot)  
        alpha_serie=np.append(alpha_serie,alpha_i)
        
    if HTlim > 0:
        t_range=np.append(t_range,HTlim)
        alpha_serie=np.append(alpha_serie,alpha_limit)
        
    fig=plt.figure(10)
    ax=fig.add_subplot(111)
    ax.plot(t_range,alpha_serie,"k*") 
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax.set_xlabel("T (K)")
    ax.set_ylabel("Alpha (K^-1)")
    plt.title("Thermal expansion") 
    if prt:
       serie=(t_range, alpha_serie)
       df=pd.DataFrame(serie,index=['Temp.','alpha'])
       df=df.T
       print("\n")
       df['alpha']=df['alpha'].map('{:,.3e}'.format)
       df['Temp.']=df['Temp.'].map('{:,.2f}'.format)
       print(df.to_string(index=False)) 
    if fit:
       if flag_alpha==False:
          print("\nWarning: no polynomium defined for fitting alpha's")
          print("Use ALPHA keyword in input file")
          return None
       coef_ini=np.ones(lpow_a)
       alpha_fit, alpha_cov=curve_fit(alpha_fun,t_range,alpha_serie,p0=coef_ini)
       
       tvfin=tfin
       if HTlim > 0:
           tvfin=HTlim
           
       t_value=np.linspace(tini,tvfin,pr.ntemp_plot_cp)
       alpha_value=[]
       for ict in t_value:
           alpha_i=alpha_fun(ict,*alpha_fit)
           alpha_value=np.append(alpha_value,alpha_i)
       plt.plot(t_value,alpha_value,"k-")
    if save !='':
       plt.savefig(fname=path+'/'+save,dpi=dpi)
    plt.show()
    if prt:
        return None
    elif fit:
        return alpha_fit
    else:
        return None
     

def alpha_fun(tt,*coef):
    """
    Outputs the thermal expansion at a given temperature, from
    the fit obtained with the alpha_serie function
    """
    alpha=0.
    jc=0
    while jc<lpow_a:
        alpha=alpha+coef[jc]*(tt**power_a[jc])
        jc=jc+1
    return alpha



def dalpha_dt(tt,pp,**kwargs):
    """
    Outputs the derivative of alpha with respect to T
    at constant pressure. It is used by dCp_dP
    """
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    delta=pr.delta_alpha
    nump=pr.nump_alpha
    degree=pr.degree_alpha
    alpha=[]
    min_t=tt-delta/2.
    max_t=tt+delta/2.
    if min_t < 0.1:
       min_t=0.1
    t_range=np.linspace(min_t,max_t,nump)
    for ict in t_range:
        if fixpar:
           alpha_i=thermal_exp_p(ict,pp,fix=fix_value,exit=True)
        else:
           alpha_i=thermal_exp_p(ict,pp,exit=True) 
        alpha=np.append(alpha,alpha_i)
    fit=np.polyfit(t_range,alpha,degree)
    dfit=np.polyder(fit,1)
    return np.polyval(dfit,tt)

def alpha_dir(tt,pp):
    
    dt=5.
    nt=pr.nump
    dt2=dt/2.
    deg=2
    
    alpha_opt.on()
    
    v0=volume_dir(tt,pp,alpha_flag_1=True, alpha_flag_2=False)
    if not vol_opt.flag:
        alpha_opt.off()
        
    t_list=np.linspace(tt-dt2, tt+dt2, nt)
    vl=np.array([])
    tl=np.array([])
    
    
    for it in t_list:
        iv=volume_dir(it,pp,alpha_flag_1=True, alpha_flag_2=True)
        if vol_opt.flag:
           vl=np.append(vl,iv)
           tl=np.append(tl,it)
        
    fit=np.polyfit(tl,vl,deg)
    fit_d=np.polyder(fit,1)
    
    alpha=np.polyval(fit_d,tt)
    alpha=alpha/v0
    
    return alpha

def alpha_dir_serie(tmin, tmax, nt, pp, fit=True, prt=True):
    
    t_list=np.linspace(tmin,tmax,nt)
    
    t_l=np.array([])
    alpha_l=np.array([])
    
    for it in t_list:
        ial=alpha_dir(it,pp)
        if alpha_opt.flag:         
           alpha_l=np.append(alpha_l,ial)
           t_l=np.append(t_l,it)
        
    if fit:
       if flag_alpha==False:
          print("\nWarning: no polynomium defined for fitting alpha's")
          print("Use ALPHA keyword in input file")
          return None
       
       coef_ini=np.ones(lpow_a)
       alpha_fit, alpha_cov=curve_fit(alpha_dir_fun,t_l,alpha_l,p0=coef_ini)    
        
    if fit:
       t_list=np.linspace(tmin,tmax,nt*4)
       alpha_fit_c=alpha_dir_fun(t_list,*alpha_fit)

    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_l,alpha_l,"k*")
    if fit:
       ax.plot(t_list, alpha_fit_c,"k-")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    plt.xlabel("T (K)")
    plt.ylabel("Alpha (K^-1)")
    plt.title("Thermal expansion")
    plt.show()
        
    if not prt and fit:
        return alpha_fit
    
    volume_ctrl.shift=0.
    
def alpha_dir_fun(tt,*coef):
    """
    Outputs the thermal expansion at a given temperature, from
    the fit obtained with the alpha_dir_serie function
    """
    alpha=0.
    jc=0
    while jc<lpow_a:
        alpha=alpha+coef[jc]*(tt**power_a[jc])
        jc=jc+1
    return alpha
  
def cp(tt,pp,plot=False,prt=False,dul=False,**kwargs):
    """
    Specific heat at constant pressure (Cp) and entropy (S)
    
    Args:
        tt: temperature
        pp: pressure
        fix (optional): optimizes Kp if fix=0, or keeps Kp 
                        fixed if fix=Kp > 0
        plot (optional): checks against numerical issues 
                         (experts only)
        prt (optional): prints formatted results
        
    Note: 
        Cp = Cv + V*T*K*alpha^2
     
        Cp, Cv (J/mol K), Cp/Cv, alpha (K^-1), K=K0+K'P (GPa) 
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    pr_e=False
    if fixpar:
       vol=new_volume(tt,pp,fix=fix_value)
       [ent,cv]=entropy_v(tt,vol,plot,pr_e,fix=fix_value)
       [alpha,k,pressure]=thermal_exp_v(tt,vol,plot,fix=fix_value)
    else:
       vol=new_volume(tt,pp)
       [ent,cv]=entropy_v(tt,vol,plot,pr_e)
       [alpha,k,pressure]=thermal_exp_v(tt,vol,plot) 
    cp=cv+vol*(avo*1e-30/zu)*tt*k*1e9*alpha**2
    if prt:
       print("\nCp: %6.2f, Cv: %6.2f, Cp/Cv: %7.5f, alpha: %6.3e, K: %6.2f\n"\
               % (cp, cv, cp/cv, alpha, k))
       return None
    elif dul == False:
       return cp[0], ent
    else:
       return cp[0],ent,cp/cv 
        
      
def cp_fun(tt,*coef):
    """
    Computes the specific heat a constant pressure, at a given temperature 
    from the fit Cp(T) performed with the cp_serie function  
    """
    cp=0.
    jc=0
    while jc<lpow:
        cp=cp+coef[jc]*(tt**power[jc])
        jc=jc+1
    return cp

def dcp_dp(tt,pp,**kwargs):
    """
    Derivative of Cp with respect to P (at T constant)
    
    Args:
        tt: temperature
        pp: pressure
        fix (optional): fixed Kp value; if fix=0., Kp is
                        optimized 
    
    Notes:
        The derivative is evaluated from the relation
        (dCp/dP)_T = -VT[alpha^2 + (d alpha/dT)_P]
        It is **strongly** advised to keep Kp fixed (Kp=fix)
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
     
    if fixpar:     
       vol=new_volume(tt,pp,fix=fix_value)
       dalpha=dalpha_dt(tt,pp,fix=fix_value)
       alpha,k,pres=thermal_exp_v(tt,vol,fix=fix_value,plot=False)
    else:
       vol=new_volume(tt,pp)
       dalpha=dalpha_dt(tt,pp)
       alpha,k,pres=thermal_exp_v(tt,vol,plot=False)
    dcp=-1*(vol*avo*1e-21/zu)*tt*(alpha**2+dalpha)
    print("\n(dCp/dP)_T:   %5.2f J/(mol K GPa) " % dcp)
    print("(dAlpha/dT)_P: %6.2e K^-2 " % dalpha) 
    


def compare_exp(graph_exp=True,save="",dpi=300,**kwargs):
    """
    Compare experimental with computed data for Cp and S; 
    makes a plot of the data  
    """
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if not flag_exp:
        print("Warning: experimental data file not found")
        return
    t_list=data_cp_exp[:,0]
    cp_exp_list=data_cp_exp[:,1]
    s_exp_list=data_cp_exp[:,2]
    cp_calc=[]
    s_calc=[]
    for ti in t_list:
        if fixpar:
           cp_i, ent_i=cp(ti,0.,fix=fix_value,plot=False,prt=False)
        else:
           cp_i, ent_i=cp(ti,0.,plot=False,prt=False)   
        cp_calc=np.append(cp_calc,cp_i)
        s_calc=np.append(s_calc, ent_i)
    cp_diff=cp_calc-cp_exp_list
    s_diff=s_calc-s_exp_list
    exp_serie=(t_list,cp_exp_list,cp_calc,cp_diff,s_exp_list,s_calc,\
               s_diff)
    df=pd.DataFrame(exp_serie,\
       index=['Temp','Cp exp','Cp calc','Del Cp','S exp','S calc','Del S'])
    df=df.T
    df2=df.round(2)
    print("")
    print(df2.to_string(index=False))
    dcp=abs(df["Del Cp"].values)
    ds=abs(df["Del S"].values)
    mean_error_cp=dcp.mean()
    mean_error_s=ds.mean()
    max_error_cp=dcp.max()
    max_error_s=ds.max()
    print("\nAverage error on Cp: %.2f; maximum error: %.2f" % \
          (mean_error_cp, max_error_cp))
    print("Average error on S:  %.2f; maximum error: %.2f" % \
          (mean_error_s, max_error_s))
    if graph_exp:
       if not flag_cp:
          print("\nWarning: no polynomium defined for fitting Cp's" )
          print("No graphical comparison made") 
          return
       tmin=np.min(t_list)
       tmax=np.max(t_list)
       npoint=pr.ntemp_fit_compare
       if fixpar:
          cp_fit=cp_serie(tmin,tmax,npoint,0.,fix=fix_value, plot=False,\
                    prt=False,fit=True,graph=False)
       else:
          cp_fit=cp_serie(tmin,tmax,npoint,0., plot=False,\
                    prt=False,fit=True,graph=False)       
       ntemp=pr.ntemp_plot_compare
       t_graph=np.linspace(tmin,tmax,ntemp)
       cp_graph=[]
       for it in t_graph:
           cp_graph_i=cp_fun(it,*cp_fit)
           cp_graph=np.append(cp_graph,cp_graph_i)
       plt.figure(11)
       plt.plot(t_graph,cp_graph,"k-", label='Calc')
       plt.plot(t_list,cp_exp_list,"k*", label='Exp')
       plt.xlabel("T(K)")
       plt.ylabel("Cp (J/mol K)")
       plt.title("Experimental vs. calculated Cp(T)")
       plt.legend()
       if save != '':
          plt.savefig(fname=path+'/'+save, dpi=dpi)
       plt.show()
       if not flag_warning.value:
           print("Warning: issue on volume repeated %d times" % \
                   flag_warning.jwar)
           flag_warning.reset()
           flag_warning.value=True
      
      
def cp_serie(tini,tfin,points,pp, HTlim=0., model=1, g_deg=1, plot=False,prt=False, \
             fit=True,graph=True,save='',dpi=300,**kwargs):
    
    """
    Outputs a list of Cp values (J/mol K) in a given temperature range,
    at a fixed pressure
    
    Args:
        tini: minimum temperature (K)
        tfin: maximum temperature 
        points: number of points in the T range
        pp: pressure (GPa)
        HTlim: if HTlim > 0, the Dulong-Petit limit (DP)  for Cp is imposed at a 
               high T value (HTlim); the procedure is performed by computing
               Cv in the [tini, tfin] T range and by fitting the Cv curve by the
               Einstein's model after the DP limit is added for T=HTlim;
               The gamma value (Cp/Cv) at T > tfin is linerarly extrapolated from the 
               gamma(T) fit obtained in the [tini,tfin] range. For T > tfin 
               (and up to HTlim) Cp is computed as the product of Cv (from the 
               Einstein's model) and the extrapolated gamma.  
        fix (optional): keeps Kp fixed at the value Kp=fix if
                        fix > 0.1
        prt (optional): print a table of Cp(T) values if prt=True (default)
        fit (optional): fits the Cp(T) values with a polynomial function
                        whose powers must be specified in the input.txt
                        file
        graph (optional): makes a plot of the Cp(T) serie and its fit
                          if graph=True (default)
        save (optional, string): saves the plot image in the file name
                                 specified
        dpi (optional): dpi resolution of the saved image
        
    Note: 
        to output the coefficients of the fit, prt must be set to
        False
          
        The optional argument plot (default: False) is for checking
        possible numerical issues
          
        It is advised to keep Kp fixed during the computation
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    dlflag=False
    if HTlim > 0.:
       dlflag=True
       print("\n*** High temperature Cp estimation from the Dulong-Petit limit\n")
       print(" T limit: %5.2f" % HTlim)
       t_extra=np.linspace(tfin+20,HTlim,16)
       cp_lim=np.array([])
       if model==1:
          ein_t=einstein_t(tini,tfin,12,HTlim,dul=True)
       else:
          ein_t=einstein_t(tini,tfin,12,HTlim,dul=True,model=2) 
       pol_gamma=gamma_estim(tini,tfin,npoint=12,g_deg=g_deg)
       print("Gamma estimation (extrapolation from lower T values)\n")
       for ix in t_extra:  
           if model==1:
              cv_ix=einstein_fun(ix,ein_t[0])
           else:
              cv_ix=einstein_2_fun(ix,ein_t[0],ein_t[1]) 
           gam_ix=gamma_calc(ix,pol_gamma)
           cp_ix=cv_ix*gam_ix
           print("T: %8.2f  Cv: %8.2f  gamma: %5.3f  Cp: %8.2f" % (ix, cv_ix, gam_ix, cp_ix))
           cp_lim=np.append(cp_lim,cp_ix)
          
    prt_c=False
    t_serie=np.linspace(tini,tfin,points)
    cp_serie=[]
    for ict in t_serie:
        if fixpar:
            cpi, ent_i=cp(ict,pp,plot,prt_c,fix=fix_value)
        else:
            cpi, ent_i=cp(ict,pp,plot,prt_c)
        cp_serie=np.append(cp_serie,cpi)
    if dlflag:
       cp_serie=np.append(cp_serie,cp_lim)
       t_serie=np.append(t_serie,t_extra)
    if prt:
       serie=(t_serie, cp_serie)
       df=pd.DataFrame(serie,index=['Temp.','Cp'])
       df=df.T
       print("\n")
       df2=df.round(2)
       print(df2.to_string(index=False))
    if graph:
       plt.figure(6)
       plt.plot(t_serie,cp_serie,"k*")
    if fit:
       if not flag_cp:
          print("\nWarning: no polynomium defined for fitting Cp's")
          print("Use CP keyword in input file")
          return None
       coef_ini=np.ones(lpow)
       with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cp_fit, cp_cov=curve_fit(cp_fun,t_serie,cp_serie,p0=coef_ini)
       if dlflag:
          tfin=HTlim
       t_value=np.linspace(tini,tfin,pr.ntemp_plot_cp)      
       cp_value=[]
       for ict in t_value:
          cpi=cp_fun(ict,*cp_fit)
          cp_value=np.append(cp_value,cpi)
       if graph:     
          plt.plot(t_value,cp_value,"k-")
          plt.xlabel("T(K)")
          plt.ylabel("Cp (J/mol K)")
          plt.title("Specific heat as a function of T")
          if save !='':
              plt.savefig(fname=path+'/'+save,dpi=dpi)
          plt.show()
    if prt:
       return None
    elif fit:
       return cp_fit
    else: 
        return None
    
def gamma_estim(tini,tfin,npoint=12,g_deg=2):
    t_list=np.linspace(tini,tfin,npoint)
    gamma_list=np.array([])
    for it in t_list:
       dum1,dum2,gamma=cp(it,0,dul=True)
       gam=gamma[0]
       gamma_list=np.append(gamma_list,gam)
    pol=np.polyfit(t_list,gamma_list,g_deg)
    gamma_fit.upload(g_deg,pol)
    
    gamma_calc_list=list(gamma_calc(it, pol) for it in t_list)
    
    plt.figure()
    plt.plot(t_list,gamma_list,"*")
    plt.plot(t_list,gamma_calc_list,"k-")
    plt.xlabel("T(K)")
    plt.ylabel("Gamma")
    plt.title("Gamma (Cp/Cv) as a function of T")
    plt.show()
     
    return pol

def gamma_calc(tt,pol):
    return np.polyval(pol,tt)
    

def bulk_serie(tini,tfin,npoint,fit=True,degree=2,update=False,\
               save='',dpi=300,**kwargs):
    """
    Computes the bulk modulus K0 as a function of temperature in a given
    T range
    
    Args:
        tini: minimum temperature
        tfin: maximum temperature
        npoint: number of points in the T range
        fix (optional): keeps Kp constant in the calculation of K0
                        if fix=Kp > 0.1. If fix=0. Kp is optimized
                        at every different temperature.
        fit (optional): makes a polynomial fit of the K0(T) values
        degree (optional): degree of polynomial for the fitting
        save (optional, string): file name of the saved plot
        dpi (optional, integer): dpi resolution of the saved image                 
        
    Note: 
        the fix argument overrides the value of Kp possibly set 
        by the set_fix function
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    t_serie=np.linspace(tini,tfin,npoint)
    b_serie=[]
    for ict in t_serie:
        if fixpar:
           [free_energy, pterm, pcov_term]=bmx_tem(ict,fix=fix_value)
        else:
           [free_energy, pterm, pcov_term]=bmx_tem(ict)
        k0t=pterm[1]*conv/1e-21
        b_serie=np.append(b_serie,k0t)
    plt.figure(7)
    plt.plot(t_serie,b_serie,"k*")
    plt.title("Bulk modulus (K0)")
    plt.xlabel("T(K)")
    plt.ylabel("K (GPa)")
    plt.title("Bulk modulus as a function of T")
    if fit:
        fit_b=np.polyfit(t_serie,b_serie,degree)
        b_fit=np.polyval(fit_b,t_serie)
        plt.plot(t_serie,b_fit,"k-")
        print("\nResults from the fit (from high to low order)")
        np.set_printoptions(formatter={'float': '{: 4.2e}'.format})
        print(fit_b)
        np.set_printoptions(formatter=None)
    if save !='':
       plt.savefig(fname=path+'/'+save,dpi=dpi)
    plt.show()
    
    if update:
       return fit_b
    

def free_v(tt,vol,**kwargs):
    """
    Helmholtz free energy at a given temperature and volume
    Unit: a.u.
    
    Args:
        tt: temperature (K)
        vol: cell volume (A^3)
        
    Keyword Args: 
        fix: if fix is provided, it controls (and overrides the setting 
             possibly chosen by set_fix) the optimization of kp in BM3; 
             if fix > 0.1, kp = fix and it is not optimized.
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    if fixpar:
       [ff,pterm,pcov]=bmx_tem(tt,fix=fix_value)
    else:
       [ff,pterm,pcov]=bmx_tem(tt) 
    return v_bm3(vol,*pterm)


def gibbs_p(tt,pp,**kwargs):
    l_arg=list(kwargs.items())
    fixpar=False
    v0_flag=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
       elif 'v0' == karg_i[0]:        
          v0_flag=True
          v0_value=karg_i[1]
          
    if fixpar:
        vol=new_volume(tt,pp,fix=fix_value)
        f_energy=free_v(tt,vol[0],fix=fix_value)
    else:
        vol=new_volume(tt,pp)
        f_energy=free_v(tt,vol[0])
    
    if disp.flag:
        f_disp=disp.free_func(tt)+v_bm3(vol[0],*popt)*disp.molt
        f_energy=(f_energy+f_disp)/(disp.molt+1)
        
    fact=1.
    if v0_flag:
       if fixpar:
           v0_qm=new_volume(298.15,0.0001,fix=fix_value)
       else:
           v0_qm=new_volume(298.15,0.0001)   
           
       fact=((1e25*zu*v0_value)/avo)/v0_qm[0]
     
    vol=vol*fact    
    gibbs_pt=(avo/zu)*(f_energy*conv) + (avo/zu)*pp*vol*1e-21
    
    return gibbs_pt

def gibbs_tp(tt,pp,**kwargs):
    l_arg=list(kwargs.items())
    fixpar=False
    v0_flag=False
    g0_flag=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
       elif 'g0' == karg_i[0]:
          g0_flag=True
          gexp=karg_i[1]
       elif 'v0' == karg_i[0]:
          v0_flag=True
          v0_value=karg_i[1]
    
    if v0_flag:      
       if fixpar:
          gref=gibbs_p(298.15,0.,fix=fix_value,v0=v0_value)
       else:
          gref=gibbs_p(298.15, 0.,v0=v0_value)
    else:
       if fixpar:
          gref=gibbs_p(298.15,0.,fix=fix_value)
       else:
          gref=gibbs_p(298.15, 0.)
          
    if g0_flag:      
       gref=gref-gexp        
    
    if v0_flag:        
       if fixpar:
          gtp=gibbs_p(tt,pp,fix=fix_value,v0=v0_value)
       else:
          gtp=gibbs_p(tt,pp,v0=v0_value)
    else:
       if fixpar:
          gtp=gibbs_p(tt,pp,fix=fix_value)
       else:
          gtp=gibbs_p(tt,pp)
        
    return gtp-gref
               
        
def gibbs_serie_p(pini, pfin, npres, tt, prt=True, **kwargs):
    
    """
    Gibbs free energy in a pressure interval, at a given temperature
    
    Args:
        pini:    minimum pressure (GPa)
        pfin:    maximum pressure (GPa)
        npres:   number of points in the interval
        tt:      temperature (K)
        prt (optional): if True, prints a numerical table G(P)
        
    Keyword Args:
        fix: Kp fixed, if fix=Kp > 0.1 
        g0:  Experimental G at the reference T and P (J/mol)
        v0:  Experimental V at the reference T and P (J/mol bar)
        
    Returns: 
        Gibbs free energy in J/mol
    
    Note: 
        the free energy is given with reference to the energy
        at the standard state (298.15 K; 0 GPa)
    """
    
    l_arg=list(kwargs.items())
    fixpar=False
    g0_flag=False
    v0_flag=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
       elif 'g0' == karg_i[0]:
          g0_flag=True
          gexp=karg_i[1]
       elif 'v0'==karg_i[0]:
          v0_flag=True
          v0_value=karg_i[1]
          
    if fixpar:
        if v0_flag:
           gref=gibbs_p(298.15,0.0001,fix=fix_value,v0=v0_value)
        else:
           gref=gibbs_p(298.15,0.0001,fix=fix_value)
    else:
        if v0_flag:
           gref=gibbs_p(298.15,0.0001,v0=v0_value)
        else:
           gref=gibbs_p(298.15,0.0001)
           
        
    if g0_flag:      
       gref=gref-gexp     
          
    p_list=np.linspace(pini,pfin, npres)
    g_list=np.array([])
    
    if not v0_flag:
        if fixpar:
           for pi in p_list:
             gi=gibbs_p(tt,pi,fix=fix_value)
             g_list=np.append(g_list,gi)
        else:
           for pi in p_list:
             gi=gibbs_p(tt,pi)
             g_list=np.append(g_list,gi)
    else:
        if fixpar:
           for pi in p_list:
             gi=gibbs_p(tt,pi,fix=fix_value,v0=v0_value)
             g_list=np.append(g_list,gi)
        else:
           for pi in p_list:
             gi=gibbs_p(tt,pi,v0=v0_value)
             g_list=np.append(g_list,gi)        
        
    g_list=g_list-gref
    
    maxg=max(g_list)*0.99
    ming=min(g_list)*1.01
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.title.set_text("Gibbs free energy vs P\n")
    ax.plot(p_list,g_list,"k-")
    ax.axis([pini,pfin,ming,maxg])
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))   
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("G (J/mol)")
    plt.show()
    
    if prt:
       print("\nPress (GPa)  Gibbs energy (J/mol)\n" )
       for ip,ig in zip(p_list,g_list):
           print("  %5.2f        %8.1f" % (ip, ig))
    
        
def gibbs_serie_t(tini, tfin, ntemp, pp, prt=True, **kwargs):
    
    """
    Gibbs free energy in a temperature interval, at a given pressure
    
    Args:
        tini:    minimum temperature (K)
        tfin:    maximum temperature (K)
        ntemp:   number of points in the interval
        pp:      pressure (GPa)
        prt (optional): if True, prints a numerical table G(Y)
        
    Keyword Args:
        fix: Kp fixed, if fix=Kp > 0.1
        g0:  Experimental G at the reference T and P (J/mol)
        v0:  Experimental V at the reference T and P (J/mol bar)
        
    Returns: 
        Gibbs free energy in J/mol
    
    Note: 
        The free energy is given with reference to the energy
        at the standard state (298.15 K; 0 GPa)
    """
    
    l_arg=list(kwargs.items())
    fixpar=False
    g0_flag=False
    v0_flag=False
    
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
       elif 'g0' == karg_i[0]:
          g0_flag=True
          gexp=karg_i[1]
       elif 'v0'==karg_i[0]:
          v0_flag=True
          v0_value=karg_i[1]    
             
    if fixpar:
        if v0_flag:
           gref=gibbs_p(298.15,0.0001,fix=fix_value,v0=v0_value)
        else:
           gref=gibbs_p(298.15,0.0001,fix=fix_value) 
    else:
        if v0_flag:
           gref=gibbs_p(298.15,0.0001,v0=v0_value)
        else:
           gref=gibbs_p(298.15,0.0001)   
           
    if g0_flag:      
       gref=gref-gexp
    
    t_list=np.linspace(tini,tfin, ntemp)
    g_list=np.array([])
    
    if not v0_flag:       
        if fixpar:
           for ti in t_list:
             gi=gibbs_p(ti,pp,fix=fix_value)
             g_list=np.append(g_list,gi)
        else:
           for ti in t_list:
             gi=gibbs_p(ti,pp)
             g_list=np.append(g_list,gi)
    else:
        if fixpar:
           for ti in t_list:
             gi=gibbs_p(ti,pp,fix=fix_value,v0=v0_value)
             g_list=np.append(g_list,gi)
        else:
           for ti in t_list:
             gi=gibbs_p(ti,pp, v0=v0_value)
             g_list=np.append(g_list,gi)        
         
    g_list=g_list-gref
    
    maxg=max(g_list)*0.999
    ming=min(g_list)*1.001
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.title.set_text("Gibbs free energy vs T\n")
    ax.plot(t_list,g_list,"k-")
    ax.axis([tini,tfin,ming,maxg])
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))   
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("G (J/mol)")
    plt.show()
    
    if prt:
        print("\nTemp  (K)   Gibbs energy (J/mol)\n" )
        for it,ig in zip(t_list,g_list):
            print("  %6.2f        %8.1f" % (it, ig))
             

def eos_temp(tt,prt=True,update=False,kp_only=False):
    """
    Outputs the EoS (BM3) at a given temperature
    
    Args:
        tt: temperature (K)
        prt (optional): if prt=True (default) plots the F(V) function
                        and a list o volume/pressure at the chosen
                        temperature
                        
    Note: 
        In the optimization, Kp can be kept fixed to the value
        set by the set_fix function
    """
    volb=data_vol_freq
    if flag_poly.flag:
        volb=flag_poly.fit_vol
    elif flag_spline.flag:
        volb=flag_spline.fit_vol
    [free_energy, pterm,pcov_term]=bmx_tem(tt)
    k_gpa=pterm[1]*conv/1e-21
    kp=pterm[2]
    v0=pterm[0]
    
    info.temp=tt
    info.k0=k_gpa
    info.kp=kp
    info.v0=v0
    
    if kp_only:
       return 
    
    if bm4.flag:
        ks=pterm[3]
    perr_t=np.sqrt(np.diag(pcov_term))
    if f_fix.flag:
        perr_t[2]=0.00
    ke=perr_t[1]*conv/1e-21
    if bm4.flag:
        print("\n ** BM4 fit **")
    else:
        print("\n ** BM3 fit **")
    print("\nEoS at the temperature of %5.2f K" % tt)
    print("\nBulk Modulus: %5.2f   (%4.2f) GPa" % (k_gpa, ke))
    print("Kp:            %5.2f   (%4.2f)" % (kp, perr_t[2]))
    if bm4.flag:
       print("Kpp            %5.2f   (%4.2f)" % (ks, perr_t[3]))
    print("V0:           %7.4f (%5.3f) A^3\n" % (v0, perr_t[0]))
    fix_status()
    fit_status()
    if update:
        return v0, k_gpa, kp
    if not prt:
        print("\n")
        return 
    vol_min=np.min(volb)
    vol_max=np.max(volb)
    nvol=pr.nvol_eos
    vol_range=np.linspace(vol_min,vol_max,nvol)
    plt.figure(1)
    plt.title("F(V) curve at T= %5.2f K" % tt)
    plt.plot(volb, free_energy, "*")
    if bm4.flag:
        plt.plot(vol_range, bm4.energy(vol_range, *pterm),'b-')
    else:        
        plt.plot(vol_range, v_bm3(vol_range, *pterm), 'b-')
    plt.xlabel("V (A^3)")
    plt.ylabel("F (a.u.)")
    plt.show()
    print("\nVolume-Pressure list at %5.2f K\n" % tt)
    for vp_i in volb:
        if bm4.flag:
           pv_i=bm4.pressure(vp_i,v0,k_gpa,kp,ks)
        else:
           pv_i=bm3(vp_i,v0,k_gpa,kp)
        print(" %5.3f   %5.2f" % (vp_i, pv_i))


def eosfit_dir(file_name, unit=False): 
    """
    Writes a PVT file to be used with EosFit
    Temperature data are in the temperature_list list
    
    Args: 
        file_name: name of the output file
        unit: if unit=True, volumes are converted in cm^3/mol
    
    Example:
       >>> eosfit_dir("myfile.dat")
 
    enclose the file name in quotes.
    
    Note:
       The computation of P(V,T) is performed without reference to 
       any EoS, as pressure at (V,T) is computed as numerical 
       derivative of F with respect to V at constant temperature.
    """
    
    file_name=path+'/'+file_name
    
    if not flag_eos:
       print("\nWarning: set of temperatures for EoSFit output not defined")
       print("Use TEMP keyword in input file")
       return

    if (not flag_poly.flag) and (not flag_spline.flag):
        war1="Warning: frequency fit is off; use of poly or spline fits"
        war2=" is mandatory for bulk_dir"
        print(war1+war2)
        return
    
    flag_volume_max.value=False
    
    if flag_poly.flag:
        volb=flag_poly.fit_vol
    elif flag_spline.flag:
        volb=flag_spline.fit_vol
      
    volmin=min(volb)
    
    eos_data=np.array([])
    for ti in temperature_list:
        volmax=volume_dir(ti,0.)
        if flag_volume_max.value:
            print("Warning: volume exceeds maximum set in volume_range")
            print("Temperature %4.2f, Volume %8.4f" % (ti, volmax))
            continue
        volnew=np.linspace(volmin,volmax,16)
        for vi in volnew:
            pi=pressure_dir(ti,vi)
            if supercell.flag:
                vi=vi/supercell.number
            if unit:
                vi=vi*1e-24*avo/zu
            if pi >=-0.02:
               pvt=np.array([pi, vi, ti])
               eos_data=np.append(eos_data,pvt)
    
    iraw=np.int(eos_data.size/3)
    eosdata=np.reshape(eos_data,(iraw,3))
    string='TITLE  Input prepared with Numpy script\nformat 1 P  V  T'
    np.savetxt(file_name, eosdata, fmt="%5.3f %12.4f %8.2f", \
               header=string, comments="")
    print("\nEoSFit file %s saved" % file_name)
                      

def eosfit(file_name,**kwargs):
    """
    Writes a PVT file to be used with EosFit
    Temperature data are in the temperature_list list
    
    Args:
        file_name: name of the output file
        
    Keyword Args:
        if the optional argument 'fix' is larger than 0.1, Kp=fix is fixed
    
    Example:
       >>> eosfit("myfile.dat")
 
    enclose the file name in quotes
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if not flag_eos:
        print("\nWarning: set of temperatures for EoSFit output not defined")
        print("Use TEMP keyword in input file")
        return None
    volb=data_vol_freq_orig
    if flag_poly.flag:
        volb=flag_poly.fit_vol
    elif flag_spline.flag:
        volb=flag_spline.fit_vol
    file_name=path+'/'+file_name
    maxvol=max(volb)
    npoint=volb.size
    eos_data=np.array([])
    for itt in temperature_list:
        volt=new_volume(itt,0.)
        if volt >=maxvol:
            print("Temperature %4.2f not considered" % itt)
            print("Equilibrium volume (%6.3f) exceeds the volume range"\
                   % volt)
            break
        volb_t=np.linspace(min(volb),volt,npoint)
        if fixpar:
           [ff,pterm,pcov]=bmx_tem(itt,fix=fix_value)
        else:
           [ff,pterm,pcov]=bmx_tem(itt) 
        k_gpa=pterm[1]*conv/1e-21
        kp=pterm[2]
        v0=pterm[0]
        if bm4.flag:
            ks=pterm[3]
        for vi in volb_t:
            if bm4.flag:
               pv_i=bm4.pressure(vi,v0,k_gpa,kp,ks)
            else:
               pv_i=bm3(vi,v0,k_gpa,kp)
            if pv_i>=0.:
                pvt=np.array([pv_i, vi, itt])
                eos_data=np.append(eos_data,pvt)
    iraw=np.int(eos_data.size/3)
    eosdata=np.reshape(eos_data,(iraw,3))
    string='TITLE  Input prepared with Numpy script\nformat 1 P  V  T'
    np.savetxt(file_name, eosdata, fmt="%5.2f %12.4f %8.2f", \
               header=string, comments="")
    print("EoSFit file %s saved" % file_name)
    


def new_volume(tt,pr,**kwargs):
    """
    Computes the volume (A^3) at a given pressure and temperature
    
    Args:
        tt: temperature (K)
        pp: pressure (GPa)
        
    Keyword Args:
        fix (optional): used to keep Kp fixed 
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
    if fixpar:      
       [free_energy, pterm,pcov_term]=bmx_tem(tt,fix=fix_value)
    else:
       [free_energy, pterm,pcov_term]=bmx_tem(tt)
    
    k0=pterm[1]*conv/1e-21
    kp=pterm[2]
    v0=pterm[0]
    if bm4.flag:
       p_fun=lambda vv: ((bm4.pressure(vv,pterm[0],pterm[1],pterm[2],pterm[3])*\
                         conv/1e-21)-pr)**2
    else:
       p_fun=lambda vv: ((3*k0/2)*((v0/vv)**(7/3)-(v0/vv)**(5/3))* \
                     (1+(3/4)*(kp-4)*((v0/vv)**(2/3)-1))-pr)**2
           
    vol=scipy.optimize.minimize(p_fun,v0,tol=1e-18)
    return vol.x
                       
def freq_v_fit(ifr):
    """
    Computes the coefficients of the polynomium fitting the frequency
    of the "ifr" mode with respect to volume; the degree of the fitting
    polynomium ("dg") is specified in the input.txt file, under
    the keyword FITVOL, or it can be set by the set_poly function.
    """
    ifl=np.array([])
    dg=flag_poly.degree
    new_vol_range=np.array([])
    vmin, vmax=min(flag_poly.fit_vol), max(flag_poly.fit_vol)   
    for ivol in int_set:
        if (data_vol_freq_orig[ivol] >= vmin) and (data_vol_freq_orig[ivol] <= vmax):
           ifl=np.append(ifl,lo.data_freq[ifr,ivol+1])
           new_vol_range=np.append(new_vol_range, data_vol_freq_orig[ivol])
    pol=np.polyfit(new_vol_range,ifl,dg)
   
    return pol


def freq_stack_fit():
    """
    Accumulates all the coefficients of the polynomia fitting the
    frequencies of all the modes, computed by freq_v_fit. 
    Outputs the array "pol_stack" used by freq_v_fun
    """
    pol_stack=np.array([])
    dg=flag_poly.degree
    for ifr in int_mode:
        pol_i=freq_v_fit(ifr)
        pol_is=np.array(pol_i)
        pol_stack=np.append(pol_stack, pol_is)
    pol_stack=pol_stack.reshape(int_mode.size,dg+1)
    return pol_stack


def freq_v_fun(ifr,vv):
    """
    Outputs the frequency of the "ifr" mode as a function of volume
    by using the polynomial fit computed with freq_v_fit    
    """
    if not flag_poly.flag_stack:
        print("Polynomial stack not present; use set_poly to create it")
        return
    pol_stack=flag_poly.pol_stack
    pol=pol_stack[ifr,:]
    return np.polyval(pol,vv)
     
# Spline section
def freq_spline(ifr):
    ifl=np.array([])
    degree=flag_spline.degree
    smooth=flag_spline.smooth
    new_vol_range=np.array([])
    vmin, vmax=min(flag_spline.fit_vol), max(flag_spline.fit_vol)
    for ivol in int_set:
        if (data_vol_freq_orig[ivol] >= vmin) and (data_vol_freq_orig[ivol] <= vmax):
          ifl=np.append(ifl,lo.data_freq[ifr,ivol+1])
          new_vol_range=np.append(new_vol_range, data_vol_freq_orig[ivol])
    f_uni=UnivariateSpline(new_vol_range,ifl,k=degree,s=smooth)
    return f_uni

def freq_stack_spline():
    pol_stack=np.array([])
    for ifr in int_mode:
        pol_i=freq_spline(ifr)
        pol_stack=np.append(pol_stack, pol_i)
    pol_stack=np.array(pol_stack)
    return pol_stack

def freq_spline_v(ifr,vv):
    if not flag_spline.flag_stack:
        print("Spline stack not present; use set_spline to create it")
        return
    return flag_spline.pol_stack[ifr](vv)


def freq_poly_p(ifr,tt=300., p0=0., plot=True, prt=True, **kwargs):
    """
    Prints the frequency of a given mode at some temperature
    and pressure if a spline fitting method has been chosen
    
    Args:
        ifr: mode number (starting from 0)
        tt: temperature (K)
        pp: pressure (GPa)
        
    Keyword Args::
        fix (optional): Kp value fixed to *fix* if *fix* > 0.1
     
    Note:
       A polynomial fitting must be active
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if not flag_poly.flag:
        print("Polinomial fit not active: use set_poly to active it")
        return
    pol_stack=flag_poly.pol_stack
    degree=flag_poly.degree
    if fixpar:
       vpt=new_volume(tt,p0,fix=fix_value)
    else:
       vpt=new_volume(tt,p0)
    f0=np.polyval(pol_stack[ifr],vpt)
    ifl=np.array([])
    vol_min=np.min(data_vol_freq)
    vol_max=np.max(data_vol_freq)
    nvol=pr.nvol
    vol_range=np.linspace(vol_min,vol_max,nvol)
    for ivol in int_set:
        ifl=np.append(ifl,lo.data_freq[ifr,ivol+1])
    pol=np.polyfit(data_vol_freq,ifl,degree)
    ivalf=np.polyval(pol,vol_range)
    if prt:
       if plot:
          plt.figure(2)
          plt.plot(data_vol_freq,ifl,"*")
          plt.plot(vol_range,ivalf,"b-")
          plt.xlabel("V (A^3)")
          plt.ylabel("Freq (cm^-1)")
          plt.show()
    if prt:
       print("\nFrequency: %6.2f cm^-1" % f0)
       print("Pressure %4.2f GPa, temperature %5.2f K, " \
             "volume %8.4f A^3" % (p0, tt, vpt[0]))
       return
    else:
       return f0
   
def freq_spline_p(ifr,tt=300.,pp=0.,prt=True,**kwargs):
    """
    Prints the frequency of a given mode at some temperature
    and pressure if a spline fitting method has been chosen
    
    Args:
        ifr: mode number (starting from 0)
        tt: temperature (K)
        pp: pressure (GPa)
        fix (optional): Kp value fixed to *fix* if *fix* > 0.1
        
    A spline fitting must be active
    """
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if not flag_spline.flag:
        print("Spline fit not active: use set_spline to active it")
        return   
    if fixpar:
       volp=new_volume(tt,pp,fix=fix_value)
    else:
       volp=new_volume(tt,pp) 
    fr=freq_spline_v(ifr,volp)
    vol=np.linspace(min(data_vol_freq),max(data_vol_freq),pr.nvol)
    freq=flag_spline.pol_stack[ifr](vol)
    ifl=[]
    for ivol in int_set:
       ifl=np.append(ifl,lo.data_freq[ifr,ivol+1])
    
    if prt:
       plt.figure()
       plt.plot(vol,freq,"k-")
       plt.plot(data_vol_freq,ifl,"k*")
       plt.xlabel("Volume (A^3)")
       plt.ylabel("Frequency (cm^-1)")
       plt.show()
       print("Frequency: %5.2f" % fr)
       print("Pressure: %4.2f GPa, temperature %5.2f K, volume %8.4f A^3" %\
          (pp, tt, volp))
    else:
        return fr
    

def check_spline(ifr):
    if not flag_spline.flag:
        print("Spline fit not active: use set_spline")
        return
    vol=np.linspace(min(data_vol_freq),max(data_vol_freq),pr.nvol)
    freq=flag_spline.pol_stack[ifr](vol)
    ifl=[]
    for ivol in int_set:
       ifl=np.append(ifl,lo.data_freq[ifr,ivol+1])
    plt.figure()
    leg="Mode number "+str(ifr)
    if ifr in exclude.ex_mode:
        leg=leg+"\nExcluded from free energy computation"
    plt.plot(vol,freq,"k-")
    plt.plot(data_vol_freq,ifl,"k*")
    plt.xlabel("Volume (A^3)")
    plt.ylabel("Frequency (cm^-1)")
    plt.title(leg)
    plt.show()
    
def check_poly(ifr):
    if not flag_poly.flag:
        print("Polynomial fit not active: use set_poly")
        return
    pol_stack=flag_poly.pol_stack
    vol=np.linspace(min(data_vol_freq),max(data_vol_freq),pr.nvol)
    freq=np.polyval(pol_stack[ifr],vol)
    ifl=[]
    for ivol in int_set:
       ifl=np.append(ifl,lo.data_freq[ifr,ivol+1])
    plt.figure()
    leg="Mode number "+str(ifr)
    if ifr in exclude.ex_mode:
        leg=leg+"\n Excluded from free energy computation"
    plt.plot(vol,freq,"k-")
    plt.plot(data_vol_freq,ifl,"k*")
    plt.xlabel("Volume (A^3)")
    plt.ylabel("Frequency (cm^-1)")
    plt.title(leg)
    plt.show()
    
def check_spline_total():
    """
    Plots the frequencies of all the modes as a function of
    volumes along with their fitting according to the
    spline parameters chosen. 
    
    The fitting is restricted to the volume range set by the 
    set_volume_range function.
    """
    for ifr in int_mode:
        check_spline(ifr)
        
def check_spline_list(list_of_modes):
    """
    Plots the frequencies of a given list of normal modes as functions
    of volumes, along with their fitting according to the spline
    parameters chosen.
    
    Args: list_of_modes (a list of integers)
    
    Example:
        >>> check_spline_list([0, 1, 2])
        
    """
    
    for ifr in list_of_modes:
        check_spline(ifr)
        
def check_poly_total():
    """
    Plots the frequencies of all the modes as a function of
    volumes along with their fitting according to the
    polynomial degree chosen.
    
    The fitting is restricted to the volume range set by the 
    set_volume_range function.
    """
    for ifr in int_mode:
        check_poly(ifr)
        
def check_poly_list(list_of_modes):
    """
    Plots the frequencies of a given list of normal modes
    
    Args: 
        list_of_modes (a list of integers)
    
    Example:
        >>> check_poly_list([0, 1, 2])
    """
    
    for ifr in list_of_modes:
        check_poly(ifr)
    
def pressure_freq_list(tt,ifr,**kwargs):
    
    if (not flag_poly.flag) and (not flag_spline.flag):
        msg1='** Error ** This function can be used only in connection with '
        msg2='a fitting of frequencies;\n'
        msg3=' '*12 +'POLY or SPLINE must be used'
        print(msg1+msg2+msg3)
        return
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    pmin=0.
    volmin=min(data_vol_freq)
    if fixpar:
       pmax=pressure(tt,volmin,fix=fix_value)
    else:
       pmax=pressure(tt,volmin)
       
    p_list=np.linspace(pmin, pmax, pr.npres)
    
    freq_list=np.array([])
    for p_i in p_list:
        if flag_spline.flag:
            if fixpar:
               f_i=freq_spline_p(ifr,tt,p_i,prt=False,fix=fix_value)
            else:
               f_i=freq_spline_p(ifr,tt,p_i,prt=False)
        else:
            if fixpar:
               f_i=freq_poly_p(ifr,tt,p_i,prt=False,fix=fix_value)
            else:
               f_i=freq_poly_p(ifr,tt,p_i,prt=False) 
               
        freq_list=np.append(freq_list,f_i)
        
    return freq_list, p_list

def pressure_freq(ifr,freq,tt,degree=4,**kwargs):
    """
    Computes the pressure given the frequency of a normal mode, at a fixed
    temperature.
    
    Args:
        ifr:               normal mode number
        freq:              value of the frequency
        tt:                temperature
        degree (optional): degree of the polynomial fitting the P/freq
                           values from the pressure_freq_list function
        
    Keyword Args: 
        fix: Kp fixed, if fix=Kp > 0.1
        
    Notes: 
        it is advised to keep Kp fixed by either specifying fix, or
        by using set_fix.
        
        For "noisy" modes, use polynomial fits (set_poly), or 
        a spline fit (set_spline) with a large smooth parameter.
    """
    
    if (not flag_poly.flag) and (not flag_spline.flag):
        msg1='** Error ** This function can be used only in connection with '
        msg2='a fitting of frequencies;\n'
        msg3=' '*12 +'POLY or SPLINE must be used'
        print(msg1+msg2+msg3)
        return
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
          
    if fixpar:
        freq_list, p_list=pressure_freq_list(tt,ifr,fix=fix_value)
    else:
        freq_list, p_list=pressure_freq_list(tt,ifr)
        
    plt.figure()
    plt.plot(freq_list, p_list,"k*")
    plt.ylabel("P (GPa)")
    plt.xlabel("Freq (cm^-1)")
    
    f_opt=np.polyfit(freq_list,p_list,deg=degree)
    pres_f=np.polyval(f_opt,freq)
    
    plt.plot(freq_list, p_list,"k-")
    plt.show()
    
    print("Pressure: %6.2f GPa" % pres_f)
    print("Mode: %d, frequency: %6.2f cm^-1, temperature: %6.2f K" %\
          (ifr, freq, tt))
    
    
    
def temperature_freq(ifr,freq, tmin, tmax, npt, pp,degree=2,**kwargs):
    """
    Computes the temperature given the frequency of a normal mode, at a fixed
    pressure. A T range must be specified
    
    Args:
        ifr:               normal mode number
        freq:              value of the frequency
        tmin:              minimum value of T
        tmax:              maximum value of T
        npt:               number of T points in the range
        pp:                pressure
        degree (optional): degree of the polynomial fitting the P/freq
                           values from the pressure_freq_list function
        
    Keyword Args: 
            fix: Kp fixed, if fix=Kp > 0.1
        
    Note: 
        it is advised to keep Kp fixed by either specifying fix, or
        by using set_fix.
        
        For "noisy" modes, use polynomial fits (set_poly), or 
        a spline fit (set_spline) with a large smooth parameter.
    """
    
    if (not flag_poly.flag) and (not flag_spline.flag):
        msg1='** Error ** This function can be used only in connection with '
        msg2='a fitting of frequencies;\n'
        msg3=' '*12 +'POLY or SPLINE must be used'
        print(msg1+msg2+msg3)
        return
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
        
    freq_list=np.array([])
    t_list=np.linspace(tmin,tmax,npt)
    if fixpar:
        for ti in t_list:
            nvi=new_volume(ti,pp,fix=fix_value)
            if flag_spline.flag:
                fi=flag_spline.pol_stack[ifr](nvi)
            else:
                fi=np.polyval(flag_poly.pol_stack[ifr],nvi)
            freq_list=np.append(freq_list,fi)               
    else:
        for ti in t_list:
            nvi=new_volume(ti,pp)
            if flag_spline.flag:
                fi=flag_spline.pol_stack[ifr](nvi)
            else:
                 fi=np.polyval(flag_poly.pol_stack[ifr],nvi)
            freq_list=np.append(freq_list,fi)  
        
    plt.figure()
    plt.plot(freq_list, t_list,"k*")
    plt.ylabel("T (K)")
    plt.xlabel("Freq (cm^-1)")
    
    f_opt=np.polyfit(freq_list,t_list,deg=degree)
    temp_f=np.polyval(f_opt,freq)
    
    plt.plot(freq_list, t_list,"k-")
    plt.show()
    
    print("Temperature: %6.2f K" % temp_f)
    print("Mode: %d, frequency: %6.2f cm^-1, Pressure: %6.2f GPa" %\
          (ifr, freq, pp))
    
        
def grun_mode_vol(ifr,vv, method='poly',plot=False):
    """
    Mode-gamma Gruneisen parameter of a normal mode at a given volume
    
    Args:
        ifr: mode number
        vv:  volume
        method (optional): method chosen for the frequency/volume values
                           (default: 'poly'; other possible method='spline')
        plol (optional):   if not False (default), plots the frequency values
                           of the selected mode in a neighborhood of the
                           volume vv
                           
    Returns:
        Mode-gamma Gruneisen parameter and the frequency of the mode at the 
        volume vv
    """
    v_range=np.linspace(vv-pr.delta_v,vv+pr.delta_v,pr.nump_v)
    f_list=np.array([])
    for iv in v_range:
        if flag_poly.flag_stack and method=='poly':
            f_i=freq_v_fun(ifr,iv)
        elif flag_spline.flag_stack and method=='spline':
            f_i=freq_spline_v(ifr,iv)
        else:
            print("No fitting stack present for the method chosen")
            print("Create the appropriate stack by using set_poly or set_spline")
            return
        f_list=np.append(f_list,f_i)
    fit_f=np.polyfit(v_range,f_list,pr.degree_v)
    derv=np.polyder(fit_f,1)
    derf=np.polyval(derv,vv)
    ffit=np.polyval(fit_f,vv)
    
    if plot:
       v_fit_list=np.linspace(vv-pr.delta_v,vv+pr.delta_v,40)
       f_poly_list=np.polyval(fit_f,v_fit_list)
       fig=plt.figure()
       ax=fig.add_subplot(111)
       ax.plot(v_range,f_list,"k*")
       ax.plot(v_fit_list,f_poly_list,"k-")
       ax.set_xlabel("V (A^3)")
       ax.xaxis.set_major_locator(plt.MaxNLocator(5))
       ax.set_ylabel("Freq (cm^-1)")
       plt.show()
    return -1*vv*derf/ffit,ffit

def gruneisen(vol, method='poly',plot=True):
    """
    Mode-gamma Gruneisen parameter of all the normal modes at a given volume
    
    Args:
        vv:  volume
        method (optional): method chosen for the frequency/volume values
                           (default: 'poly'; other possible method='spline')
        plot (optional):   if True (default), plots the mode-gamma Gruneisen
                           parameters of all the modes
                           
    Returns:
        if plot=False, outputs the list of the mode-gamma Gruneisen parameters
        of all the modes
    """
    
    if method=='poly' and not flag_poly.flag_stack:
        print("Polynomial stack not present; use set_poly to create")
        return
    if method=='spline' and not flag_spline.flag_stack:
        print("Spline stack not present; use set_spline to create")
        return        
        
    grun_list=np.array([])
    freq_list=np.array([])
    for im in int_mode:
        g_i,f_i=grun_mode_vol(im,vol,method=method,plot=False)
        grun_list=np.append(grun_list, g_i)
        freq_list=np.append(freq_list, f_i)
    if plot:
       fig=plt.figure()
       ax=fig.add_subplot(111)
       ax.plot(freq_list,grun_list,"k*")
       ax.set_xlabel("Mode frequency")
       ax.set_ylabel("Gruneisen")
       plt.show()   
    
    if not plot:
       return grun_list
   
def gruneisen_therm(tt,pp,ex_data=False,prt=True,**kwargs):
    
    """
    Gruneisen parameter: alpha*K_S / rho*Cp
    
    Args:
        tt:  temperature
        pp:  pressure
        
    Keyword Args:
        fix: Kp fixed, if fix=Kp > 0.1   
    """
    
    l_arg=list(kwargs.items())
    fixpar=False
    for karg_i in l_arg:
       if 'fix' == karg_i[0]:
          fix_value=karg_i[1]
          fixpar=True
       
    if fixpar:    
       vol=new_volume(tt,pp, fix=fix_value)
    else:
       vol=new_volume(tt,pp)
       
    volume=(vol/zu)*avo*1e-30      # volume of a mole in m^3
    density=1/volume               # density in mole/m^3
 
    if fixpar:
        ent,cv=entropy_v(tt,vol,fix=fix_value)
        alp,k,p=thermal_exp_v(300,vol,fix=fix_value)
    else:
        ent,cv=entropy_v(tt,vol)
        alp,k,p=thermal_exp_v(300,vol)
        
    grun_th=alp*k*1e9/(density*cv)
    
    if prt:
      print("\nGruneisen parameter (adimensional): %6.3f\n" % grun_th)
      print("Thermal expansion: %6.2e (K^-1)" % alp)
      print("Bulk modulus: %6.2f (GPa)" % k)
      print("Specific heat at constant volume: %6.2f (J/mol K)" % cv) 
    
    if ex_data:
       return volume[0],cv,alp[0],k[0],grun_th[0]
    
def grun_therm_serie(tini,tfin,npoint=12,HTlim=2000,degree=1,g_deg=1, ex=False):
    
    print("\n---- High temperature estimation of the thermal expansion coefficient ----\n")
    v0, k_gpa, kp=eos_temp(298.15,prt=False, update=True)
    set_fix(kp)
    
    vol=new_volume(298.15,0.0001)
    ent, cve=entropy_v(298.15,vol[0])
    dp_limit=apfu*3*avo*kb                  # Dulong Petit limit
    emp=10636/(ent/apfu+6.44)               # Empirical Einstein T
    
    print("\nDulong-Petit limit of Cv %6.2f (J/mol K)" % dp_limit)
    print("Einstein's temperature: %6.2f (K)" % emp)
    
    t_list=np.linspace(tini,tfin,npoint)
    v_list=np.array([])
    cv_list=np.array([])
    k_list=np.array([])
    alp_list=np.array([])
    for it in t_list:
        iv,icv,ialp,ik,igr=gruneisen_therm(it,0,ex_data=True,prt=False)        
        v_list=np.append(v_list,iv)
        cv_list=np.append(cv_list,icv)
        k_list=np.append(k_list,ik)
        alp_list=np.append(alp_list,ialp)
 
    if not gamma_fit.flag:
       pol=gamma_estim(tini,tfin,npoint,g_deg)
    else:
       pol=gamma_fit.pol 
       print("Gamma(V) fit from already stored data")
       
    grun_list=np.polyval(pol,t_list)

    fact_list=1e-9*grun_list/(k_list*v_list)
    f_coef=np.polyfit(t_list,fact_list,degree)
    fact_calc=np.polyval(f_coef,t_list)
            
    plt.figure()
    plt.plot(t_list,fact_list*1e9,"*")    
    plt.plot(t_list,fact_calc*1e9)
    plt.xlabel("Temperature (K)")
    plt.ylabel("J^-1")
    plt.title("Gamma/VK_T")
    plt.show()   
    
    fact_lim=np.polyval(f_coef,HTlim)
    alpha_limit=dp_limit*fact_lim   
    
    print("\nGamma/VK fit of degree %1i" % degree)
    print("Alpha constrained at the high temperature limit: %6.2e K^-1" % alpha_limit)
    print("\n -------------------------------")
    if ex:
       return alpha_limit
    
    
def number_phonon_mode(ifr,tt,vol,method='poly'):
    """
    Number of phonons of a given mode at given temperature and volume
    
    Args:
        ifr:               mode number
        tt:                temperature
        vv:                volume
        method (optional): method chosen for the frequency/volume values
                           (default: 'poly'; other possible method='spline')
                           
    Returns:
        Number of phonons computed according to the Bose-Einstein statistics
    """
    
    if method=='poly' and not flag_poly.flag_stack:
        print("Polynomial stack not present; use set_poly to create")
        return
    if method=='spline' and not flag_spline.flag_stack:
        print("Spline stack not present; use set_spline to create")
        return
        
    if method=='poly':
        f_i=freq_v_fun(ifr,vol)
    else:
        f_i=freq_spline_v(ifr,vol)
         
    f_i=csl*f_i
    exp_fact=np.exp(h*f_i/(kb*tt))
    return 1./(exp_fact-1),vol
        
def pressure_phonon_mode(ifr,tt,vol,method='poly'):
    """
    Contribution to the vibrational pressure from a given mode, at fixed
    temperature and volume
    
    Args:
        ifr:               mode number
        tt:                temperature
        vv:                volume
        method (optional): method chosen for the frequency/volume values
                           (default: 'poly'; other possible method='spline')
                           
    Returns:
        Vibrational pressure of the "ifr" mode (in GPa) at the selected
        temperature (tt) and volume (vv)
    
    """
    
    if method=='poly' and not flag_poly.flag_stack:
        print("Polynomial stack not present; use set_poly to create")
        return
    if method=='spline' and not flag_spline.flag_stack:
        print("Spline stack not present; use set_spline to create")
        return

    nph,vol=number_phonon_mode(ifr,tt,vol,method=method) 
    g_i,f_i=grun_mode_vol(ifr,vol,method=method,plot=False)
       
    pz_i=(1./(2*vol*1e-21))*h*(f_i*csl)*g_i
    pth_i=(1./(vol*1e-21))*nph*h*(f_i*csl)*g_i
    p_total_i=(pz_i+pth_i)*deg[ifr]
       
    return p_total_i

def pressure_phonon(tt,vol,method='poly',plot=True):
    """
    Vibrational pressure from all the normal modes at given temperature
    and volume
    
    Args:
        tt:                temperature
        vv:                volume
        method (optional): method chosen for the frequency/volume values
                           (default: 'poly'; other possible method='spline')
        plot (optional):   if True (default), plots the contribution to the
                           vibrational pressure of all the normal modes.
                           
    Returns:
        If plot=False, outputs the vibrational pressure of all the modes 
        (in GPa) at the selected temperature (tt) and volume (vv). 
    """
    
    if method=='poly' and not flag_poly.flag_stack:
        print("Polynomial stack not present; use set_poly to create")
        return
    if method=='spline' and not flag_spline.flag_stack:
        print("Spline stack not present; use set_spline to create")
        return
        
    p_list=np.array([])    
    for ifr in int_mode:
            p_total_i=pressure_phonon_mode(ifr,tt,vol,method=method)
            p_list=np.append(p_list,p_total_i)
            
    p_total=p_list.sum()
    
    if plot:        
       plt.figure()
       plt.plot(int_mode,p_list,"r*")
       plt.xlabel("Mode number")
       plt.ylabel("Pressure (GPa)")
       plt.show()
    
       print("\nTotal phonon pressure: %4.2f GPa " % p_total)
    
    if not plot:
       return p_list
    else:
       return

def upload_mineral(tmin,tmax,points=12,HT_lim=0., deg=1, g_deg=1, model=1, mqm='py',\
                   b_dir=False, blk_dir=False, extra_alpha=True, volc=False):
    """
    Prepares data to be uploaded in the mineral database.
    
    Args:
        tmin:   minimum temperature for fit K0, Cp and alpha
        tmax:   maximum temperature for fit
        points: number of points in the T range for fit
        mqm:    name of the mineral, as specified in the internal 
                database,
        b_dir:  if True, the alpha_dir_serie function is used for the
                computation of thermal expansion
        blk_dir:  if True, the bulk_modulus_p_serie function is used
                     to compute the bulk modulus as a function of T 
                     (with noeos=False); K0, V0 and Kp are from an eos_temp
                     computation.
                  If False, the function bulk_serie is used.
        HT_lim: Temperature at which the Dulong-Petit limit for Cv
                is supposed to be reached (default 0.: no Dulong-Petit
                                           model)
        model:  Used in the HT_limit estimation of Cv; Einstein model
                for Cv(T) with one frequency (default model=1), or with
                2 frequencies (model=2)
        deg:    Used in the HT limit estimation of alpha (relevant if
                HT_lim > 0; default: 1)
        g_deg:  Used in the HT limit estimation of Cp (relevant if
                HT_lim > 0.; default 1)
        extra_alpha: if True, the high temperature estrapolation
                     is done (relevant if HT_lim > 0; default: True)
        volc:   if True, V0 is set at the value found in the database
                (default: False)
        
                
    Example:
        >>> upload_mineral(300,800,16,mqm='coe', b_dir=True)
    """
    
    
    flag_int=False
    if f_fix.flag:
       kp_original=f_fix.value
       flag_int=True
       reset_fix()
              
    if not volume_correction.flag:
       volume_correction.set_volume(eval(mqm).v0)
       volume_correction.on()
      
    g0=eval(mqm).g0
    
    if b_dir and blk_dir:
       v0, k_gpa, kp=eos_temp(298.15,prt=False, update=True)
       fit_b,_=bulk_modulus_p_serie(tmin, tmax,5,0, noeos=False, fit=True, deg=1, out=True)
       dkt=fit_b[0]
    else:    
       v0, k_gpa, kp=eos_temp(298.15,prt=False, update=True)
       set_fix(kp)
       fit_b=bulk_serie(tmin,tmax,5,degree=1,update=True)
       dkt=fit_b[0]
    
    if not volc:
       v0=v0*1e-30*1e5*avo/zu 
    else:
       v0=volume_correction.v0_init
        
    ex_internal_flag=False
    if exclude.flag & (not anharm.flag):
        exclude.restore()
        ex_internal_flag=True
        print("\nWarning ** Excluded modes restored in order\nto "
              "correctly compute entropy and specific heat")
        print("At the end of computation, exclusion of modes "
              "will be rectivated")
    s0,dum=entropy_p(298.15,0.0001,prt=False)
    
    if ex_internal_flag:
        exclude.on()
        ex_internal_flag=False
    
    mdl=1
    if model==2:
       mdl=2
    if HT_lim > 0.:
       fit_cp=cp_serie(tmin,tmax,points,0.0001,HTlim=HT_lim, g_deg=g_deg, model=mdl, prt=False)
       if extra_alpha:
          fit_al=alpha_serie(tmin,tmax,points,0.0001,HTlim=HT_lim,degree=deg,prt=False)
       else:
          fit_al=alpha_serie(tmin,tmax,points,0.0001,prt=False)
    else:
       fit_cp=cp_serie(tmin,tmax,points,0.0001, g_deg=g_deg, prt=False)
       if not b_dir:
          fit_al=alpha_serie(tmin,tmax,points,0.0001,prt=False,g_deg=g_deg)
       else:
          fit_al=alpha_dir_serie(tmin,tmax,points,0.0001,prt=False)

    eval(mqm).load_ref(v0,g0,s0)
    eval(mqm).load_bulk(k_gpa,kp,dkt)
    eval(mqm).load_cp(fit_cp,power)
    eval(mqm).load_alpha(fit_al,power_a)
    eval(mqm).eos='bm'
    
    reset_fix()
    if flag_int:
        set_fix(kp_original)
        

def reaction_dir(tt,pp,mqm,prod_spec, prod_coef, rea_spec, rea_coef):
    
    mv0=eval(mqm+'.v0')
    mg0=eval(mqm+'.g0')
    
    qm_energy=g_vt_dir(tt,pp,v0=mv0, g0=mg0)
    
    gprod=0.
    for pri, pci in zip(prod_spec, prod_coef):
        if pri != mqm:
           gprod=gprod+(eval(pri+'.g_tp(tt,pp)'))*pci
        else:
           gprod=gprod+qm_energy*pci
        
    grea=0.    
    for ri,rci in zip(rea_spec, rea_coef):
        if ri != mqm:
           grea=grea+(eval(ri+'.g_tp(tt,pp)'))*rci
        else:
           grea=grea+qm_energy*rci
        
    return gprod-grea    
    
def pressure_react_dir(tt,mqm,prod_spec, prod_coef, rea_spec, rea_coef):

    fpr=lambda pp: (reaction_dir(tt,pp,mqm, \
                    prod_spec, prod_coef, rea_spec, rea_coef))**2
                                                  
    pres=scipy.optimize.minimize_scalar(fpr,tol=0.001)
    return pres.x
       
      
def equilib_dir(tini,tfin,npoint, mqm='py', \
                prod=['py',1], rea=['ens',1.5,'cor', 1]):
    """
    Computes the equilibrium pressure for a reaction involving a
    given set of minerals, in a range of temperatures.
    
    Args:
        tini:   minimum temperature in the range
        tfin:   maximum temperature in the range
        npoint: number of points in the T range
        mqm:    mineral phase dealt at the quantum mechanical level,
                whose Gibbs free energy is computed as G=F+PV
        prod:   list of products of the reaction in the form 
                [name_1, c_name_1, name_2, c_name_2, ...]
                where name_i is the name of the i^th mineral, as stored
                in the database, and c_name_i is the corresponding
                stoichiometric coefficient
        rea:    list of reactants; same syntax as the "prod" list.
        
    Example:
        >>> equilib_dir(300, 500, 12, mqm='py', prod=['py',1], rea=['ens', 1.5, 'cor', 1])
    """   
    lprod=len(prod)
    lrea=len(rea)
    prod_spec=prod[0:lprod:2]
    prod_coef=prod[1:lprod:2]
    rea_spec=rea[0:lrea:2]
    rea_coef=rea[1:lrea:2]

    flag_volume_max.value=False
    
    lastr=rea_spec[-1]
    lastp=prod_spec[-1]
    
    prod_string=''
    for pri in prod_spec:
        prod_string=prod_string + pri
        if pri != lastp:
            prod_string=prod_string+' + '
        
    rea_string=''
    for ri in rea_spec:
        rea_string = rea_string + ri
        if ri != lastr:
            rea_string=rea_string+' + '
    t_list=np.linspace(tini,tfin,npoint)
    p_list=np.array([])
    
    for ti in t_list:
        pi=pressure_react_dir(ti,mqm, prod_spec, prod_coef, rea_spec, rea_coef)
        p_list=np.append(p_list,pi)
        
    p_new=np.array([])
    t_new=np.array([])    
    for pi, ti in zip(p_list, t_list):
        p_new=np.append(p_new,pi)
        t_new=np.append(t_new,ti)
            
    serie=(t_new,p_new)
    pd.set_option('colheader_justify', 'center')
    df=pd.DataFrame(serie, index=['T (K)','P (GPa)'])
    df=df.T
    df2=df.round(2)
    print("")
    print(df2.to_string(index=False))
        
    ymax=max(p_new)+0.1*(max(p_new)-min(p_new))
    ymin=min(p_new)-0.1*(max(p_new)-min(p_new))
    
    xloc_py, yloc_py, xloc_en, yloc_en=field_dir(min(t_new),max(t_new), \
      ymax, ymin, mqm,prod_spec, prod_coef, rea_spec, rea_coef)
    
    print("\n")      
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.title.set_text("Reaction "+ rea_string + " <--> " + prod_string + "\n" )
    ax.text(xloc_en, yloc_en, rea_string)
    ax.text(xloc_py,yloc_py, prod_string)
    ax.plot(t_new,p_new,"k-")
    ax.axis([min(t_new), max(t_new),ymin,ymax])
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (GPa)")
    plt.savefig(fname=path+'/'+"react",dpi=600)
    plt.show()
    
    if flag_volume_max.jwar > 0:
        print("\nWarning on volume repeated %d times" % flag_volume_max.jwar)
        flag_volume_max.reset()
    
def field_dir(tmin,tmax,pmin,pmax,mqm,\
          prod_spec, prod_coef, rea_spec, rea_coef, nx=6, ny=6):
    
    t_range=np.linspace(tmin,tmax,nx)
    p_range=np.linspace(pmin,pmax,ny)
    
    flag_volume_max.value=False
    
    fld=np.array([])
    for ti in t_range:
        for pi in p_range:
            de=reaction_dir(ti,pi,mqm, prod_spec, prod_coef, rea_spec, rea_coef)
            fld=np.append(fld,[ti,pi,de])
            
    fld=fld.reshape(nx*ny,3)
    
    prodx=np.array([])
    prody=np.array([])    
    reax=np.array([])
    reay=np.array([])
    
    for fi in fld:
        if fi[2]>0:
            reax=np.append(reax,fi[0])
            reay=np.append(reay,fi[1])
        else:
            prodx=np.append(prodx,fi[0])
            prody=np.append(prody,fi[1])
            
    return prodx.mean(), prody.mean(), reax.mean(), reay.mean()
    
def einstein_t(tini, tfin, npoint, HT_lim=3000,dul=False,model=1):
    """
    Computes the *Einstein temperature*
    
    Args:
        tini: minimum temperature (K) of the fitting interval
        tfin: maximum temperature
        npoint: number of points in the T range
        HT_lim: high temperature limit where Cv approaches the Dulong-Petit value
        model: if model=1 a single Einstein oscillator is considered (default),
               if model > 1, 2 Einstein oscillators are considered
        
    """
    
    flag_int=False
    if f_fix.flag:
       kp_original=f_fix.value
       flag_int=True
       reset_fix()
       
    v0, k_gpa, kp=eos_temp(298.15,prt=False, update=True)
    set_fix(kp)
    print("Kp fixed to %4.2f" % kp)
    
    vol=new_volume(298.15,0.0001)
    ent, cve=entropy_v(298.15,vol[0])
    dp_limit=apfu*3*avo*kb                  # Dulong Petit limit
    emp=10636/(ent/apfu+6.44)               # Empirical Einstein T
    
    t_range=np.linspace(tini, tfin, npoint)
    cv_list=np.array([])
    
    for ti in t_range:
        enti, cvi=entropy_v(ti, vol, plot=False, prt=False)
        cv_list=np.append(cv_list, cvi)
        
    reset_fix()
    if flag_int:
        set_fix(kp_original)
        
    t_range=np.append(t_range,HT_lim)
    cv_list=np.append(cv_list, dp_limit)
        
    sigma=np.ones(len(t_range))
    sigma[len(sigma)-1]=0.1
    
    if model==1:
       ein_fit, ein_cov=curve_fit(einstein_fun, t_range, cv_list, p0=emp, \
                      sigma=sigma, xtol=1e-15, ftol=1e-15)
    else:
       ein_fit, ein_cov=curve_fit(einstein_2_fun, t_range, cv_list, \
                      sigma=sigma,p0=[emp,emp], xtol=1e-15, ftol=1e-15) 
    
    t_range_new=np.linspace(tini,HT_lim,50)
    
    plt.figure()
    if model==1:
       plt.plot(t_range_new, einstein_fun(t_range_new, ein_fit[0]), "k-", \
                t_range, cv_list, "k*")
    else:
       plt.plot(t_range_new, einstein_2_fun(t_range_new, ein_fit[0],ein_fit[1]), "k-", \
                t_range, cv_list, "k*")  
    plt.xlabel("Temperature (K)")
    plt.ylabel("Cv (J/mol K)")
    plt.show()
    
    print("\nEinstein temperature")
    print("empirical estimation (from molar entropy): %6.2f K" % emp)
    if model==1:
       print("result from fit:                           %6.2f K" % ein_fit[0])
    else:
       print("result from fit:                           %6.2f, %6.2f K" % (ein_fit[0],ein_fit[1])) 
    print("Dulong-Petit limit for Cv (T = %5.2f K): %6.2f J/mol K" % \
          (HT_lim, dp_limit))
    
    t_table=np.linspace(tini,tfin,10)
    cv_real=np.array([])
    cv_ein=np.array([])
    for ti in t_table:
        enti, cri=entropy_v(ti, vol, plot=False, prt=False)
        if model==1:
           ce=einstein_fun(ti,ein_fit[0])
        else:
           ce=einstein_2_fun(ti,ein_fit[0],ein_fit[1])
        cv_real=np.append(cv_real, cri)
        cv_ein=np.append(cv_ein, ce)
        
    serie=(t_table,cv_real,cv_ein)
    pd.set_option('colheader_justify', 'center')
    df=pd.DataFrame(serie, index=['T (K)','Cv "real"','Cv "fit"'])
    df=df.T
    df2=df.round(2)
    print("")
    print(df2.to_string(index=False))    
    if model==1:
       print("\nFit at T = %6.2f K: Cv = %6.2f J/mol K" % \
            (HT_lim, einstein_fun(HT_lim, ein_fit[0])))
    else:
       print("\nFit at T = %6.2f K: Cv = %6.2f J/mol K" % \
            (HT_lim, einstein_2_fun(HT_lim, ein_fit[0], ein_fit[1])))  
    if dul:
       return ein_fit           

def einstein_fun(tt,eps):
    
    return apfu*3*avo*kb*((eps/tt)**2)*np.exp(eps/tt)/((np.exp(eps/tt)-1)**2)

def einstein_2_fun(tt,eps1,eps2):
    f1=apfu*3*avo*kb/2.
    f2=((eps1/tt)**2)*np.exp(eps1/tt)/((np.exp(eps1/tt)-1)**2)
    f3=((eps2/tt)**2)*np.exp(eps2/tt)/((np.exp(eps2/tt)-1)**2)
    return f1*(f2+f3)
    
           
def kief_intnd_t(x,tt,xnti):
    xi=xnti/tt
    num=((np.arcsin(x/xi))**2)*(x**2)*(np.exp(x))
    denom=((xi**2 - x**2)**0.5)*(np.exp(x)-1)**2
    return num/denom    
    
def kief_int_t(tt,xnti):
    
    kief_l=lambda x: kief_intnd_t(x,tt,xnti)
    integ,err=scipy.integrate.quad(kief_l,0.,(xnti/tt),epsrel=1e-8)
    
    return integ*(3/(apfu*zu))*avo*kb*((2/np.pi)**3)

def cv_kief(tt):
    
    cv_k=np.array([])
    for ix in kieffer.kief_freq:
        ci=kief_int_t(tt,ix)
        cv_k=np.append(cv_k,ci)
        
    return cv_k.sum()

    
def kief_intnd_ent_t(x,tt,xnti):
    xi=xnti/tt
    num1=((np.arcsin(x/xi))**2)*x
    denom1=((xi**2 - x**2)**0.5)*(np.exp(x)-1)**2
    
    num2=((np.arcsin(x/xi))**2)*np.log(1-np.exp(-x))
    denom2=(xi**2 - x**2)**0.5
    
    val=num1/denom1 - num2/denom2
    
    return val 
        
def kief_int_ent_t(tt,xnti):
    
    kief_l=lambda x: kief_intnd_ent_t(x,tt,xnti)
    integ,err=scipy.integrate.quad(kief_l,0.,(xnti/tt), epsrel=1e-8)
    
    return integ*(3/(apfu*zu))*avo*kb*((2/np.pi)**3)

def s_kief(tt):
    
    s_k=np.array([])
    for ix in kieffer.kief_freq:
        si=kief_int_ent_t(tt,ix)
        s_k=np.append(s_k,si)
        
    return s_k.sum()    
            
        
def kief_intnd_free_t(x,tt,xnti):
    xi=xnti/tt
    
    num=((np.arcsin(x/xi))**2)*np.log(1-np.exp(-x))
    denom=(xi**2 - x**2)**0.5
    
    val=num/denom
    
    return val 
        
def kief_int_free_t(tt,xnti):
    
    kief_l=lambda x: kief_intnd_free_t(x,tt,xnti)
 
    integ,err=scipy.integrate.quad(kief_l,0.,(xnti/tt), epsrel=1e-8)
      
    return integ*(3*tt/(apfu*zu))*avo*kb*((2/np.pi)**3)

def free_kief(tt):
    
    f_k=np.array([])
    for ix in kieffer.kief_freq:
        fi=kief_int_free_t(tt,ix)
        f_k=np.append(f_k,fi)
        
    return f_k.sum()   

def free_stack_t(tini, tfin, npoints):
    
    """
    Computes the contribution to the Helmholtz free energy from acoustic
    phonons, in a given temperature range, according to the Kieffer model.
    Values of free energies are stored in a stack (of the kieffer class)
    and recovered, when needed, by the method get_value of the class.
    
    The function is called by quick_start or by the freq method of the
    kieffer class. 
    """
    
    t_range=np.linspace(tini, tfin, npoints)  
    f_list=np.array([])
    for ti in t_range:
        fi=free_kief(ti)
        f_list=np.append(f_list, fi)
       
    kieffer.stack(t_range, f_list)
    kieffer.stack_flag=True
    
    
def cv_k_plot(tini, tfin, npoints):
    t_range=np.linspace(tini, tfin, npoints)
    cv_range=np.array([])
    s_range=np.array([])
    
    for ti in t_range:
        cvi=cv_kief(ti)
        si=s_kief(ti)
        cv_range=np.append(cv_range, cvi)
        s_range=np.append(s_range,si)
    
    plt.figure()    
    plt.plot(t_range, cv_range,"k-")
    
    plt.figure()
    plt.plot(t_range, s_range,"k--")    

def kief_freq(vel):
    V0=popt[0]
    Z=zu
    prefact=(6*np.pi*np.pi/(Z*V0*1e-30))**(1/3)
    fa=prefact*vel*1e3/(2*np.pi)
    return fa/csl      

def anharm_setup():
    anh_inp=path+'/'+'anh.txt'
    file=np.array([])
    fi=open(anh_inp)
    for iff in np.arange(anharm.nmode):
        line=fi.readline().rstrip()        
        file=np.append(file,line)
    fi.close()
    
    anharm.param=np.array([])    
    for iff in np.arange(anharm.nmode):
        anh_file=path+'/'+file[iff]
        with open(anh_file) as fi:
             anh_pow=np.array([])
             l0=['']
             while (l0 !='END'): 
               str=fi.readline()
               lstr=str.split()
               l0=''
               if lstr !=[]:
                 l0=lstr[0].rstrip()
               if lstr != ['END']:
                 anh_pow=np.append(anh_pow,int(lstr[0]))
                 anh_pow=np.append(anh_pow,int(lstr[1]))
                 anh_pow=np.append(anh_pow,float(lstr[2]))
        fi.close()
    
        ll=len(anh_pow)
        row=int(ll/3)
        anh_pow=anh_pow.reshape(row,3)
    
        anharm.param=np.append(anharm.param,anh_pow)
    
    size=anharm.param.size
    fact=int(anharm.nmode*3)
    row=int(size/fact)
        
    anharm.param=anharm.param.reshape(anharm.nmode,row,3)
    
def helm_anharm_func(mode,vv,tt):
    h=0.
    term=len(anharm.param[mode])
    ntr=np.arange(term)
    for it in ntr:
        p1=int(anharm.param[mode][it][0])
        p2=int(anharm.param[mode][it][1])
        h=h+anharm.param[mode][it][2]*(vv**p1)*(tt**p2)
    
    return h

def anharm_pressure_vt(mode,vv,tt,deg=2,dv=2,prt=True):
    """
    Pressure (GPa) of a single anharmonic mode at a given cell volume and 
    temperature from the derivative -(dF/dV)_T
    
    Args:
       mode: mode number (a number in the list [0,..., anharm.nmode])
       vv:   volume (A^3)
       tt:   temperature (K)
       deg:  degree of the polynomial fitting the F(V) function (default: 2)
       dv:   V range (A^3) for the calculation of the F(V) function (default: 2)
       prt:  print formatted result (default: True)
    """
    
    if not anharm.flag:
       print("Anharmonic correction is off. No calculation is done")
       return
   
    vmin=vv-dv/2.
    vmax=vv+dv/2.
    nv=8
    v_list=np.linspace(vmin,vmax,nv)
    h_list=np.array([])
    
    for iv in v_list:
        ih=helm_anharm_func(mode,iv,tt)
        h_list=np.append(h_list, ih)
        
    fit_h=np.polyfit(v_list,h_list,deg)
    fit_hd=np.polyder(fit_h,1)
    pressure=np.polyval(fit_hd,vv)
    
    pressure=-1*pressure*conv*1e21*anharm.wgt[mode]
    
    if prt:
        print("Pressure: %5.2f GPa" % pressure )
    else:
        return pressure
    
def anharm_pressure(mode,tmin,tmax,nt,deg=2,dv=2,fit=True, fit_deg=4, prt=True):
    
    """
    Pressure (GPa) of an anharmonic mode in a given T range
    
    Args: 
       mode:     mode number (a number in the list [0,..., anharm.nmode])
       tmin:     minimum temperature
       tmax:     maximum temperature
       nt:       number of points in the T interval
       deg, dv:  see doc for anharm_pressure_vt
       fit:      polynomial fit of the P(T) values
       fit_deg:  degree of the fitting polynomial
       prt:      if True, prints a list of T, V, P values
    """
    
    if not anharm.flag:
       print("Anharmonic correction is off. No calculation is done")
       return
        
    nt_plot=50
    t_list=np.linspace(tmin,tmax,nt)
    v_list=list(volume_dir(it,0) for it in t_list)
    p_list=list(anharm_pressure_vt(mode,iv,it,deg=deg,dv=dv,prt=False) for iv, it in zip(v_list, t_list))
    
    if prt:
       serie=(t_list, v_list, p_list)
       pd.set_option('colheader_justify','right')
       serie=pd.DataFrame(serie,index=["T(K)","V(A^3)","P(GPa)"])
       serie=serie.T
       serie['T(K)']=serie['T(K)'].map('{:,.1f}'.format)
       serie['V(A^3)']=serie['V(A^3)'].map('{:,.4f}'.format)
       serie['P(GPa)']=serie['P(GPa)'].map('{:,.2f}'.format)    
       print("")
       print(serie.to_string(index=False))
       
    if fit:
        fit_p=np.polyfit(t_list,p_list,fit_deg)
        t_plot=np.linspace(tmin,tmax,nt_plot)
        p_plot=np.polyval(fit_p,t_plot)
    
    title="Anharmonic mode number "+str(mode)
    plt.figure()
    plt.plot(t_list,p_list,"k*")
    if fit:
       plt.plot(t_plot,p_plot,"k-")
    plt.xlabel("T (K)")
    plt.ylabel("P (GPa)")
    plt.title(title)
    plt.show()
    
def debye_limit(tmin=0.1,tmax=50,nt=24):
    
    """
    Debye temperature estimation at the low temperature limit
    
    Args:
      tmin: lower limit of the temperature range
      tmax: higher limit of the temperature range
      nt:   number of point is the T range
    """
    
    t_list=np.linspace(tmin,tmax,nt)
    t_med=np.mean([tmin,tmax])
    
    [k0,kp]=bulk_dir(t_med,serie=True)
    set_fix(kp)
    cv_list=np.array([])
    
    for it in t_list:
        iv=new_volume(it,0)
        [ent,cv]=entropy_v(it,iv)
        cv_list=np.append(cv_list, cv)
        

    p0=[1.0e-5]   
    deb_fit, deb_cov=curve_fit(cv_fit_func, t_list, cv_list, p0, \
                               xtol=1e-15, ftol=1e-15)
        
    cv_calc=np.array([])
    for it in t_list:
        ic=cv_fit_func(it, deb_fit[0])
        cv_calc=np.append(cv_calc, ic)
        
    par3=deb_fit[0]
    debye_t=(apfu*12.*(np.pi**4)*r/(5.*par3))**(1./3.)
   
    print("Debye Temperature estimation: %5.1f K\n" % debye_t)
    print("Tmin, Tmax, number of points: %4.1f K, %4.1f K, %3i" % (tmin, tmax, nt))
    print("K0, Kp: %5.2f GPa, %4.2f" % (k0, kp))
    
    plt.figure()
    plt.plot(t_list, cv_list, "*", label="Cv")
    plt.plot(t_list, cv_calc, "-", label="Cv from fit")
    plt.legend(frameon=False)
    plt.xlabel("T (K)")
    plt.ylabel("Cv (J/mol K")
    plt.show()
      
    
def cv_fit_func(tt,par3):
    return par3*(tt**3)

def debye(tmin=5.,tmax=300.,nt=24, d_min=50., d_max=1500., nd=48):
    
    """
    Debye temperature estimation
    
    Args:
       tmin: lower limit of the temperature range
       tmax: higher limit of the temperature range
       nt:   number of point is the T range
    
       d_min, d_max, nd: define the range where the Debye
                         temperature is to be searched
    """
    
    if tmin < 5.:
        tmin=5.
        print("Warning: Minimum T set to 5. K")
        
    t_list=np.linspace(tmin,tmax,nt)
    t_med=np.mean([tmin,tmax])
    
    
    [k0,kp]=bulk_dir(t_med,serie=True)
    set_fix(kp)
    cv_list=np.array([])
    
    for it in t_list:
        iv=new_volume(it,0)
        [ent,cv]=entropy_v(it,iv)
        cv_list=np.append(cv_list, cv)
        
    t_deb=np.linspace(d_min, d_max, nd)
    
    err_array=np.array([])
    for itd in t_deb:
        fit_array=np.array([])
        for it in t_list:
            icv_fit=cv_fit_func_high(it,itd)
            fit_array=np.append(fit_array, icv_fit)
            
        fit_err=np.sum((fit_array-cv_list)**2)
        err_array=np.append(err_array,fit_err)
        min_t=np.argmin(err_array)
        
    
    t_deb_min=t_deb[min_t]-50.
    t_deb_max=t_deb[min_t]+50.
    
    t_deb=np.linspace(t_deb_min, t_deb_max, nd*2)
    
    err_array=np.array([])
    for itd in t_deb:
        fit_array=np.array([])
        for it in t_list:
            icv_fit=cv_fit_func_high(it,itd)
            fit_array=np.append(fit_array, icv_fit)
            
        fit_err=np.sum((fit_array-cv_list)**2)
        err_array=np.append(err_array,fit_err)
        min_t=np.argmin(err_array)
           
        
    min_t0=min_t-2
    min_t1=min_t+3
        
    t_deb_list=t_deb[min_t0:min_t1]
    err_list=err_array[min_t0:min_t1]
    
    fit_poly=np.polyfit(t_deb_list,err_list,2) 
    fit_der=np.polyder(fit_poly,1) 
    
    debye_t=-1*fit_der[1]/fit_der[0]
        
    cv_calc=np.array([])
    for it in t_list:
        ic=cv_fit_func_high(it, debye_t)
        cv_calc=np.append(cv_calc, ic)
 
    ierr=cv_calc-cv_list
    ierr2=ierr**2
    mean_err=np.sqrt(np.mean(ierr2))
   
    print("\nDebye Temperature estimation: %5.1f K" % debye_t)
    print("Error on fit: %6.1f J/mol K\n" % mean_err)
    print("Tmin, Tmax, number of points: %4.1f K, %4.1f K, %3i" % (tmin, tmax, nt))
    print("K0, Kp: %5.2f GPa, %4.2f" % (k0, kp))
    
    plt.figure()
    plt.plot(t_list, cv_list, "*", label="Cv")
    plt.plot(t_list, cv_calc, "-", label="Cv from fit")
    plt.legend(frameon=False)
    plt.xlabel("T (K)")
    plt.ylabel("Cv (J/mol K")
    plt.title("Fit with the Debye model")
    plt.show()
    
    plt.figure()
    plt.plot(t_list,ierr,"*")
    plt.xlabel("T (K)")
    plt.ylabel("Cv error (J/mol K)")
    plt.title("Cv_fit - Cv distribution")
    plt.show()

def cv_fit_func_high(tt,debye_t):
    t_lim=debye_t/tt
    d_integ, err=scipy.integrate.quad(d_integrand, 0.0, t_lim)
    debye_val=9*r*d_integ*(tt/debye_t)**3
    return apfu*debye_val
    
def d_integrand(y):
    y_val=(y**4)*np.exp(y)
    y_val=y_val/((np.exp(y)-1.)**2)
    return y_val
    
    
def reset_flag():
    
    flag_list=['disp.flag', 'disp.input', 'kieffer.flag', 'kieffer.input', 'anharm.flag',
               'lo.flag', 'flag_spline.flag', 'flag_poly.flag', 'f_fix.flag', 'verbose.flag',
               'bm4.flag', 'disp.eos_flag', 'disp.fit_vt_flag', 'static_range.flag']
    
    for iflag in flag_list:      
        r_flag=iflag+'=False'
        exec(r_flag)
        
    flag_view_input.on()
    volume_correction.off()
    
    print("All global flags set to False; flag list:\n")
    
    for iflag in flag_list:
        print(iflag)
        
    print("")
    
    exclude.restore()
    if supercell.flag:
        supercell.reset()



def user(tmax=1000):   
    data_exp=np.loadtxt(path+'/Angel_exp.txt')
    t_exp=data_exp[:,0]
    idx=np.argsort(t_exp)   
    b_exp=data_exp[:,1]
    
    t_exp=t_exp[idx]
    b_exp=b_exp[idx]

    t_list=np.linspace(50,tmax,40)
    
    b_par,v_par=bulk_modulus_p_serie(50,tmax,36,0,noeos=True,fit=True,type='spline',\
                               deg=3,smooth=5,out=True)
    
    b_list=(b_par(tt).item(0) for tt in t_list)
    b_list=list(b_list)
    
    b_calc=(b_par(tt).item(0) for tt in t_exp)
    b_calc=list(b_calc)
    
    v_calc=(v_par(tt).item(0) for tt in t_exp)
    v_calc=list(v_calc)

    delta=b_calc-b_exp
    delta=np.array(delta).round(1)
        
    plt.figure()
    plt.plot(t_list,b_list,label="Calculated")
    plt.plot(t_exp,b_exp,"k*",label="Experimental")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Bulk Modulus (GPa)")
    plt.xlim(50, 1000)
    plt.ylim(0,45)
    plt.legend(frameon=False)
    plt.show()

    serie=(t_exp, np.array(b_calc).round(1), b_exp, delta, np.array(v_calc).round(3))
    df_out=pd.DataFrame(serie, index=['T', 'CALC', 'EXP', 'Delta', 'Vol'])
    df_out=df_out.T
    print(df_out.to_string(index=False))   

     
def remark(string):
    print(string)
   
def main():    
    global h, kb, r, csl, avo, conv, fact, e_fact, ez_fact,ctime, version
    global al_power_list, cp_power_list
    global flag_fit_warning, flag_volume_warning, flag_volume_max, flag_warning
    global flag_view_input, flag_dir, f_fix, vol_opt, alpha_opt, info, lo, gamma_fit
    global verbose, supercell, static_range, flag_spline, flag_poly, exclude
    global bm4, kieffer, anharm, disp, volume_correction, volume_ctrl
    
    ctime=datetime.datetime.now()
    version="2.4.0 - 18/02/2021"
    print("This is BMx-QM program, version %s " % version)
    print("Run time: ", ctime)
    
    h=6.62606896e-34                # Planck's constant (SI units)
    kb=1.3806505e-23                # Boltzman's constant (SI unit) 
    r=8.314472                      # gas constant (SI unit)
    csl=29979245800                 # speed of light (m/sec)
    avo=6.02214179e23               # Avogadro number
    conv=4.3597447222071E-18        # conversion factor for energy a.u. --> j
    fact=csl*h*1e-9/(2*1e-30)
    e_fact=-1*h*csl/kb
    ez_fact=0.5*h*csl/conv
    
    al_power_list=(0, 1, -1, -2, -0.5)
    cp_power_list=(0, 1, -1, 2, -2, 3, -3, -0.5)
    
    flag_fit_warning=flag(True)
    flag_volume_warning=flag(True) 
    flag_volume_max=flag(False)     
    flag_warning=flag(True)
    flag_view_input=flag(True)
    flag_dir=flag(False)
    f_fix=fix_flag()
    vol_opt=fit_flag()
    alpha_opt=fit_flag()
    info=data_info()
    lo=lo_class()
    gamma_fit=gamma_class()
    verbose=verbose_class(False)
    supercell=super_class()
    static_range=static_class()
    flag_spline=spline_flag()
    flag_poly=poly_flag()
    exclude=exclude_class()
    bm4=bm4_class()
    kieffer=kieffer_class()
    anharm=anh_class()
    disp=disp_class()
    volume_correction=vol_corr_class()
    volume_ctrl=volume_control_class()
    
    vol_opt.on()
    alpha_opt.on()    
    
    if os.path.isfile("quick_start.txt"):
        with open("quick_start.txt") as fq:
            l0='#'
            while (l0 == '#'):
                 input_str=fq.readline().rstrip()
                 
                 if input_str != '':
                    l0=input_str[0]
                 else:
                    l0='#'
                    
            path=input_str
            path=path.rstrip()
            print("\nFile quick_start.txt found in the master folder")
            print("Input files in '%s' folder" % path)
            instr=np.array([])  
            input_str=''
            while (input_str != 'END'):
              input_str=fq.readline().rstrip()
              if input_str !='':
                 l0=input_str[0]
                 if l0 !='#':
                    instr=np.append(instr,input_str)
                
        quick_start(path)
    
        instr=instr[:-1]
        len_instr=len(instr)
    
        if len_instr > 0:
           print("Instructions from quick_start file will be executed:\n")
           [print(i_instr) for i_instr in instr]
           print("-"*50)
           print("")
           rspace=False
           
           for i_instr in instr:
              
               if not (i_instr[0:3] == "rem"): 
                  print("")
                  print(i_instr)
                  rspace=True
               else:
                  rspace=False
               exec(i_instr)
               if rspace:
                   print("")
           
if __name__=="__main__":
    main()
    
           
    
