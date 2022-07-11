# Anharmonic correction to vibrational frequencies

# Version 1.1 - 16/07/2020


# The file anharm_path.txt must be present in the root folder (the
# one containing the program). The content of anharm_path.txt is the name
# of the folder containing the data (usually, the folder relative to
# the phase to be investigated). Such name is assigned to the abs_path
# variable

# Input file: input_anharm.txt (under the abs_path folder)

# Structure of the input (input_anharm.txt):
# 
#  1) folder name where SCAN data from CRYSTAL are stored
#  2) output file name (it will be written in the folder
#         specified at line 1)
#  3) minimim, maximum temperatures and number of points
#         where the anharmonic Helmholtz function will be
#         computed       
#  4) order of the polynomial used to fit the Helmholtz 
#         free energy as a function of V and T. The unit
#         of the computed free energy is the hartree. 
#
# The output file contains the power of the fitting polynomial
# together with the optimized coefficents to reconstruct the
# Helmholtz free energy as a function of V and T in the specified 
# ranges. Volume ranges are from the input files found in the 
# specified folder.

# Files required to be found in the specified folder:
# 1) volumes.dat: it contains the volumes at which the SCANMODE's
#                 where done together with the harmonic frequencies
#                 computed by CRYSTAL.
#                 If not both 0., the last two columns, specifies
#                 the minimum and maximum q to select.  
#                 Volumes of the primitive cell in cubic A; 
#                 frequencies in cm^-1. 
# 2) vect.dat:    eigenvectors of the normal mode: une column for
#                 each volume, n the same order as specified in
#                 the volumes.dat file
# 3) input.txt:   it contains the names of the files where the Q
#                 energies from the SCANMODE's are stored, as 
#                 they are copied and pasted from the CRYSTAL
#                 output
# 4) files whose names are stored in the input.txt file. 

# NOTE: in order to be used with the BM3_thermal_2 program,
# fits from more than one normal modes must be of he same order
# All the output files produced here must be copied in the relevant 
# input folder specified for the BM3_thermal_2. 
# The Anharmonic correction in BM3_thermal_2 program is activated
# by the ANH keyword in the input file for that program.

# Usage: 
# At the simplest level, just use the helm_fit() function to read 
# the all the input and to make the relevant fits. 

# from IPython import get_ipython
# get_ipython().magic('clear')
# get_ipython().magic('reset -sf')

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit

from IPython import get_ipython

# get_ipython.run_line.magic('cls')
# get_ipython().magic('reset -sf')

class anh_class():
    pass

class data_class():
    def __init__(self,dim):
        self.dim=dim
        self.nlev=int(self.dim/2)
        
class data_flag():
    def __init__(self):
        self.comp=np.array([],dtype=bool)
        self.setup=False
        
def load_files():
   '''
   Loads data files and file names of the SCAN data
   '''
 
   data=np.loadtxt(path+'volumes.dat')
   
   if data.size==4:
      volumes=np.ndarray((1,), buffer=np.array(data[0]))
      h_freq=np.ndarray((1,), buffer=np.array(data[1]))
      qmn=np.ndarray((1,), buffer=np.array(data[2]))
      qmx=np.ndarray((1,),buffer=np.array(data[3]))
      nvol=1
   else:
      volumes=data[:,0]
      h_freq=data[:,1]
      qmn=data[:,2]
      qmx=data[:,3]
      nvol=volumes.size
      
   scan_name=np.loadtxt(path+"input.txt", dtype=str)
   mode_vect=np.loadtxt(path+"vect.dat", dtype=float)
   glob.data=data
   glob.volumes=volumes
   glob.h_freq=h_freq
   glob.nvol=nvol
   
   if data.size==4:
      glob.scan_name=np.array([scan_name]) 
      glob.mode_vect=mode_vect
      glob.qmn=qmn
      glob.qmx=qmx
   else:    
      glob.scan_name=scan_name
      glob.mode_vect=mode_vect
      glob.qmn=qmn
      glob.qmx=qmx

   prn_vol=str(volumes)
   print("Number of data SCAN's: %3i:" % nvol)
   print("Volumes: %s" % prn_vol)   
 
def set_up():  
         
    for i in np.arange(glob.nvol):
        qmn=glob.qmn[i]
        qmx=glob.qmx[i]
        anh[i]=anh_class()
        anh[i].name=glob.scan_name[i]
        anh[i].vol=glob.volumes[i]
        anh[i].h_freq=glob.h_freq[i]
        energy_data=np.loadtxt(path+glob.scan_name[i])
        anh[i].q=energy_data[:,0].astype(float)
        anh[i].q_orig=np.copy(anh[i].q)
        energy=energy_data[:,1].astype(float)
        min_e=np.min(energy)
        anh[i].e=energy-min_e
        if (qmn != 0.) or (qmx != 0.):
            test=((anh[i].q >= qmn) & (anh[i].q <= qmx))
            anh[i].q = anh[i].q[test]
            anh[i].e = anh[i].e[test]
        
        if glob.nvol==1:
            anh[i].vector=glob.mode_vect
        else:
            anh[i].vector=glob.mode_vect[:,i]
        
        fh_crys=anh[i].h_freq*csl
        anh[i].omega=2*np.pi*fh_crys
        anh[i].qmax=np.sqrt(sum(anh[i].vector**2))
        anh[i].q2max=(anh[i].qmax**2)*(bohr**2)  
        anh[i].red=ht/(anh[i].omega*anh[i].q2max); 
        anh[i].q=anh[i].q*anh[i].qmax 
        flag.comp=np.append(flag.comp, False)
     
    flag.setup=True

def energy_func(qq, a, b, c, d):
    return a+b*qq**2+c*qq**3+d*qq**4

def energy_quad(qq, a, b):
    return a+b*qq**2

def start_fit(iv, npt=40):
  q=anh[iv].q
  e=anh[iv].e
  
  fit_par,_ =curve_fit(energy_func,q,e)
  fit_quad,_ =curve_fit(energy_quad,q,e)
  
  anh[iv].par=fit_par

#  anh[iv].par=fit_quad
#  anh[iv].par=np.append(anh[iv].par, [0., 0.])
  
  min_q=np.min(q)
  max_q=np.max(q)
  q_list=np.linspace(min_q,max_q,npt)
  e4_list=np.array([])
  e2_list=np.array([])
  
  for iq in q_list:
      ieq4=energy_func(iq,*anh[iv].par)
      ieq2=energy_quad(iq,*fit_quad)
      e4_list=np.append(e4_list,ieq4)
      e2_list=np.append(e2_list,ieq2)
  
  
  plt.figure()
  plt.plot(q_list,e4_list,"-",label='Quartic fit')
  plt.plot(q_list,e2_list,"--",label='Quadratic fit')
  plt.plot(anh[iv].q,anh[iv].e,"*",label='Actual values')
  plt.xlabel("Q")
  plt.ylabel("E")
  plt.legend(frameon=True)
  plt.show()
  
  anh[iv].ko=2*anh[iv].par[1]*conv/(bohr**2)
  lam=anh[iv].par[3]
  d3l=anh[iv].par[2]
  anh[iv].zero_l=anh[iv].par[0]
  anh[iv].om=np.sqrt(anh[iv].ko/anh[iv].red)
  anh[iv].nu=anh[iv].om/(2*np.pi*csl)
  
  anh[iv].lam=lam*conv/(bohr**4);                 
  anh[iv].d3l=d3l*conv/(bohr**3);                  
  anh[iv].fact=(ht/(2*anh[iv].red*anh[iv].om))**2;        
  anh[iv].factd=(ht/(2*anh[iv].red*anh[iv].om))**(3/2);
  anh[iv].fact_1=anh[iv].lam*anh[iv].fact;
  anh[iv].factd_1=iun*anh[iv].factd*anh[iv].d3l;
  anh[iv].h_omeg=ht*anh[iv].om;      
  
def diag_n(iv, n):
 dn=(anh[iv].fact_1*6*(n**2+n+1/2))+(anh[iv].h_omeg*(n+1/2));
 return dn

def extra_1(iv, n):
   ext1=-3*anh[iv].factd_1*(n+1)*(np.sqrt(n+1));
   return ext1

def extra_2(iv, n):
   ext2=-2*anh[iv].fact_1*(3+2*n)*(np.sqrt((n+2)*(n+1)));
   return ext2

def extra_3(iv, n):
   ext3=anh[iv].factd_1*np.sqrt((n+3)*(n+2)*(n+1));
   return ext3

def extra_4(iv, n):
   ext4=anh[iv].fact_1*np.sqrt((n+4)*(n+3)*(n+2)*(n+1));
   return ext4

def H_matrix(iv):
    ind=np.arange(glob.dim)
    H=np.zeros((glob.dim,glob.dim),dtype=complex)
    for ii in ind:
        for jj in ind:
            if ii==jj: 
                H[jj][ii]=diag_n(iv, ii)
            elif jj==ii+2:
                H[jj][ii]=extra_2(iv, ii)
            elif jj==ii-2:
               H[jj][ii]=extra_2(iv, jj)
            elif jj==ii+4:
               H[jj][ii]=extra_4(iv, ii)
            elif jj==ii-4:
               H[jj][ii]=extra_4(iv, jj)
            elif jj==ii+1:
               H[jj][ii]=extra_1(iv, ii)
            elif jj==ii-1:
               H[jj][ii]=-1*extra_1(iv, jj)
            elif jj==ii+3:
               H[jj][ii]=extra_3(iv, ii)
            elif jj==ii-3:
               H[jj][ii]=-1*extra_3(iv, jj)   
               
    return H

def energy_anh(iv):
     H_mat=H_matrix(iv)
     vals=np.linalg.eigvals(H_mat)
     vals=np.real(vals)
     anh[iv].vals=np.sort(vals)
     anh[iv].e_zero=anh[iv].zero_l+anh[iv].vals/conv
    
def partition(iv, temp, nl=10):
    """
    Computes the partition function by direct summation of the 
    exponential terms. By default, the number of the energy levels 
    involved in the summation is in the variable glob.nlev, whose 
    value is 1/2 of the dimension of the Hamiltonian matrix.
    
    Args:
        v:    volume index (according to the list of volumes specified
              in the volumes.dat file)
        temp: temperature (K)
        nl:   number of energy levels considered in the summation
              (default: 10)  
    """
    lev_list=np.arange(nl)
    z=0.

    for i in lev_list:
        z=z+np.exp(-1*anh[iv].vals[i]/(k*temp))
        
    return z

def helm(iv, temp):
    """
    Computes the Helmholtz free energy (in hartree)
    
    Args: 
        iv:   volume index (according to the list of volumes specified
              in the volumes.dat file) 
        temp: temperature (K)
    """
    z=partition(iv, temp, nl=glob.nlev)
    return -1*k*temp*np.log(z)/conv

def check_partition(iv, temp, from_plot=False):
    """
    Checks convergence of the partition function at a given 
    temperature
    
    Args:
       iv:   volume index (according to the list of volumes specified
             in the volumes.dat file) 
       temp: temperature (k)
    """
    tol_der=0.005
    min_lev=5
    max_lev=glob.nlev
    lev_list=np.arange(min_lev,max_lev)
    
    z_list=np.array([])
    for il in lev_list:
        iz=partition(iv,temp,il)
        z_list=np.append(z_list,iz)
        
    der_z=np.gradient(z_list)
     
    tlt="Partition function: convergence test for T = " + str(temp) + " K"
    plt.figure()
    plt.plot(lev_list, z_list)
    plt.title(tlt)
    plt.xlabel('Number of vibrational levels')
    plt.ylabel('Partition function')
    plt.show()
    
    test=(der_z >= tol_der)
    st=sum(test)+min_lev
    
    print("Threshold for convergence (on the variation of Z): %4.4f" % tol_der)
    if (st < glob.nlev):
       print("Convergence reached at the %3i level" % st)
    else:
       print("Warning: convergence never reached")
       
    eth=anh[iv].e_zero[st]
    test_scan=(eth-anh[iv].e) >= 0.
    
    zero_scan=True
    scan_sum=sum(test_scan)
    if scan_sum == 0:
        zero_scan=False
    
    if zero_scan:
       min_q=0.
       max_q=0.
       q_test=anh[iv].q[test_scan]
       min_q=np.min(q_test)
       max_q=np.max(q_test)
    else:
       min_q=np.min(anh[iv].q)*anh[iv].qmax
       max_q=np.max(anh[iv].q)*anh[iv].qmax
    
    min_sc=np.min(anh[iv].q)
    max_sc=np.max(anh[iv].q)
    
    mn_qmax=min_q/anh[iv].qmax
    mx_qmax=max_q/anh[iv].qmax
    
    if from_plot:
       print("Minimum and maximum q values: %4.2f, %4.2f" % (mn_qmax, mx_qmax))
    else:
       print("Minimum and maximum q values: %4.2f, %4.2f" % (min_q, max_q)) 
    
    if min_q <= min_sc or max_q >= max_sc:
        print("Warning: Q-SCAN out of range")
        
def frequencies(iv, mxl=5, spect=False):
    delta_e=np.gradient(anh[iv].vals)
    freq=delta_e/(csl*h)
    if not spect:
       print("\nFrequencies (cm^-1) from the first %2i levels\n" % mxl)
       il=0
       while il <= mxl:
            print(" %6.2f" % freq[il])
            il=il+1  
    else:
        return freq

def computation(iv):
    
    if not flag.setup:
        set_up()
        
    start_fit(iv)
    energy_anh(iv)
    flag.comp[iv]=True
    
def start(temp=300):
    set_up()
    for ii in np.arange(glob.nvol):
        print("\n--------------\nVolume N. %3i" % ii)
        print("Volume %6.3f A^3, harmonic freq.: %6.2f cm^-1" %\
             (anh[ii].vol, anh[ii].h_freq)) 
        print("Scan data file:  %s" % glob.scan_name[ii])
        computation(ii)
        check_partition(ii,temp)
        frequencies(ii)

def helm_fit(temp=300):
    """
    Main function of the program: the produces the final result of
    the F(V,T) surface. 
    
    Args:
        temp: temperature (in K) used in the test for convergence
              of the partition function (default: 300 K)
    """
    start(temp)    
    tl=np.linspace(tmin,tmax,nt)
    vl=glob.volumes
    
    helm_val=np.array([])
    
    for it in tl:
        for iv in np.arange(glob.nvol):
            ih=helm(iv,it)
            helm_val=np.append(helm_val,ih)
    
    helm_val=helm_val.reshape(nt,glob.nvol)
    vl,tl=np.meshgrid(vl,tl)

    ptt=np.arange(pt+1)
    pvv=np.arange(pv+1)
    p_list=np.array([],dtype=int)
    
    for ip1 in ptt:
        for ip2 in pvv:
            i1=ip2
            i2=ip1-ip2
            if i2 < 0:
                break
            ic=(i1, i2)
            p_list=np.append(p_list,ic)
    
    psize=p_list.size
    pterm=int(psize/2)
    
    glob.p_list=p_list.reshape(pterm,2) 
    
    x0=np.zeros(pterm)
    
    vl=vl.flatten()
    tl=tl.flatten()
    helm_val=helm_val.flatten()
    
    fit, pcov = curve_fit(helm_func, [vl, tl], helm_val, p0 = x0)
    
    t_plot=np.linspace(tmin,tmax,40)
    v_plot=np.linspace(np.min(vl),np.max(vl),40)
    
    v_plot,t_plot=np.meshgrid(v_plot,t_plot)
    v_plot=v_plot.flatten()
    t_plot=t_plot.flatten()
    h_plot=helm_func([v_plot, t_plot], *fit)
    
    h_plot=h_plot.reshape(40,40)
    v_plot=v_plot.reshape(40,40)
    t_plot=t_plot.reshape(40,40)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection='3d', )
    
    ax.plot_surface(t_plot, v_plot, h_plot)
    ax.scatter(tl,vl,helm_val,c='r')
    ax.set_ylabel("Volume", labelpad=7)
    ax.set_xlabel("Temperature", labelpad=7)
    ax.set_zlabel('F(V,T)', labelpad=8)
    plt.show()
    
    glob.fit=fit
    
    file=open(outfile,"w")
    
    for iw, iff in zip(glob.p_list,fit):    
        issw0=iw[0]
        issw1=iw[1]
        siw=str(issw0)+" "+str(issw1)+"   "+str(iff)+"\n"
        file.write(siw)
   
    file.write('END')
    file.close()
    
    print("\nFile %s written" % outfile)
    print("V, T polynomial fit of degree %3i %3i" % (pv, pt))
    print("Temperature range: tmin=%4.1f, tmax=%4.1f" % (tmin,tmax))
    
    vmin=np.min(glob.volumes)
    vmax=np.max(glob.volumes)
    
    print("Volume range: Vmin=%5.3f, Vmax=%5.3f" % (vmin, vmax))
    
    hc=helm_func([vl, tl],*glob.fit)
    
    df2=(helm_val-hc)**2
    mean_error=np.sqrt(sum(df2))/df2.size
    max_error=np.max(np.sqrt(df2))
    
    print("Mean error from fit: %6.2e" % mean_error)
    print("Maximum error: %6.2e" % max_error)
    
    
def helm_func(data,*par):
    vv=data[0]
    tt=data[1]

    nterm=glob.p_list.shape[0]
    func=0.   
    for it in np.arange(nterm):
        pvv=glob.p_list[it][0]
        ptt=glob.p_list[it][1]
        func=func+par[it]*(vv**pvv)*(tt**ptt)
            
    return func

def plot_levels(iv, max_lev, qmin=0., qmax=0., tmin=300, tmax=1000, nt=5, \
                degree=4,chk=False, temp=300):
    
    """
    Computes and plots vibrational energy levels on top of the
    potential curve of the mode.
    
    Args:
        iv:          Volume index (select the volume according 
                                   to the input list)
        max_lev:     Number of levels to plot
        qmin, qmax:  Q-range (default: qmin=qmax=0.  --> full range)
        tmin, tmax:  T-range for the computation of probability of
                     of occupation of the vibrational levels
                     (default: 300, 1000K)
        nt:          number of points in the T-range
        degree:      degree of the polynomial fitting the potential
                     function (default: 4)
        chk:         check on the corvengence of the partition function
                     (default: False)
        temp:        temperature for the check of the partition function
                     (default: 300K)   
    """
    
    npoint=200
    
    if not flag.setup:
        set_up()
    if not flag.comp[iv]:
        computation(iv)        
    if chk:
        check_partition(iv, temp, from_plot=True)
        
    levels=anh[iv].vals/conv
    pot=anh[iv].e
    q=anh[iv].q/anh[iv].qmax
    
    t_list=np.linspace(tmin, tmax, nt)
    prob=np.array([])
    
    for it in t_list:
        z=partition(iv,it)
        for idx in np.arange(max_lev):
           energy=levels[idx]*conv
           iprob=(np.exp(-1*energy/(k*it)))/z
           iprob=(iprob*100).round(1)
           prob=np.append(prob, iprob)
           
    prob=prob.reshape(nt,max_lev)
    df=pd.DataFrame(prob,index=t_list.round(1))
    df=df.T
    print("Energy levels occupation (probabilities) at several")
    print("temperatures in the %4.1f - % 4.1f interval\n" % (tmin, tmax))
    print(df.to_string(index=False))
     
    if (qmin == 0.) & (qmax == 0.):
       qmin=np.min(q)
       qmax=np.max(q)
        
    test=((q>=qmin) & (q<=qmax))
    
    pot=pot[test]
    q=q[test]
    
    fit=np.polyfit(q,pot,degree)
    q_fit=np.linspace(qmin,qmax,npoint)
    e_fit=np.polyval(fit,q_fit)    
    
    q_l=np.array([])
    
    for idx in np.arange(max_lev):
        ie=levels[idx]
        test=(e_fit < ie)
        iqmin=np.min(q_fit[test])
        iqmax=np.max(q_fit[test])
        q_l=np.append(q_l,[iqmin,iqmax])
        
    q_l=q_l.reshape(max_lev,2)
        
    plt.figure()
    plt.plot(q,pot)
    
    for idx in np.arange(max_lev):
        p1=q_l[idx][0]
        p2=q_l[idx][1]
        
        qp=(p1,p2)
        ep=(levels[idx],levels[idx])
        plt.plot(qp,ep,"k--",linewidth=1)
     
    volume=anh[iv].vol.round(3)    
    tlt="Volume: "+str(volume)+" A^3; Num. of levels: "+str(max_lev) 
    plt.xlabel("Q (in unit of Q_max)")
    plt.ylabel("E (hartree)")
    plt.title(tlt)
    plt.show()
    
def spectrum(iv,temp,nline=5,tail=8., head=8., sigma=2., fwhm=2., eta=0., npp=240):
    """
    Computes the spectrum of the anharmonic mode by using a specified peak shape
    
    Args:
        iv:      Volume index
        temp:    Temperature (K)
        nline:   Number of lines to be considered
        tail, head: the plotted range is [min(freq)-tail. max(freq)+head]
                    where min(freq) and max(freq) are respectively the minum and
                    maximum frequencis resulting from the "nline" transitions
        sigma: sigma associated to the Gaussian profile
        fwhm:  full widthat half maximum associated to the Lorentzian profile
        eta: Gaussian/Lorentzian ratio; 
             eta=0: full Gaussian (G) profile
             eta=1: full Lorentzian (L) profile
             in general: profile=G*(1-eta)+L*eta
        npp: number of points used for the plot
        
    Note:
        The vertical lines drawn under the spectrum mark the positions
        of the transition frequencies. If the number of lines is greater
        than 3, a color code is associated to such lines;  
        blue - transitions involving levels associated to low quantum numbers;
        green -transitions at intermediate quantum numbers;
        red - transition at high quantum numbers
    """

 
    if not flag.setup:
       set_up()
    if not flag.comp[iv]:
       computation(iv)  
       
    freq=frequencies(iv,nline,spect=True)
    freq=freq[0:nline]
        
    z=partition(iv,temp)
    levels=anh[iv].vals/conv
    prob=np.array([])
    
    for idx in np.arange(nline):
         energy=levels[idx]*conv
         iprob=(np.exp(-1*energy/(k*temp)))/z
         prob=np.append(prob, iprob)
         
    f_min=np.min(freq)-tail
    f_max=np.max(freq)+head              
    s_list=np.linspace(f_min, f_max, npp)
    
    ps_list=np.array([])
    
    for ff in s_list:
        ff_int=0.
        idx=0
        for if0 in freq:
            ig=gauss(if0,ff,sigma)
            il=lorentz(if0,ff,fwhm)
            ff_int=ff_int+prob[idx]*(ig*(1.-eta)+il*eta) 
            idx=idx+1
        ps_list=np.append(ps_list,ff_int)
            
    int_max=np.max(ps_list)
    y_mx_lim=int_max+int_max*0.1
    
    
    if nline > 2:
       n3=nline/3.
       c3=int(round(n3,0))
       t1=c3
       t2=2*c3
    
       color_l=np.array([])
       idx=0
       for idx in np.arange(nline):
           if idx < t1:
              icol="b"
           elif (idx >= t1) and idx < t2:
              icol="g"
           else: 
              icol="r"
            
           color_l=np.append(color_l,icol)
           
    else:
       color_l=["r"]*nline
         
    
    lin=["-"]*nline
    v_style=list(color_l[idx] + lin[idx] for idx in np.arange(nline))
    
    
    if nline > 2:
        idx=0
        v_line=np.array([])
        for if0 in np.arange(nline):
            if color_l[idx]=="b":
                iv_line=int_max/20
            elif color_l[idx]=="g":
                iv_line=int_max/30
            else:
                iv_line=int_max/50
    
            idx=idx+1            
            v_line=np.append(v_line,iv_line)
    else:
        v_line=[int_max/50]*nline
    
            
    y_line=list([0., iv_line] for iv_line in v_line)
    
    title="Spectrum at T = "+str(temp)+" K"+"; volume = " + str(anh[iv].vol.round(3)) + " A^3"
    plt.figure()
    plt.plot(s_list, ps_list, "k-")
    idx=0
    for if0 in freq:
        plt.plot([if0,if0],y_line[idx],v_style[idx])
        idx=idx+1
    
    plt.ylim(0., y_mx_lim)    
    plt.xlabel("Wave Number cm^-1")
    plt.ylabel("Relative Intensity")
    plt.title(title)
    plt.show()
    
    prob=prob*100.
    
    print("\nFrequencies and relative weights\n")
          
    idx=0      
    for if0 in freq:
        print("%5.2f  %5.1f" % (freq[idx], prob[idx]))
        idx=idx+1
             
def gauss(f0,ff,sigma):
    sig=sigma/2.355
    return np.exp((-1*(ff-f0)**2)/(2*sig**2))

def lorentz(f0, ff, fwhm):
    f2=fwhm/2.
    numer=(1./np.pi)*f2
    denom=(ff-f0)**2+(f2**2)
    return numer/denom

def single(temp=300, max_lev=5, qmin=0., qmax=0., tmin=300, tmax=1000, nt=4, nline=4,\
           tail=4., head=4., sigma=2., fwhm=2., eta=1., npp=240):
    """
    Computation in case of a single volume. Optional input parameters are
    those described for the plot_levels and spectrum functions.
    """
    set_up()
    start_fit(0)
    energy_anh(0)
    plot_levels(0, max_lev, qmin, qmax, tmin, tmax, nt, \
                    degree=4,chk=False, temp=300)
    spectrum(0,temp,nline, tail, head, sigma, fwhm, eta, npp)
    
    
def main():
    global ctime, h, k, r, csl, avo, ht, bohr, uma, iun, conv, anh
    global glob, flag, abs_path, path, outfile, temp, power_limit
    global tmin, tmax, nt, Version, pv, pt
    
    Version="1.1 - 16/07/2020"
    ctime=datetime.datetime.now()
    print("Run time: ", ctime)
    
    h=6.62606896e-34
    k=1.3806505e-23
    r=8.314472
    csl=29979245800
    avo=6.02214179e23
    conv=4.35981E-18
    ht=h/(2*np.pi)
    bohr=5.291772108e-11
    uma=1.6605386e-27
    iun=complex(0,1)
        
    glob=data_class(200)
    flag=data_flag()

    fi=open('anharm_path.txt')
    abs_path=fi.readline().rstrip()
    fi.close()

    fi=open(abs_path+'/'+'input_anharm.txt')
    path=fi.readline()
    path=path.rstrip()
    path=abs_path+'/'+path+'/'

    outfile=fi.readline()
    outfile=abs_path + '/' + outfile.rstrip()

    temp=fi.readline().rstrip()
    temp=temp.split()

    pwl=fi.readline()
    power_limit=pwl.rstrip().split()
    if len(power_limit) == 1:
        power_limit=int(power_limit[0])
        pv=power_limit
        pt=pv
    else:
        pv=int(power_limit[0])
        pt=int(power_limit[1])
    

    tmin=float(temp[0])
    tmax=float(temp[1])
    nt=int(temp[2])

    fi.close()

    print("This is Anharm, version %s \n" % Version)
    print("Basic usage: helm_fit()\n")
    
    load_files()
    anh=np.ndarray((glob.nvol,), dtype=object)
        
if __name__=="__main__":
    main()

    
    
           
        
           
    
    
    
    
    
    
    

