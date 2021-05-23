# File of parameters for bm3_thermal_2.py

# Temperature range (delta), number of points in the range (nump) and 
# degree (degree) of the fitting polynomium for the calculation of the F 
# free energy and its first and second derivatives with respect to T, 
# in order to calculate the entropy and the specific heat at constant volume   
# Parameters used by the entropy_v function
delta=10.
nump=16
degree=4

# Same as above parameters, used by the thermal_exp_v function used
# to calculate the thermal expansion
delta_alpha=20.
nump_alpha=10
degree_alpha=4

# Same as above, used for the calculation of the bulk modulus
# K at a given pressure, by direct numerical derivative with
# respect to V. Used by the functions bulk_modulus_p, grun_mode_vol
# and pressure_dir
delta_v=0.5
nump_v=9
degree_v=3
v_frac=0.0015

# Number of points in the volume range for plotting the function 
# F(V) in eos_temp
nvol_eos=40

# Function compare_exp: number of points for fitting Cp(T)
# and number of points for plotting 
ntemp_fit_compare=30
ntemp_plot_compare=120

# Functions alpha_serie and cp_serie: number of points for plotting 
# fitted Cp(T) and alpha(T)
ntemp_plot_cp=80

# Number of points for plotting frequency fits
nvol=40

# Number of points for the pressure_freq_list function
npres=15

# Kieffer model stack
kt_init=0.01
kt_fin=3500
kt_points=1000