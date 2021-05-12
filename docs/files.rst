Input file
===========

.. |nbsp| unicode:: 0xA0 

Structure of the input file (*input.txt*)


The input file can contain blank lines or commented # characters that are ignored. Its structure is keyword oriented. 
Keywords can be inserted in any order and must be followed by the parameters they specify. The input is key sensitive.

List of keywords:

- ALPHA   (**optional**):   powers of a polynomial for the fitting of thermal expansion as a function of *T* 
- ANH     (**optional**):   input for anharmonic calculation
- CP      (**optional**):   powers of a polynomial for the fitting of specific heat (at *P* constant) as a function of *T*
- DISP    (**optional**):   input for phonon dispersion correction
- END     (**mandatory**):  end of keywords list
- EXCLUDE (**optional**):   specifies a list of modes to be excluded from the computation of the free energy
- EXP     (**optional**):   file name of experimental Cp and S data from the literature, for comparison with computed results
- FITVOL  (**optional**):   fitting of the frequencies in a given V range and use of the fitted values in the calculation of *F* free energy
- FREQ    (**mandatory**):  file name of the frequencies data set
- FU      (**mandatory**):  number of formula units in the primitive cell, and number of atoms in the formula unit 
- INI     (**optional**):   initial parameters for the fit of a volume integrated 3° order Birch-Murnaghan EoS to the *F* free energy(volume) data
- KIEF    (**optional**):   input for the modified Kieffer model
- LO      (**optional**):   file name of the LO-TO splitting data
- SET     (**mandatory**):  set of volumes/frequencies to be used in the calculation (a list of integers indexing the file specified by the VOLUME keyword)
- STATIC  (**mandatory**):  file name of static energy curve (volumes and static internal energy of the crystal)
- TEMP    (**optional**):   set of temperature for the output of a *PVT* file to be used with the EoSFit program
- VOLUME  (**mandatory**):  file name of the volumes corresponding to the frequencies in the FREQ file 

Below, an example of input file is shown

|  ----------------- input.txt file ---------------------
|
| # STATIC:  file of the E(V) static energy curve
| # VOLUME:  file with the set of volumes for which the frequencies were calculated
| # END:     closes the input list
| # EXP: 	  file with experimental data
| # FITVOL:  type of fit (SPLINE POLY), range of volume, number of V points and degree of the
| # |nbsp| |nbsp| |nbsp| |nbsp| |nbsp| |nbsp| |nbsp| |nbsp| spline or polynomial fit of the frequencies, and *smoothness** parameter for the spline 
| # FREQ:    file with the sets of frequencies (one set for each volume)
| # LO:      file with the modes affected by LO-TO splitting 
| # FU:      number of formula units in the (primitive) cell
| # SET:     set of volumes to be used in the calculation (list of integers)
| # TEMP:    temperature list (for EoSFit)
| # INI:     initial parameters for the EoS
| # CP       powers for the polynomial fit of Cp(T)
| # ALPHA:   Powers for the polynomial fit of alpha(T) (fixed pressure)
| # DISP:    Phonon dispersion correction from a supercell calculation
| # ANH:     Anharmonic correction
|   
| STATIC
| pyrope_static.dat
| VOLUME
| volume.dat
| FREQ
| pyrope_freq.dat
| LO
| LO.txt
| EXP
| experimental.txt
| FITVOL
| SPLINE
| 725. 768. 16 3 10
| EXCLUDE
| 88 36
| FU
| 4 20
| SET
| 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
| # TEMP
| # 40 50 60 70 80 90 100 120 150 170 180 200 230 260 300 350 400 500
| INI
| 755 170. 5 -1.1418e+04
| CP
| 0. 1. -2. 2. -0.5 -1. -3. 3.
| ALPHA
| 0. 1. -1. -2. -0.5
| END


------------------------ End of file -------------------