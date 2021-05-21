# QM-thermodynamics

## Ab initio calculation of an equation of state and thermodynamics of minerals

Thermodynamic properties, equation of state (3th or 4th-order Birch-Murnaghan EoS) and thermal expansion of crystal phases can be computed, 
along with their temperature and pressure dependence, by using the code here described, starting from data calculated at the ab initio quantum-mechanical (QM) level, 
which are essentially static energies (internal energies) and vibrational frequencies as functions of the unit cell volume.

Ab initio data are processed within the general framework of Statistical Thermodynamics, within the limit of the quasi-harmonic approximation (QHA).

The language of the program is Python. The files in this repository are:

- bm3_thermal_2.py: main module
- mineral_data.py:  Perplex like module
- anharm.py:        module for computation of intrinsic anharmonicity
- parame.py:        definition of computational parameters used in the main module
- quick_start.txt:  input file
- perplex_db.dat:   Perplex like database

The *pyrope* folder contains the input files to run the program on a selected example (the pyrope mineral)

The input description, explanations and tutorials can be found `here <https://qm-thermodynamics.readthedocs.io/en/main/>`_ 




