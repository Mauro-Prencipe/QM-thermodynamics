Files Required
==============

.. |nbsp| unicode:: 0xA0 

Several files must be provided to the program. They must all be contained in a data subfolder whose name should 
be specified as an input datum when the program is started (see the *read_file* function).

Mandatory files:
----------------

1. A **volume* v.s. *static energy** file: this file contains only two columns; the first one lists *unit cell volumes* (in A^3); 
   the second one lists *static energies* (in a.u.). The name of this file must be written in the :doc:`files` under the *STATIC* keyword

2. A **volume** file listing the unit cell volumes (in A^3) at which vibrational frequencies of the normal modes were computed. 
   The name of this file must be written in the :doc:`files`, under the *VOLUME* keyword

3. A **frequencies** file: it consists of several columns and, precisely, one for each volume listed in the *volume* file; each column lists 
   the set of frequencies computed at each volume. Frequencies must be given in cm^-1. The name of this file must be written in the :doc:`files`, 
   under the *FREQ* keyword. **Note** that the first column of the frequencies file must contains the *degeneracy* of each vibrational mode (an integer number
   in the interval 1-3)

4. An **input.txt** file containing instructions and parameters for the execution of the program. 

Optional files:
---------------
5. A file containing data for the LO-TO splitting: it is required to activate a correction for the LO-TO splitting. The name of the such file must be
   provided under the keyword *LO* in the :doc:`files`. Such LO file must contains two columns: the first one contains the progressive number of the modes
   affected by LO-TO splitting (the numbering follows the same order of the modes found in the frequencies file); the second column contains the corresponding
   split values (in cm^-1)   

6. A file containing experimental data of specific heat and entropy, to be used for comparisons with computed results (by the function *compare_exp*). 
   The name of this file must be written in the :doc:`files`, under the *EXP* keyword
   
7. Files for the phonon dispersion correction (see the tutorial :doc:`_static/Dispersion.html`)

8. A **quick_start.txt** file in the master folder. This file contains the name of the folder of the data to be processed. If this file exists, 
   the program immediately proceeds to read the relevant data files, without having to issue the *read_file* and the *static* commands

Note:
   A special *parame.py* **mandatory** file must be always present in the master folder. It contains several parameters for the execution of some 
   functions of the program, which could possibly be modified by *expert* users and developers only. 
