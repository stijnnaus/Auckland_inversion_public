# Short description code base
This is a code base developed to perform atmospheric transport model inversions over Auckland, New Zealand for CO2 emission estimation. The inverse system combines transport model footprints with flux priors and atmospheric observations of CO2 at four sites in Auckland to create optimized flux estimates for biospheric and anthropogenic CO2 fluxes. This code base has been used to prepare a manuscipt on synthetic data inversions for Auckland.

Important parts of the code are:

- env/base_paths.py: Theoretically this is the only part that needs to be adjusted to run the code on different systems, although of course the NAME-III footprints, priors, etc would still need to be obtained somehow.
- preprocessing/ prepares inputs for the inversion that are used to construct all prior matrices. This often uses code from read_data/ to read in e.g., emission priors and model footprints.
- inversion/ contains the core of the inversion, with the matrices created in construct_ scripts. inversion_main.py performs the matrix calculations to retrieve optimized state and posterior covariances.
- config/ contains a script to generate yaml input files that the describe the set-up of the inversion, e.g., number of grid cells, priors to optimize, etc.
- run_scripts/ contain the scripts used to run inversions, including scripts to submit jobs on supercomputers since calculations are expensive.
- analysis/ is a loose collection of scripts used to analyze the inversion output. These were written on-the-go and are the least well structured part of the code base.

This version reads a test set-up in the run script (run_scripts/run_inversion.py), with e.g., only 1 grid cell to test if you can get the inversion running.
It should work when the environment is set up correctly, and this also describes very roughly how the code should be used. 

However, if you plan in any way to use this code it is highly adviced to contact Stijn Naus (stijn.naus@wur.nl) or Daemon Kennett (daemon.kennett@niwa.co.nz). 
An attempt has been made to structure and comment the code, but it is a work in progress, and the system is developed specifically for Auckland, New Zealand and was only used on two different computer systems, i.e., it might not be very robust wrp changing systems.
