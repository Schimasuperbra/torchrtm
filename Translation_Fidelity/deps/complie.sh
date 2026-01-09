
rm -rf *.mod *.o ./executable.exe  
gfortran -o executable.exe MODULE_PRO4SAIL.f90 dataSpec_P5B.f90 main_PROSAIL.f90 LIDF.f90 dladgen.f PRO4SAIL.f90 prospect_5B.f90 tav_abs.f90 volscatt.f90

rm -rf *.mod *.o 

