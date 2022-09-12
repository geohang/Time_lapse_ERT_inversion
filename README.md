# Time_lapse_ERT_inversion
This project includes the windowed time-lapse ERT inversion codes for the BATS and HOTBENT projects. 

The file directory:

-BATS: Original codes for inverting the BAT2 data

-HOTBENT:  Original codes for inverting the HOTBENT data

-Generalized: Modified codes for general problems

                - 2D problem
                - 3D subsurface
                - 3D column problem
                
                
Note:
1. It will be easier to directly see the codes in the Generalized file directory. The codes in Generalized file are fully commented. The other two file directories are raw codes I used for projects.
2. The only problem for the Generalized codes is that I just modified them recently and haven't fully tested them. They may have some small bugs, some may be intuitive but some may not be easy to tackle and if you find these bugs which cannot be easily solved, please let me know. 
3. The time lapse ERT inversion is still using L2 time space regularization and L1 seems to have better performance. I will update the L1 version as soon as possible.
