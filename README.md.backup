# Time_lapse_ERT_inversion
This project includes the windowed time-lapse ERT inversion codes for the BATS and HOTBENT projects. 

The file directory:

-BATS: Original codes for inverting the BAT2 data

-HOTBENT:  Original codes for inverting the HOTBENT data

-Generalized: Modified codes for general problems

                - 2D problem
                - 3D subsurface
                - 3D column problem
                
                
### Note:
1. It will be easier to directly see the codes in the Generalized file directory. The codes in Generalized file are fully commented. The other two file directories are raw codes I used for projects.
2. The only problem for the Generalized codes is that I just modified them recently and haven't fully tested them. They may have some small bugs, some may be intuitive but some may not be easy to tackle and if you find these bugs which cannot be easily solved, please let me know. 
3. The time lapse ERT inversion is still using L2 time space regularization and L1 seems to have better performance. I will update the L1 version as soon as possible.

### Explanation to the inversion parameters

**para_flag** = 0: 
 
flag for parallel computing, 1 for the parallel computing, 0 for not

**Nums_cpu** = 3:

The number of used cpu, be careful about you service memory when you use a large number of cpu

#### Inversion parameters
During the inversion, I use the objective function as below

![eqn](c.jpg)

Therefore, compared with the parameters in Pygimili, you should consider Lambda_here = sqrt(Lambda_Pygimili). Same for the alpha, the magnitude you may add a sqrt to make it more intuitive.

**Lambda** = 5

the regularization parameter for the model smooth parameter

**alpha** = 2 

the regularization parameter for the time space smooth parameter

**decay_rate** = -0.01 

The decay rate for the non-uniform time space. The parameter should be negative as figure shows
![decay_rate](image.png)

**widsize** = 3

the window size for window time lapse inversion
