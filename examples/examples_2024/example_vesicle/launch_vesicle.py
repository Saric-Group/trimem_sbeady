# ---------------------------------------------------------------------#
# TEST TO VERIFY TRILMP INSTALLATION                                   #
# Author: Maitane Muñoz-Basagoiti (maitane.munoz-basagoiti@ista.ac.at) #
#                                                                      #
# This code launches TriLMP for a fluid membrane.                      #
# In this simulation, the membrane is a vesicle.                       #
#                                                                      #
# OUTPUTS:                                                             #  
# - TriLMP prints:                                                     #
#   + Total Trimem energy (see inp_system.dat)                         #
#   + Volume enclosed by the mesh (see inp_system.dat)                 #
#   + Area of the mesh (see inp_system.dat)                            #
#   + Bending energy of the mesh (see inp_system.dat)                  #
#   + Performance information (see inp_performance.dat)                #
# - In this example, we use LAMMPS computes to obtain                  #
#   + Membrane shape (gyration tensor)                                 #
#   + Position of the vesicle COM                                      #
# ---------------------------------------------------------------------#

from pathlib import Path

import trimesh
import numpy as np
import pandas as pd
from trimem.mc.trilmp import TriLmp

# generate directory to save sim. checkpoints
Path("checkpoints").mkdir(exist_ok=True)

# mesh initialization
mesh = trimesh.creation.icosphere(3)
N = len(mesh.vertices)

# rescaling mesh distances
# [IMPORTANT: Hard-coded in TriLMP]
# minimum edge length = 1.0
# therefore, rescaling is necessary
desired_average_distance = 2**(1.0/6.0) * 1.0
current_average_distance = np.mean(mesh.edges_unique_length)
scaling = desired_average_distance/current_average_distance
mesh.vertices *=scaling

print("")
print("** STARTING TRILMP SIMULATION **")
print("")
print("Vertex properties: ")
print(f"MESH VERTICES : ", len(mesh.vertices))
print(f"MESH FACES    : ", len(mesh.faces))
print(f"MESH EDGES    : ", len(mesh.edges))
print("")

# MD simulation time step
dt = 0.001

# initialization of the trilmp object
trilmp=TriLmp(initialize=True,                            # use mesh to initialize mesh reference
            debug_mode=False,                             # DEBUGGING: print everything
            num_particle_types=1,                         # PART. SPECIES: total particle species in system 
            mass_particle_type=[1.0],                     # PART. SPECIES: mass of species in system
            group_particle_type=['vertices'],             # PART. SPECIES: group names for species in system

            mesh_points=mesh.vertices,                  # input mesh vertices 
            mesh_faces=mesh.faces,                      # input of the mesh faces

            kappa_b=20.0,                            # MEMBRANE MECHANICS: bending modulus (kB T)
            kappa_a=1000000.0,                       # MEMBRANE MECHANICS: constraint on area change from target value (kB T)
            kappa_v=1000000.0,                       # MEMBRANE MECHANICS: constraint on volume change from target value (kB T)
            kappa_c=0.0,                             # MEMBRANE MECHANICS: constraint on area difference change (kB T)
            kappa_t=10000.0,                         # MEMBRANE MECHANICS: tethering potential to constrain edge length (kB T)
            kappa_r=1000.0,                          # MEMBRANE MECHANICS: repulsive potential to prevent surface intersection (kB T)
            
            step_size=dt,                            # MD PART SIMULATION: timestep of the simulation
            traj_steps=100,                          # MD PART SIMULATION: number of MD steps before bond flipping
            flip_ratio=0.1,                          # MC PART SIMULATION: fraction of edges to flip
            initial_temperature=1.0,                 # MD PART SIMULATION: temperature of the system
            pure_MD=True,                            # MD PART SIMULATION: accept every MD trajectory
            switch_mode="random",                    # MD/MC PART SIMULATION: 'random' or 'alternating' flip-or-move
            box=(-20, 20, -20, 20, -20, 20),         # MD PART SIMULATION: simulation box properties, periodic

            equilibration_rounds=100,   # MD PART SIMULATION: HOW LONG DO WE LET THE MEMBRANE EQUILIBRATE
            
            info=100,                              # OUTPUT: frequency output in shell
            thin=100,                              # OUTPUT: frequency trajectory output
            performance_increment=10,              # OUTPUT: output performace stats to prefix_performance.dat file - PRINTED MD+MC FREQUENCY
            energy_increment=100,                  # OUTPUT: output energies to energies.dat file - PRINTED MD FREQUENCY
            checkpoint_every=100*10000,            # OUTPUT: interval of checkpoints (alternating pickles) - PRINTED MD+MC FREQUENCY 
            )

# -------------------------#
# -------------------------#
#  LAMMPS COMMANDS         #
# -------------------------#
# -------------------------#

# dump particle trajectories (vertex coordinates)
trilmp.lmp.command(f"dump XYZ all custom 1000 trajectory.dump id x y z type")

# compute position CM vesicle
trilmp.lmp.command("compute MembraneCOM vertices com")

# compute shape of the vesicle
trilmp.lmp.command("compute RadiusGMem vertices gyration")
trilmp.lmp.command("compute MemShape vertices gyration/shape RadiusGMem")

# compute temperature of the vesicle
trilmp.lmp.command("compute TempComputeMem vertices temp")

# print out all computes
trilmp.lmp.command(f"fix  aveMEM all ave/time 10000 1 10000 c_TempComputeMem c_MembraneCOM[1] c_MembraneCOM[2] c_MembraneCOM[3] c_MemShape[1] c_MemShape[2] c_MemShape[3] c_MemShape[4] c_MemShape[5] c_MemShape[6] file 'membrane_properties.dat'")

# include the integrators (pre-equilibration)
trilmp.lmp.command("fix NVEMEM vertices nve")
trilmp.lmp.command(f"fix LGVMEM vertices langevin 1.0 1.0 1.0 123 zero yes")

# cleanup pair style in case
trilmp.lmp.command("pair_style none")

# pair interactions - include harmonic repulsion in case (probably not necessary)
trilmp.lmp.command(f"pair_style hybrid/overlay table linear 2000 harmonic/cut")

# compulsory lines 
trilmp.lmp.command("pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no")
trilmp.lmp.command("pair_coeff 1 1 table trimem_srp.table trimem_srp")

# set all interactions to zero just in case for added potentials (careful with the mass and damping values)
trilmp.lmp.command("pair_coeff * * harmonic/cut 0 0")

# -------------------------#
#         RUN              #
# -------------------------#

# RUN THE SIMULATION
MDsteps = 100000
SimTime = MDsteps*dt
trilmp.run(MDsteps)

print("")
print("*** End of the simulation ***")
print("")
print(f"You have simulated a fluctuating vesicle for {SimTime} simulation time units.")
print("Now you can: ")
print(" - Load sim_setup.in in Ovito to check the initial configuration")
print(" - Load trajectory.dump in Ovito to check the simulated trajectory")
print(" - Check the performance of the code in inp_performance.dat")
print(" - Check the properties of the vesicle in inp_system.dat and membrane_properties.dat")
print(" - Reload pickle checkpoints from checkpoints/ckpt_*")
print("")
print("Have fun!")
print("")

        
