# ---------------------------------------------------------------------#
# TEST TO VERIFY TRILMP INSTALLATION                                   #
# Author: Maitane Muñoz-Basagoiti (maitane.munoz-basagoiti@ista.ac.at) #
#                                                                      #
# This code launches TriLMP for a fluid membrane patch.                #
#                                                                      #
# OUTPUTS:                                                             #  
# - TriLMP prints:                                                     #
#   + Total Trimem energy (see inp_system.dat)                         #
#   + Area of the mesh (see inp_system.dat)                            #
#   + Bending energy of the mesh (see inp_system.dat)                  #
#   + Performance information (see inp_performance.dat)                #
# ---------------------------------------------------------------------#

from pathlib import Path

import trimesh
import numpy as np
import pandas as pd
from trimem.mc.trilmp import TriLmp

# generate directory to save sim. checkpoints
Path("checkpoints").mkdir(exist_ok=True)

# mesh initialization as a triangulated lattice
mesh_coordinates = pd.read_csv('mesh_coordinates.dat', header = None, index_col = False, sep = ' ')
mesh_coordinates_array = mesh_coordinates[[0, 1, 2]].to_numpy()
mesh_faces = pd.read_csv('mesh_faces.dat', header = None, index_col = False, sep = ' ')
mesh_faces_array = mesh_faces[[0, 1, 2]].to_numpy()
mesh = trimesh.Trimesh(vertices=mesh_coordinates_array, faces = mesh_faces_array) 
N = len(mesh.vertices)

# rescaling mesh distances
# [IMPORTANT: Hard-coded in TriLMP]
# minimum edge length = 1.0
# therefore, rescaling is necessary
desired_average_distance = 1.05
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
            periodic=True,
            num_particle_types=2,                         # PART. SPECIES: total particle species in system 
            mass_particle_type=[1.0, 1.0],                     # PART. SPECIES: mass of species in system
            group_particle_type=['vertices', 'bead'],             # PART. SPECIES: group names for species in system

            mesh_points=mesh.vertices,                  # input mesh vertices 
            mesh_faces=mesh.faces,                      # input of the mesh faces
            vertices_at_edge=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 29, 30, 39, 40, 49, 50, 59, 60, 69, 70, 79, 80, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],

            kappa_b=20,                            # MEMBRANE MECHANICS: bending modulus (kB T)
            kappa_a=1000000.0,                            # MEMBRANE MECHANICS: constraint on area change from target value (kB T)
            kappa_v=0,                            # MEMBRANE MECHANICS: constraint on volume change from target value (kB T)
            kappa_c=0.0,                            # MEMBRANE MECHANICS: constraint on area difference change (kB T)
            kappa_t=10000.0,                            # MEMBRANE MECHANICS: tethering potential to constrain edge length (kB T)
            kappa_r=1000.0,                            # MEMBRANE MECHANICS: repulsive potential to prevent surface intersection (kB T)
            area_frac=1.0,                # CHOOSE WHAT VERSION OF THE HAMILTONIAN
            step_size=dt,                        # FLUIDITY ---- MD PART SIMULATION: timestep of the simulation
            traj_steps=100,                      # FLUIDITY ---- MD PART SIMULATION: number of MD steps before bond flipping
            flip_ratio=0.2,                      # MC PART SIMULATION: fraction of edges to flip
            initial_temperature=1.0,            # MD PART SIMULATION: temperature of the system
            pure_MD=True,                                   # MD PART SIMULATION: accept every MD trajectory
            switch_mode="random",                    # MD/MC PART SIMULATION: 'random' or 'alternating' flip-or-move
            box=(-14.727272727272727, 14.772727272727273, -14.330127018922191, 14.330127018922195, -20, 20),         # MD PART SIMULATION: simulation box properties, periodic

            equilibration_rounds=100,   # MD PART SIMULATION: HOW LONG DO WE LET THE MEMBRANE EQUILIBRATE
            
            info=5000,                              # OUTPUT: frequency output in shell
            thin=5000,                              # OUTPUT: frequency trajectory output
            performance_increment=1000,    # OUTPUT: output performace stats to prefix_performance.dat file - PRINTED MD+MC FREQUENCY
            energy_increment=1000,         # OUTPUT: output energies to energies.dat file - PRINTED MD FREQUENCY
            checkpoint_every=100*5000,              # OUTPUT: interval of checkpoints (alternating pickles) - PRINTED MD+MC FREQUENCY
            )

# -------------------------#
#  LAMMPS MODIFICATIONS    #
# -------------------------#

# .................................................
#                GROUPS
# .................................................

trilmp.lmp.command('group vertex_edge id 1')
trilmp.lmp.command('group vertex_edge id 2')
trilmp.lmp.command('group vertex_edge id 3')
trilmp.lmp.command('group vertex_edge id 4')
trilmp.lmp.command('group vertex_edge id 5')
trilmp.lmp.command('group vertex_edge id 6')
trilmp.lmp.command('group vertex_edge id 7')
trilmp.lmp.command('group vertex_edge id 8')
trilmp.lmp.command('group vertex_edge id 9')
trilmp.lmp.command('group vertex_edge id 10')
trilmp.lmp.command('group vertex_edge id 11')
trilmp.lmp.command('group vertex_edge id 20')
trilmp.lmp.command('group vertex_edge id 21')
trilmp.lmp.command('group vertex_edge id 30')
trilmp.lmp.command('group vertex_edge id 31')
trilmp.lmp.command('group vertex_edge id 40')
trilmp.lmp.command('group vertex_edge id 41')
trilmp.lmp.command('group vertex_edge id 50')
trilmp.lmp.command('group vertex_edge id 51')
trilmp.lmp.command('group vertex_edge id 60')
trilmp.lmp.command('group vertex_edge id 61')
trilmp.lmp.command('group vertex_edge id 70')
trilmp.lmp.command('group vertex_edge id 71')
trilmp.lmp.command('group vertex_edge id 80')
trilmp.lmp.command('group vertex_edge id 81')
trilmp.lmp.command('group vertex_edge id 90')
trilmp.lmp.command('group vertex_edge id 91')
trilmp.lmp.command('group vertex_edge id 100')
trilmp.lmp.command('group vertex_edge id 101')
trilmp.lmp.command('group vertex_edge id 102')
trilmp.lmp.command('group vertex_edge id 103')
trilmp.lmp.command('group vertex_edge id 104')
trilmp.lmp.command('group vertex_edge id 105')
trilmp.lmp.command('group vertex_edge id 106')
trilmp.lmp.command('group vertex_edge id 107')
trilmp.lmp.command('group vertex_edge id 108')
trilmp.lmp.command('group vertex_edge id 109')
trilmp.lmp.command('group vertex_edge id 110')

trilmp.lmp.command("group BULK subtract vertices vertex_edge")

# .................................................
#            PAIR STYLES 
# .................................................

# cleanup pair style in case
trilmp.lmp.command("pair_style none")

# pair interactions
trilmp.lmp.command(f"pair_style hybrid/overlay table linear 2000 harmonic/cut")

# compulsory lines
trilmp.lmp.command("pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no")
trilmp.lmp.command("pair_coeff 1 1 table trimem_srp.table trimem_srp")

# set all interactions to zero just in case
trilmp.lmp.command("pair_coeff * * harmonic/cut 0 0")

# .................................................
#         COMPUTES, FIXES, ETC
# .................................................

# dump particle trajectories (vertex coordinates)
trilmp.lmp.command(f"dump XYZ all custom 1000 trajectory.dump id type x y z")

# print out the bonds (all bonds including the one we just added)
trilmp.lmp.command("compute MEMBONDS vertices property/local batom1 batom2")
trilmp.lmp.command(f"dump DMEMBONDS vertices local 1000 mem.bonds index c_MEMBONDS[1] c_MEMBONDS[2]")
trilmp.lmp.command("dump_modify DMEMBONDS format line '%d %0.0f %0.0f'")

# compute temperature of the membrane (vertices in the bulk)
trilmp.lmp.command("compute TempComputeMem BULK temp")

# print out all the computations
trilmp.lmp.command(f"fix aveMEM all ave/time 5000 1 5000 c_TempComputeMem file 'membrane_temperature.dat'")

# .................................................#
#       PRE-EQUILIBRATION INTEGRATION              #
# .................................................#

# include the integrators (pre-equilibration)
trilmp.lmp.command("fix NVEMEM BULK nve")
trilmp.lmp.command(f"fix LGVMEM BULK langevin 1.0 1.0 1.0 123 zero yes")

# !!!!!!!!!!!!!!!!!!!!!!!!!  #
# -------------------------  #
#    POST-EQUILIBRATION      #
# You could always add more  #
# LAMMPS commands once you   #
# have equilibrated your     #
# membrane. To do do,        #
# simply append your LAMMPS  #
# command to the             #
# postequilibration_commands #
# list.                      #
# -------------------------  #
# !!!!!!!!!!!!!!!!!!!!!!!!!  #

postequilibration_commands = []

# EXAMPLE: clean-up previously introduced fixes
postequilibration_commands.append("unfix NVEMEM")
postequilibration_commands.append("unfix LGVMEM")

# include them again
postequilibration_commands.append("fix NVEMEM BULK nve")
postequilibration_commands.append(f"fix LGVMEM BULK langevin 1.0 1.0 1.0 156 zero yes")

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
print(" - Add to the Ovito trajectory the mem.bonds file to visualize the network")
print(" - Check the performance of the code in inp_performance.dat")
print(" - Check the properties of the vesicle in inp_system.dat and membrane_properties.dat")
print(" - Reload pickle checkpoints from checkpoints/ckpt_*")
print("")
print("Have fun!")
print("")
