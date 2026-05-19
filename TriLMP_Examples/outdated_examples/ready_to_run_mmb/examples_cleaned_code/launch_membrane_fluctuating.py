# -----------------------------------------------------------------#
# This code launches TriLMP (clean version) for a fluid membrane.  #
# Notes:                                                           #
# - By default, membrane group is called 'vertices'.               #
# -----------------------------------------------------------------#

import trimesh
import numpy as np
import pandas as pd
from trimem.mc.trilmp import TriLmp
 
# initialization of the membrane mesh using Python's trimesh library
mesh = trimesh.creation.icosphere(4)
print(f"MESH VERTICES : {len(mesh.vertices)}")
print(f"MESH FACES    : {len(mesh.faces)}")
print(f"MESH EDGES    : {len(mesh.edges)}")

# rescaling mesh distances (necessary due to hard-coded values in TriLMP)
sigma=1.0
desired_average_distance = 2**(1.0/6.0) * sigma
current_average_distance = np.mean(mesh.edges_unique_length)
scaling = desired_average_distance/current_average_distance
mesh.vertices *=scaling
 
# membrane mechanical properties
kappa_b = 20.0
kappa_a = 1.0e6
kappa_v = 0.0
kappa_c = 0.0
kappa_t = 1.0e4
kappa_r = 1.0e3
 
# MD properties
step_size = 0.001
traj_steps = 50
langevin_damp = 1.0
temperature   = 1.0
langevin_seed = 123
 
# MC/TRIMEM bond flipping properties
flip_ratio=0.1
switch_mode='random'

# simulation step structure (given in time units)
total_sim_time=500
MD_simulation_steps=int(total_sim_time/step_size)
 
# ouput and printing (given in time units)
discret_snapshots=10
print_frequency = int(discret_snapshots/step_size)

# output from TriLMP (MC+MD)
print_program_iterations = 10

# simulation box
xlo = -30
xhi = 30
ylo = -30
yhi = 30
zlo = -30
zhi = 30

# initialization of the trilmp object
trilmp=TriLmp(initialize=True,                           # use mesh to initialize mesh reference
              debug_mode=True,                          # DEBUGGING: print everything
              num_particle_types=1,                      # PART. SPECIES: total particle species in system 
              mass_particle_type=[1],                    # PART. SPECIES: mass of species in system
              group_particle_type=['vertices'],          # PART. SPECIES: group names for species in system

              mesh_points=mesh.vertices,                  # input mesh vertices 
              mesh_faces=mesh.faces,                      # input of the mesh faces

              kappa_b=kappa_b,                            # MEMBRANE MECHANICS: bending modulus (kB T)
              kappa_a=kappa_a,                            # MEMBRANE MECHANICS: constraint on area change from target value (kB T)
              kappa_v=kappa_v,                            # MEMBRANE MECHANICS: constraint on volume change from target value (kB T)
              kappa_c=kappa_c,                            # MEMBRANE MECHANICS: constraint on area difference change (kB T)
              kappa_t=kappa_t,                            # MEMBRANE MECHANICS: tethering potential to constrain edge length (kB T)
              kappa_r=kappa_r,                            # MEMBRANE MECHANICS: repulsive potential to prevent surface intersection (kB T)
              
              step_size=step_size,                        # FLUIDITY ---- MD PART SIMULATION: timestep of the simulation
              traj_steps=traj_steps,                      # FLUIDITY ---- MD PART SIMULATION: number of MD steps before bond flipping
              flip_ratio=flip_ratio,                      # MC PART SIMULATION: fraction of edges to flip
              initial_temperature=temperature,            # MD PART SIMULATION: temperature of the system
              pure_MD=True,                               # MD PART SIMULATION: accept every MD trajectory
              switch_mode=switch_mode,                    # MD/MC PART SIMULATION: 'random' or 'alternating' flip-or-move
              box=(xlo, xhi, ylo, yhi, zlo, zhi),         # MD PART SIMULATION: simulation box properties, periodic
              
              info=print_frequency,                              # OUTPUT: frequency output in shell
              thin=print_frequency,                              # OUTPUT: frequency trajectory output
              output_counter=0,                                  # OUTPUT: initialize trajectory number in writer class
              performance_increment=print_program_iterations,    # OUTPUT: output performace stats to prefix_performance.dat file - PRINTED MD+MC FREQUENCY
              energy_increment=print_program_iterations,         # OUTPUT: output energies to energies.dat file - PRINTED MD FREQUENCY
              checkpoint_every=100*print_frequency,              # OUTPUT: interval of checkpoints (alternating pickles) - PRINTED MD+MC FREQUENCY
              )

# -------------------------#
#  LAMMPS MODIFICATIONS    #
# -------------------------#

# edit neighbour list skin when only membrane (in case - maybe not needed)
trilmp.lmp.commands_string("neighbor 1.0 bin")

# -------------------------#
#           DUMPS          #
# -------------------------#

# dump particle trajectories (vertex coordinates)
trilmp.lmp.commands_string(f"dump MEMXYZ vertices custom {print_frequency} trajectory.gz id type x y z")

# dump bonds (network edges)
#trilmp.lmp.commands_string("compute MEMBONDS all property/local batom1 batom2")
#trilmp.lmp.commands_string(f"dump DMEMBONDS all local {print_frequency} mem.bonds index c_MEMBONDS[1] c_MEMBONDS[2]")
#trilmp.lmp.commands_string("dump_modify DMEMBONDS format line '%d %0.0f %0.0f'")

# -------------------------#
#         COMPUTES         #
# -------------------------#

# compute potential energy
trilmp.lmp.commands_string("compute PeMembrane vertices pe/atom pair")
trilmp.lmp.commands_string("compute pe vertices reduce sum c_PeMembrane")

# compute position CM vesicle
trilmp.lmp.commands_string("compute MembraneCOM vertices com")

# compute shape of the vesicle
trilmp.lmp.commands_string("compute RadiusGMem vertices gyration")
trilmp.lmp.commands_string("compute MemShape vertices gyration/shape RadiusGMem")

# compute temperature of the vesicle
trilmp.lmp.commands_string("compute TempComputeMem vertices temp")

# print out all the computations
trilmp.lmp.commands_string(f"fix  aveMEM all ave/time {print_frequency} 1 {print_frequency} c_TempComputeMem c_pe c_MembraneCOM[1] c_MembraneCOM[2] c_MembraneCOM[3] c_MemShape[1] c_MemShape[2] c_MemShape[3] c_MemShape[4] c_MemShape[5] c_MemShape[6] file 'membrane_CM.dat'")

# -------------------------#
#         INTEGRATORS      #
# -------------------------#

# include the integrators (pre-equilibration)
trilmp.lmp.commands_string("fix NVEMEM vertices nve")
trilmp.lmp.commands_string(f"fix LGVMEM vertices langevin {temperature} {temperature} {langevin_damp} {langevin_seed} zero yes")

# -------------------------#
#    POST-EQUILIBRATION    #
# -------------------------#

# -------------------------#
#         RUN              #
# -------------------------#

# RUN THE SIMULATION
trilmp.run(MD_simulation_steps, integrators_defined=True, fix_symbionts_near=False)

print("End of the simulation.")