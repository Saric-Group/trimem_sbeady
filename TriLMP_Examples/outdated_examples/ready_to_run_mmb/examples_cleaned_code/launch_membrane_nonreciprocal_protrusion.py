# -----------------------------------------------------------------#
# This code launches TriLMP (clean version) for a fluid membrane.  #
# Additionally, the system also includes a particle (bead)         #
# that interacts in a non-reciprocal manner with the membrane      #
# Notes:                                                           #
# - By default, membrane group is called 'vertices'.               #
# - This example uses a vesicle initialization from the group      #
# -----------------------------------------------------------------#

import trimesh
import numpy as np
import pandas as pd
from trimem.mc.trilmp import TriLmp
 
# ****************************#
#     MEMBRANE PROPERTIES     #
# ****************************#

# initialization of the membrane mesh using existing mesh
N = 5072
mesh_coordinates = pd.read_csv(f'../../MembraneInitialization/mesh_coordinates_N_{N}_.dat', header = None, index_col = False, sep = ' ')
mesh_coordinates_array = mesh_coordinates[[1, 2, 3]].to_numpy()
mesh_faces = pd.read_csv(f'../../MembraneInitialization/mesh_faces_N_{N}_.dat', header = None, index_col = False, sep = ' ')
mesh_faces_array = mesh_faces[[0, 1, 2]].to_numpy()
mesh = trimesh.Trimesh(vertices=mesh_coordinates_array, faces = mesh_faces_array) 
print(f"MESH VERTICES : {len(mesh.vertices)}")
print(f"MESH FACES    : {len(mesh.faces)}")
print(f"MESH EDGES    : {len(mesh.edges)}")

# rescaling mesh distances
sigma=1.0
desired_average_distance = 2**(1.0/6.0) * sigma
current_average_distance = np.mean(mesh.edges_unique_length)
scaling = desired_average_distance/current_average_distance
mesh.vertices *=scaling
 
# membrane mechanical properties
kappa_b = 20.0
kappa_a = 2.5e5
kappa_v = 2.5e5
kappa_c = 0.0
kappa_t = 1.0e4
kappa_r = 1.0e3
 
# ****************************#
#   SIMULATION PROPERTIES     #
# ****************************#

# MD properties
step_size     = 0.001
traj_steps    = 50
langevin_damp = 1.0
temperature   = 1.0
langevin_seed = 123
 
# MC/TRIMEM bond flipping properties
flip_ratio=0.1
switch_mode='random'

# simulation step structure (given in time units)
total_sim_time=1000
MD_simulation_steps= int(total_sim_time/step_size)
 
# ouput and printing (given in time units)
discret_snapshots=5
print_frequency = int(discret_snapshots/step_size)

# output from TriLMP (MC+MD, given in program steps) 
print_program_iterations = 10

# simulation box
xlo = -50
xhi = 50
ylo = -50
yhi = 50
zlo = -50
zhi = 50

# equilibration steps for the membrane (how many MD stages need to pass)
eq_rounds = 100
equilibration_steps = eq_rounds*traj_steps

# ****************************#
#     SYMBIONT PROPERTIES     #
# ****************************#

# symbiont properties
mass_symbiont  = 1.0
sigma_symbiont = 5.0
sigma_tilde    = 0.5*(1+sigma_symbiont)

# global parameters for the non-reciprocal interaction
nr_cutoff = 1.5
nr_exponent = 2
nr_scale  = (nr_exponent-1)/sigma_tilde
nr_cutoff_tilde = nr_cutoff*sigma_tilde
check_outofrange_cutoff=1.1*nr_cutoff_tilde

# local parameters for the non-reciprocal interaction
activity_1 = 10.0
activity_2 = 20.0
mobility_1 = -1.0
mobility_2 = -1.0
mass_1     = 1.0
mass_2     = mass_symbiont
damping_1  = 1.0
damping_2  = 1.0

# harmonic repulsion
kharmonic = 1000
rcharmonic = sigma_tilde

# initialization of the trilmp object
trilmp=TriLmp(initialize=True,                                  # use mesh to initialize mesh reference
              debug_mode=False,                                  # DEBUGGING: print everything
              num_particle_types=2,                             # PART. SPECIES: total particle species in system 
              mass_particle_type=[1.0, mass_symbiont],          # PART. SPECIES: mass of species in system
              group_particle_type=['vertices', 'symbionts'],    # PART. SPECIES: group names for species in system

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

              equilibration_rounds=equilibration_steps,   # MD PART SIMULATION: HOW LONG DO WE LET THE MEMBRANE EQUILIBRATE
              
              info=print_frequency,                              # OUTPUT: frequency output in shell
              thin=print_frequency,                              # OUTPUT: frequency trajectory output
              performance_increment=print_program_iterations,    # OUTPUT: output performace stats to prefix_performance.dat file - PRINTED MD+MC FREQUENCY
              energy_increment=print_program_iterations,         # OUTPUT: output energies to energies.dat file - PRINTED MD FREQUENCY
              checkpoint_every=100*print_frequency,              # OUTPUT: interval of checkpoints (alternating pickles) - PRINTED MD+MC FREQUENCY

              n_beads=1,                          # NUMBER OF EXTERNAL BEADS
              n_bead_types=1,                     # NUMBER OF EXTERNAL BEAD TYPES
              bead_pos=np.array([[0, 0, 0]]),     # POSITION OF THE EXTERNAL BEADS
              bead_types=np.array([2]),           # BEAD TYPES (By default type 2 when single particle)
              bead_sizes=np.array([sigma_symbiont]),
              )

# -------------------------#
#  LAMMPS MODIFICATIONS    #
# -------------------------#

# edit neighbour list skin when only membrane (in case - maybe not needed)
#trilmp.lmp.command("neighbor 1.0 bin")

# -------------------------#
#           DUMPS          #
# -------------------------#

# dump particle trajectories (vertex coordinates)
trilmp.lmp.command(f"dump XYZ all custom {print_frequency} trajectory.gz id x y z type")

# dump bonds (network edges) [NOT USEFUL AT THE MOMENT]
#trilmp.lmp.command("compute MEMBONDS vertices property/local batom1 batom2")
#trilmp.lmp.command(f"dump DMEMBONDS vertices local {print_frequency} mem.bonds index c_MEMBONDS[1] c_MEMBONDS[2]")
#trilmp.lmp.command("dump_modify DMEMBONDS format line '%d %0.0f %0.0f'")

# -------------------------#
#         COMPUTES         #
# -------------------------#

# compute potential energy
trilmp.lmp.command("compute PeMembrane vertices pe/atom pair")
trilmp.lmp.command("compute pe vertices reduce sum c_PeMembrane")

# compute position CM vesicle
trilmp.lmp.command("compute MembraneCOM vertices com")

# compute shape of the vesicle
trilmp.lmp.command("compute RadiusGMem vertices gyration")
trilmp.lmp.command("compute MemShape vertices gyration/shape RadiusGMem")

# compute temperature of the vesicle
trilmp.lmp.command("compute TempComputeMem vertices temp")

# print out all the computations
trilmp.lmp.command(f"fix  aveMEM all ave/time {print_frequency} 1 {print_frequency} c_TempComputeMem c_pe c_MembraneCOM[1] c_MembraneCOM[2] c_MembraneCOM[3] c_MemShape[1] c_MemShape[2] c_MemShape[3] c_MemShape[4] c_MemShape[5] c_MemShape[6] file 'membrane_CM.dat'")

# -------------------------#
#         INTEGRATORS      #
# -------------------------#

# include the integrators (pre-equilibration)
trilmp.lmp.command("fix NVEMEM vertices nve")
trilmp.lmp.command(f"fix LGVMEM vertices langevin {temperature} {temperature} {langevin_damp} {langevin_seed} zero yes")

# cleanup for afterwards - note that passed as postequilibration commands
postequilibration_commands = []
postequilibration_commands.append("unfix NVEMEM")
postequilibration_commands.append("unfix LGVMEM")
postequilibration_commands.append("pair_style none")

# -------------------------#
#    POST-EQUILIBRATION    #
# -------------------------#

# pair interactions
postequilibration_commands.append(f"pair_style hybrid/overlay table linear 2000 nonreciprocal {nr_cutoff_tilde} {nr_scale} {nr_exponent} {sigma_tilde} harmonic/cut")

# compulsory lines
postequilibration_commands.append("pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no")
postequilibration_commands.append("pair_coeff 1 1 table trimem_srp.table trimem_srp")

# set all interactions to zero just in case for added potentials (careful with the mass and damping values)
postequilibration_commands.append("pair_coeff * * nonreciprocal 0 0 0 0 1 1 1 1")
postequilibration_commands.append("pair_coeff * * harmonic/cut 0 0")

# pair coefficients for non-reciprocal 
postequilibration_commands.append(f"pair_coeff 1 2 nonreciprocal {activity_1} {activity_2} {mobility_1} {mobility_2} {mass_1} {mass_2} {damping_1} {damping_2}")

# pair coefficients for harmonic repulsion
postequilibration_commands.append(f"pair_coeff 1 2 harmonic/cut {kharmonic} {rcharmonic}")

# integrators
postequilibration_commands.append("fix NVEALL all nve")
postequilibration_commands.append(f"fix LGVALL all langevin {temperature} {temperature} {langevin_damp} {langevin_seed} zero yes scale 2 {mass_symbiont/sigma_symbiont}")

# -------------------------#
#         RUN              #
# -------------------------#

# RUN THE SIMULATION
trilmp.run(MD_simulation_steps, integrators_defined=True, fix_symbionts_near=True, 
           postequilibration_lammps_commands = postequilibration_commands,
           check_outofrange=True,
           check_outofrange_freq=100,
           check_outofrange_cutoff=check_outofrange_cutoff)

print("End of the simulation.")
