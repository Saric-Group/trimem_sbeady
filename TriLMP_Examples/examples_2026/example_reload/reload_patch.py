# ---------------------------------------------------------------------#
# TEST TO VERIFY TRILMP INSTALLATION                                   #
# Author: Maitane Muñoz-Basagoiti (maitane.munoz-basagoiti@ista.ac.at) #
#                                                                      #
# This code relaunches TriLMP for a fluid membrane patch.              #
# For technical reasons, you need to include the commands that you     #
# used for LAMMPS again. The reloading only considers the mesh, its    #
# connectivity and general properties (bending modulus, area...).      #
# The reloading completely forgets about integrators, thermostats...   #
#                                                                      #
# OUTPUTS:                                                             #
# - TriLMP prints:                                                     #
#   + Total Trimem energy (see inp_system.dat)                         #
#   + Area of the mesh (see inp_system.dat)                            #
#   + Bending energy of the mesh (see inp_system.dat)                  #
#   + Performance information (see inp_performance.dat)                #
# ---------------------------------------------------------------------#

import trimem.core as m
from trimem.mc.trilmp import *

# load the checkpoint
loaded_checkpoint = read_checkpoint('ckpt_reloadpatch.pickle')
loaded_checkpoint.equilibration_rounds = loaded_checkpoint.traj_steps
loaded_checkpoint.equilibrated = False
print('This is the reloaded the file:', loaded_checkpoint.restart_file)
print('These are the atoms: ', loaded_checkpoint.mesh_points.shape)
wait = input()
# -------------------------#
#  LAMMPS MODIFICATIONS    #
# -------------------------#

# .................................................
#                GROUPS
# .................................................

loaded_checkpoint.lmp.command('group vertex_edge id 1')
loaded_checkpoint.lmp.command('group vertex_edge id 2')
loaded_checkpoint.lmp.command('group vertex_edge id 3')
loaded_checkpoint.lmp.command('group vertex_edge id 4')
loaded_checkpoint.lmp.command('group vertex_edge id 5')
loaded_checkpoint.lmp.command('group vertex_edge id 6')
loaded_checkpoint.lmp.command('group vertex_edge id 7')
loaded_checkpoint.lmp.command('group vertex_edge id 8')
loaded_checkpoint.lmp.command('group vertex_edge id 9')
loaded_checkpoint.lmp.command('group vertex_edge id 10')
loaded_checkpoint.lmp.command('group vertex_edge id 11')
loaded_checkpoint.lmp.command('group vertex_edge id 20')
loaded_checkpoint.lmp.command('group vertex_edge id 21')
loaded_checkpoint.lmp.command('group vertex_edge id 30')
loaded_checkpoint.lmp.command('group vertex_edge id 31')
loaded_checkpoint.lmp.command('group vertex_edge id 40')
loaded_checkpoint.lmp.command('group vertex_edge id 41')
loaded_checkpoint.lmp.command('group vertex_edge id 50')
loaded_checkpoint.lmp.command('group vertex_edge id 51')
loaded_checkpoint.lmp.command('group vertex_edge id 60')
loaded_checkpoint.lmp.command('group vertex_edge id 61')
loaded_checkpoint.lmp.command('group vertex_edge id 70')
loaded_checkpoint.lmp.command('group vertex_edge id 71')
loaded_checkpoint.lmp.command('group vertex_edge id 80')
loaded_checkpoint.lmp.command('group vertex_edge id 81')
loaded_checkpoint.lmp.command('group vertex_edge id 90')
loaded_checkpoint.lmp.command('group vertex_edge id 91')
loaded_checkpoint.lmp.command('group vertex_edge id 100')
loaded_checkpoint.lmp.command('group vertex_edge id 101')
loaded_checkpoint.lmp.command('group vertex_edge id 102')
loaded_checkpoint.lmp.command('group vertex_edge id 103')
loaded_checkpoint.lmp.command('group vertex_edge id 104')
loaded_checkpoint.lmp.command('group vertex_edge id 105')
loaded_checkpoint.lmp.command('group vertex_edge id 106')
loaded_checkpoint.lmp.command('group vertex_edge id 107')
loaded_checkpoint.lmp.command('group vertex_edge id 108')
loaded_checkpoint.lmp.command('group vertex_edge id 109')
loaded_checkpoint.lmp.command('group vertex_edge id 110')

loaded_checkpoint.lmp.command("group BULK subtract vertices vertex_edge")

# .................................................
#            PAIR STYLES 
# .................................................

# cleanup pair style in case
loaded_checkpoint.lmp.command("pair_style none")

# pair interactions
loaded_checkpoint.lmp.command(f"pair_style hybrid/overlay table linear 2000 harmonic/cut")

# compulsory lines
loaded_checkpoint.lmp.command("pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no")
loaded_checkpoint.lmp.command("pair_coeff 1 1 table trimem_srp.table trimem_srp")

# set all interactions to zero just in case
loaded_checkpoint.lmp.command("pair_coeff * * harmonic/cut 0 0")

# .................................................
#         COMPUTES, FIXES, ETC
# .................................................

# dump particle trajectories (vertex coordinates)
loaded_checkpoint.lmp.command(f"dump XYZ all custom 1000 trajectory.dump id type x y z")

# print out the bonds (all bonds including the one we just added)
loaded_checkpoint.lmp.command("compute MEMBONDS vertices property/local batom1 batom2")
loaded_checkpoint.lmp.command(f"dump DMEMBONDS vertices local 1000 mem.bonds index c_MEMBONDS[1] c_MEMBONDS[2]")
loaded_checkpoint.lmp.command("dump_modify DMEMBONDS format line '%d %0.0f %0.0f'")

# compute temperature of the membrane (vertices in the bulk)
loaded_checkpoint.lmp.command("compute TempComputeMem BULK temp")

# print out all the computations
loaded_checkpoint.lmp.command(f"fix aveMEM all ave/time 5000 1 5000 c_TempComputeMem file 'membrane_temperature.dat'")

# .................................................#
#       PRE-EQUILIBRATION INTEGRATION              #
# .................................................#

# include the integrators (pre-equilibration)
loaded_checkpoint.lmp.command("fix NVEMEM BULK nve")
loaded_checkpoint.lmp.command(f"fix LGVMEM BULK langevin 1.0 1.0 1.0 123 zero yes")

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
# MD simulation time step
dt = 0.001
SimTime = MDsteps*dt
loaded_checkpoint.run(MDsteps)

print("")
print("*** End of the simulation ***")
print("")
print(f"You have relaunched an ongoing simulation patch for {SimTime} simulation time units.")
print("Now you can: ")
print(" - Load sim_setup.in in Ovito to check the initial configuration")
print(" - Load trajectory.dump in Ovito to check the simulated trajectory")
print(" - Add to the Ovito trajectory the mem.bonds file to visualize the network")
print(" - Check the performance of the code in inp_performance.dat")
print(" - Check the properties of the vesicle in inp_system.dat and membrane_properties.dat")
print(" - Reload pickle checkpoints from ckptA.pickle or ckptB.pickle (last snapshot).")
print("")
print("Have fun!")
print("")
