# TRILMP CLEAN-UP NOTES

Notes taken during and after the clean-up of TriLMP by Maitane Muñoz-Basagoiti.

### Cleanup-motivation: Extend TriLMP functionalities
FUNCTIONALITIES EXPECTED FROM TRILMP

1. SIMULATION OF TRIANGULATED MEMBRANE
2. TRIANGULATED MEMBRANE IN PRESENCE OF
  2.1. NANOPARTICLES
  2.2. ELASTIC MEMBRANE (S-LAYER)
  2.3. RIGID BODIES
3. EXTENDED SIMULATION FUNCTIONALITIES
  3.1. GCMC SIMULATIONS
  3.2. CHEMICAL REACTIONS

### Things that were taken out of ```__init__```

These are functionalities that were taken out of the ```__init__``` function. To recover the functionality, the commands have to be externally supplied to the LAMMPS object, or added as postequilibration commands.

*[taking out of init]* 

To perform dragging experiments, one must add to the lmp object:

```
self.lmp.commands_string(f"group NAMEGROUP id {id_dragged_particle}")
self.lmp.commands_string(f"fix NAMEFIX NAMEGROUP drag {x_drag} {y_drag} {z_drag} {magnitude_drag_force} {cutoff_distance_no_force}")
```

*[taking out of init]* 

To control the checking of neighbours just add to the lmp object

```
self.lmp.commands_string(f"neigh_modify every {self.check_neigh_every} check yes")
```

*[taking out of init]* 

When redefining the interactions, it always has to include:

```
[must be] pair_style hybrid/overlay table {self.eparams.repulse_params.lc1} linear 2000 + whatever else
[must be] pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no
[must be] pair_coeff 1 1 table trimem_srp.table trimem_srp
```

*[taking out of init]*
- Fix bond react: can be done totally externally (order does not matter)
- Fix gcmc regions
- Thermostats: can be defined before the run for the corresponding cases. NOTHING WILL MOVE IF NOT!
- If desired equilibration of metabolites, this must be defined after defining the class, and before doing the 'run' -- this way things don't clash
- Removing the 'basic system' code as not everyone can have the 'fix 1 all nve' from the beginning

### Additional commands after equilibration: ```postequilibration_lammps_commands```
Pass commands to the run that will be read as LAMMPS lines and inputed only when membrane is equilibrated

**ACHTUNG!**

When redefining the interactions, it always has to include:

```
[must be] pair_style hybrid/overlay table {self.eparams.repulse_params.lc1} linear 2000 + whatever else
[must be] pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no
[must be] pair_coeff 1 1 table trimem_srp.table trimem_srp
```

To include metabolites
- System must know how many types of metabolites there will be
- In general, it seems like a better idea that we tell the system how many particle types there will be. By default the membrane will be type 1 (```num_particle_types```)
- We define the masses per type (```mass_particle_type```)
- Assign each group a name (```group_particle_type```). By default group 0 has to be 'vertices', the vertices

Moved to 'run' - no place for them in initialization
check_outofrange = False,      # stop simulation if a single nanoparticle is out of range
check_outofrange_freq=100,     # how often check for nanoparticle out of range
check_outofrange_cutoff=100,   # when to consider nanoparticle too far away from the membrane


