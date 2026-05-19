# Implementation pair non-reciprocal
Implementation by Maitane Muñoz-Basagoiti (maitane.munoz-basagoiti@ista.ac.at).

## Arguments for the pair style
The latest implementation of a non-reciprocal interaction takes the following parameters as arguments (in the following order):

- Global arguments (4)
    1. ```cut_global``` (Global interaction cutoff) 
    2. ```int_scale``` (Interaction scale)
    3. ```exponent``` (Exponent of the interaction) 
    4. ```sigma_tilde``` (Sigma tilde for particle size)
- Pair coefficients (min 10, max 11)
    1. ```i``` (Particle type i)
    2. ```j``` (Particle type j)
    3. ```activity_i``` (Activity particle i)
    4. ```activity_j``` (Activity particle j)
    5. ```mobility_i``` (Mobility particle i)
    6. ```mobility_j``` (Mobility particle j)
    7. ```masscoll_i``` (Mass particle type i)
    8. ```masscoll_j``` (Mass particle type j)
    9. ```dampingcoll_i``` (Damping coefficient particle i)
    10. ```dampingcoll_j``` (Damping coefficient particle j)
    11. ```cut_one``` (Interaction cutoff) [Optional]


## Functional implementation
The implementation below allows the pair potential to reproduce phoretic velocities in the far field approximation when the dynamics of the simulation are overdamped. 

### Justification for implementation

We describe the motion of a particle in solution with the Langevin equation
$$
m_i \ddot{\vec{r}_i} = \vec{F}_i - \gamma_i \dot{\vec{r}_i} + \eta \vec{G}(0, 1),
$$
where $\vec{G}$ is a vector sampled from a Gaussian distribution with mean 0 and variance 1, and $\eta$ represents the amplitude of the white noise, which is coupled to the friction coefficient $\gamma_i$ through the fluctuation-dissipation theorem. In the overdamped regime $m_i/\gamma_i \to 0$, and the Langevin equation simplifies to
$$
\dot{\vec{r}_i} = \frac{1}{\gamma_i} \vec{F}_i + \frac{\eta}{\gamma_i} \vec{G}(0, 1)
$$
LAMMPS' implementation of the Langevin equation writes the right hand side of the Langevin equation as follows:
$$
F = F_c + F_f + F_r \text{   with   } F_r = -\frac{m}{damp} v,
$$
where m is the mass of the particle and damp is the damping factor, specified in time units. LAMMPS' documentation tells us that we can think about it as inversely related to the viscosity of the solvent. Therefore, in reality the Langevin equation that LAMMPS is integrating is
$$
m_i \ddot{\vec{r}_i} = \vec{F}_i - \frac{m_i}{damp_i}\dot{\vec{r}_i} + \sqrt{\frac{k_B T}{dt~ damp}} \vec{G}(0, 1).
$$
A way of implementing a phoretic velocity in LAMMPS is therefore to take
$$
\vec{F}_i = \frac{m_i}{damp_i} \vec{v}_{ph}
$$
so that the Langevin equation seen by LAMMPS is
$$
m_i \ddot{\vec{r}_i} = \frac{m_i}{damp_i} \vec{v}_{ph} - \frac{m_i}{damp_i}\dot{\vec{r}_i} + \sqrt{\frac{k_B T}{dt~ damp}} \vec{G}(0, 1).
$$
In the limit where inertia is negligible (one can thing about it as the damping coefficient being very small or the friction coefficient being very large), the equation becomes
$$
\dot{\vec{r}_i} = \vec{v}_{ph} +\tilde{\eta}\vec{G}(0, 1),
$$
that is, the velocity of particle $i$ is the phoretic velocity and random deviations from it. 

### Implementation details

The implementation you will find in this directory follows this basic structure:
```
Select particle i
Extract xi, yi, zi, typei

For j in neighbours_particle_i
    Extract xj, yj, zj, typej

    Compute r2 = (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2
    if r2<cutsq

        Compute r = sqrt(r2)
        Compute rinv = 1/r
        Compute rexpin = (sigma_tilde/r)^exponent

        Compute unit vectors for direction of forces
            delx/r, dely/r, delz/r
        
        Compute coefficients for force
            catcoff_i = activity_i * mobility_j * int_scale
            catcoff_j = activity_j * mobility_i * int_scale

        Compute forces
            fxi = catcoff_j * rexpin * unitvec_x
            fxj = - catcoff_i * rexpin * unitvec_x

        Update forces
            f[i][0] += fxi * (masscoll_i/damping_i)
            f[j][0] += fxj * (masscoll_j/damping_j)
```

**Important note**

This potential must be coupled to some form of volume exclusion or repulsion. LAMMPS fixes that can help with that are ```pair_style lj/cut``` or ```pair_style cosine_square``` in the WCA form, or ```pair_style harmonic/cut``` for harmonic repulsion.

### Verification of the implementation

The specific implementation above has been tested by simulating two catalytically coated colloidal particles interacting with each other in a non-reciprocal manner and comparing the velocity of the center of mass of the system. Tests have been run both using ```fix langevin``` and ```fix brownian```. The outcome of both tests agrees with each other and with the theoretical expression.

Additionally, the implementation of the pair style has also been tested with regards to neighbour list usage. By substituting the force calculation lines for a lennard-jones potential, we compared two simulation trajectories with the same initial configuration and identical random number seeds, one run with ```pair_style lj/cut``` and one with ```pair_style nonreciprocal```. Both trajectories were identical.

### Old implementation: Lennard-Jones like non-reciprocal pair style
A previous implementation did not multiply the forces by the ratio between the mass and the damping coefficient. The rest of the pair potential, however, looked identical. The motivation behind this first implementation was to have a non-reciprocal equivalent of a Lennard-Jones type of interaction. For that purpose, we considered the purely attractive potential
$$
U(r) = -\varepsilon \left( \frac{\sigma}{r}\right)^6.
$$
The magnitude of the force that derives from this potential is
$$
F = - \frac{d U(r)}{dr} = - 6 \varepsilon \frac{\sigma^6}{r^7} = -6\frac{\varepsilon}{\sigma} \left( \frac{\sigma}{r}\right)^7.
$$
This is why the ```int_scale``` chosen in many simulations was taken as
```
int_scale = (exponent-1)/sigma_tilde
```
with ```exponent = 7```. In principle, the $\varepsilon$ in the expresion was split in two components, activity and mobility (as seen in the implementation above). Nonetheless, in practice mobility was always set to -1, and it was the activity the one that one determine $\varepsilon$.

## Far-field vs Near-field implementation
While the far-field phoretic velocities are

$$ 
\vec{v}_1 = \mu_1 \alpha_2 \frac{R^2}{D (\Delta + 2R)^2} \vec{e} 
$$

$$ 
\vec{v}_2 = -\mu_2 \alpha_1 \frac{R^2}{D (\Delta + 2R)^2} \vec{e} 
$$

where $\vec{e}$ is a unit vector that points from 2 to 1, the expression for the near-field phoretic velocities is

$$
\vec{v}_1 = \gamma(\Delta) [\mu_1 \alpha_1 \varepsilon(\Delta) + \alpha_2 \mu_1] \vec{e}
$$

$$
\vec{v}_2 = -\gamma(\Delta) [\mu_2 \alpha_2 \varepsilon(\Delta) + \alpha_1 \mu_2] \vec{e}
$$

where $\gamma(\Delta)$ and $\varepsilon(\Delta)$ come from the tables computed by Mike.
