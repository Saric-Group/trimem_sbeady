# User's guide to TriLMP

This document is a description of the code in ```trilmp.py``` (within ```src/trimem/mc```).

## The TriLMP class

### The ```__init___``` method

The ```__init__``` method contains the minimal attributes required to initialize the simulation. It sets up the system and gets it ready so that we can pass commands using LAMMPS externally. 

The dynamically triangulated mesh is a ```Mesh``` class from TriMEM (see documentation for class [here](https://trimem.readthedocs.io/en/latest/trimem.mc.html#trimem.mc.mesh.Mesh)). To initialize it, you need to pass the coordinates of the membrane vertices as well as the faces of the triangulation. You can do that by using the Python library ```trimesh``` or by turning to the in-house C code.

TriLMP assumes that the average length of the edges in your network is $2^{1/6}\sigma$ with $\sigma=1$. It is important to take this into account because upon initialization, the edge length is constrained by:

- $l_{\min} = l_{c1} = 1$
- $l_{\max} = l_{c0} = \sqrt{3}$

If the edge length goes beyond these values, then the vertices at the end of the edges are subjected to a force that will push them together or apart. The general lengthscale of the system, given by the TriMEM parameter $l_0$ is also hardcoded to 1.

TriLMP uses TriMEM to compute the forces acting on the vertices of the mesh according to the Hamiltonian presented in the TriMEM paper. We consider all contributions in the Hamiltonian except for the contribution related to surface repulsion, which is taken care as a ```table pair_style``` using LAMMPS. The repulsion only acts between vertices that are not directly connected through 1-2, 1-3 and 1-4 neighbour interactions. The TriMEM forces are passed to LAMMPS by calling the function that calls the gradient from TriMEM and updating the force array that LAMMPS is taking care of.

## A note on paralelization
OpenMP is for shared-memory computing. MPI is for distributed memory. LAMMPS was written for distributed memory (the code can run in several nodes at the same time); TriMEM was written for shared-memory (the code runs in a single now distributed among many threads therein). TriMEM could be made much faster if it can be incorporated to LAMMPS to profit from the multinode capability.
