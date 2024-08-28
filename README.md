# TriLMP

**TriLMP** is a modified version of the [**TriMEM**](https://github.com/bio-phys/trimem) python package. It performs Hybrid Monte Carlo (HMC) simulations of lipid
membranes according to the Helfrich theory[^Helfrich1973]. Please refer to Siggel et al.[^Siggel2022] for details on the hamiltonian simulated.

## Basic introduction
TriLMP couples TriMEM to [LAMMPS](https://github.com/lammps/lammps) via the python interface of the latter. Within TriLMP, the Dynamically Triangulated network (DTN) is a LAMMPS molecule made of particles (network vertices) that are connected through bonds (network edges). Throughout the simulation, TriLMP uses TriMEM to compute the Helfrich Hamiltonian as well as to obtain the list of vertices that have to be updated to ensure membrane fluidity. At each MD simulation step, TriMEM provides the force acting on each vertex in the network to LAMMPS so that vertices in the network move according to LAMMPS' time-integrators. At a prescribed frequency (see TriLMP docs), TriLMP informs LAMMPS to update the network connectivity by removing and creating new bonds between particles in the molecule. Additionally, the calculation of the surface repulsion, which prevents the network from self-intersecting, is dealt with by LAMMPS instead of TriMEM. For further details on how to use the package, please refer to the HowTo_TriLMP.md file.

[^Helfrich1973]: Helfrich, W. (1973) Elastic properties of lipid bilayers:
  Theory and possible experiments. Zeitschrift für Naturforschung C,
  28(11), 693-703

## Installation

TriLMP has been successfully installed and tested in:
- Linux
- Mac (?) Pending

We suggest installing TriLmp in a conda environment and then install trimem/trilmp as a editable install.
This allows you to alter trilmp functionality by directly tinkering with the src/trimem/trilmp.py file if needed.
This can come in handy in case you need to define or initialize a simulation set-up that is not covered by the current version of TriLMP.
The code is designed such that additional LAMMPS commands can always be passed as command strings once the ```trilmp``` object has been created.

### Conda Environment

To set up a conda environment including some prerequisites use:

```bash
conda create -n Trienv
conda install -n Trienv scikit-build libgcc pybind11
conda activate Trienv
```

Make sure the evironment is activated throughout the rest of the setup!

### TriMEM/TriLMP
TriLMP can be installed using pip.
Note that this folder will be the actual location of the modules with an editable install, i.e., if you change something in the python code here, effects will be immediate.
In case you want to change something on the c++ side of the code, make sure to run "python3 setup.py build" in order to to compile and copy 
the libaries to the src/trimem folder as described below.

Clone and install TriLMP by running:
```bash
 git clone --recurse-submodules https://github.com/Saric-Group/trimem_sbeady.git
 
 cd trimem_sbeady
 git submodule update
 pip install -e .
```

In case they are not built during install one can compile the shared libraries using
```bash
python3 setup.py build
```

Finally some shared libraries have to be copied manually from the _skbuild folder. 
The names you see below depend on system and python version! Note that all the .so files should be placed in the src/trimem folder.

```bash
cp _skbuild/linux-x86_64-3.11/cmake-build/core.cpython-311-x86_64-linux-gnu.so src/trimem/.
cp _skbuild/linux-x86_64-3.11/cmake-build/libtrimem.so src/trimem/.
cp _skbuild/linux-x86_64-3.11/cmake-build/Build/lib/* src/trimem/.
```

### LAMMPS
LAMMPS has to be installed using python shared libraries and some specific packages.
If you already have LAMMPS you might have to reinstall it. In any case you will have to 
perform the python install using your Trienv environment as we detail below.

Go back to the directory where you want to have your install folder and get lammps from git by running

```bash
git clone -b stable https://github.com/lammps/lammps.git lammps
```

We can now start installing LAMMPS. Go first to the ```lammps``` directory and create the ```build``` folder.
```bash
cd lammps                
mkdir build
cd build
```

Next we have to determine the python executable path. You can do that from the python console
using
```bash
python3              
import sys
print(sys.executable)
```
which should print something like 

```bash
/nfs/scistore15/saricgrp/Your_folder/.conda/envs/Trienv/bin/python3
```
and will be referred to as "python_path" in the next bash command. Note that this enables PyLammps in this specific conda environment only! 

Now we can set up the makefiles and build LAMMPS. The command line you see below is a suggestion for the LAMMPS packages you may want to compile for tests (these are suggested here because they have proved useful in the past in the development of works involving TriLMP).

```
cmake -D BUILD_OMP=yes -D BUILD_SHARED_LIBS=yes -D PYTHON_EXECUTABLE="/nfs/scistore15/saricgrp/mmunozba/.conda/envs/Trienv/bin/python3" -D PKG_PYTHON=yes -D PKG_OPENMP=yes -D PKG_MOLECULE=yes  -D PKG_EXTRA-PAIR=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-COMPUTE=yes -D PKG_RIGID=yes -D PKG_ASPHERE=yes -D PKG_BROWNIAN=yes -D PKG_MC=yes -D PKG_REACTION=yes ../cmake 
cmake --build .
```

Roughly, this is what some of these packages can be useful for:
- ```EXTRA-PAIR``` $\rightarrow$ ```pair_style harmonic/cut```, ```pair_style cosine/squared```
- ```EXTRA-COMPUTE``` $\rightarrow$ ```compute gyration/shape```
- ```RIGID``` $\rightarrow$ ```fix rigid```
- ```ASPHERE``` $\rightarrow$ ```atom_style ellipsoid```
- ```MC``` $\rightarrow$ ```fix gcmc```, ```fix bond/create```
- ```REACTION``` $\rightarrow$ ```fix bond/react```

If you are missing specific packages or you want to extend your installation, you can always come back and recompile LAMMPS with additional packages.

For HPC clusters: If you try the above compilation and it is not working, try loading the openmpi module first by typing ```module load openmpi``` in the cluster.

Finally to make LAMMPS accessible for python, i.e. making and copying the shared libaries, use

```bash
make install-python
```

### Dependencies


This information has been copied from TriMEM's documentation. 

TriMEM builds upon the generic mesh data structure
[OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/), which
is included as a submodule that is pulled in upon `git clone` via the
`--recurse-submodules` flag.

For the efficient utilization of shared-memory parallelism, triMEM makes
use of the [OpenMP](https://www.openmp.org/) application programming model
(`>= v4.5`) and modern `C++`. It thus requires relatively up-to-date
compilers (supporting at least `C++17`).

If not already available, the following python dependencies will be
automatically installed:

* numpy
* scipy
* h5py
* meshio
* psutil

Documentation and tests further require:

* autograd
* trimesh
* sphinx
* sphinx-copybutton
* sphinxcontrib-programoutput

### Development installation

Unit-tests for TriMEM can be run with

```bash
pip install trimem/[tests]
pytest -v tests/
```

The documentation can be generated with `sphinx` by

```bash
pip install trimem/[docs]
cd doc; make html
```

## Usage

For an introduction to the usage of TriMEM please refer to the
[documentation](https://trimem.readthedocs.io/).

## Citation

If you use triMEM for your scientific work, please consider the citation of
Siggel, M. et al[^Siggel2022].

[^Siggel2022]: Siggel, M. et al. (2022) TriMem: A Parallelized Hybrid Monte
  Carlo Software for Efficient Simulations of Lipid Membranes.
  J. Chem. Phys. (in press) (2022); https://doi.org/10.1063/5.0101118

