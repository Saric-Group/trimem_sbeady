# TriLMP = TriMEM + LAMMPS

TriLMP couples [**TriMEM**](https://github.com/bio-phys/trimem) to [LAMMPS](https://github.com/lammps/lammps) via the python interface of the latter, to perform Hybrid Monte Carlo (HMC = MD + MC) simulations of lipid
membranes according to the Helfrich theory[^Helfrich1973]. Please refer to Siggel et al.[^Siggel2022] for details on the Hamiltonian simulated, or the [TriMEM documentation](https://trimem.readthedocs.io/en/latest/) further information on TriMEM (e.g., dependencies).

## Basic introduction
In TriLMP, the Dynamically Triangulated network representing the membrane is a LAMMPS molecule made of particles (network vertices) that are connected through bonds (network edges). Throughout the simulation, TriLMP uses TriMEM to compute the Helfrich Hamiltonian, as well as to obtain the list of edges that have to be updated to ensure membrane fluidity. At each MD simulation step, TriMEM provides the gradient of the Helfrich Hamiltonian. These gradients are added to the forces acting on the vertices in the network. At a prescribed frequency (see TriLMP docs), TriLMP informs LAMMPS to update the network connectivity by removing and creating new bonds between vertices in the network. Additionally, the calculation of the surface repulsion, which prevents the network from self-intersecting, is dealt with by LAMMPS instead of TriMEM in TriLMP. For further details on how to use the package, please refer to the ```TriLMP_GuidesAndHelp/HowTo_TriLMP.md``` file.

To run simulations with TriLMP, you will be creating an object of the class ```TriLmp```. While you can always edit the class in the trilmp.py file in `src/` for details, to run a basic simulation you will just have to
1. Pass to your ```TriLmp``` object the class attributes you want (e.g., membrane mechanical properties, size of the network, ...)
2. Call the ```TriLmp.run(params)``` method to actually run the simulation

[^Helfrich1973]: Helfrich, W. (1973) Elastic properties of lipid bilayers:
  Theory and possible experiments. Zeitschrift für Naturforschung C,
  28(11), 693-703

[^Siggel2022]: Siggel, M. et al. (2022) TriMem: A Parallelized Hybrid Monte
  Carlo Software for Efficient Simulations of Lipid Membranes.
  J. Chem. Phys. (in press) (2022); https://doi.org/10.1063/5.0101118

## Installation guide

TriLMP has been successfully installed and tested in Linux and MAC OS.

We suggest installing TriLmp in a conda environment, and then installing trimem/trilmp as an editable install, which allows you to alter the ```TriLmp``` functionality by directly tinkering with the src/trimem/trilmp.py file if needed.
This can come in handy in case you need to define or initialize a simulation set-up that is not covered by the current version of TriLMP. The code is designed such that additional LAMMPS commands can always be passed as command strings once the ```TriLmp``` object has been created.

### 1. Downloading the repository

Clone the repository by running:

```
git clone --recurse-submodules [repository]
git submodule update
```

(Note that at the moment, LAMMPS is not added as a submodule so that you have the freedom to choose what LAMMPS do you want. See more details below).

### 2. Setting up the conda environment

To set up a conda environment including some prerequisites use:

```
conda env create -f environment.yml
```

You can find more info on the prerequisites in the [TriMEM documentation](https://trimem.readthedocs.io/en/latest/).

Within the ```environment.yml``` file you can choose the name you want for your environment. Here we will call it ```env_TriLMP```. Make sure that the environment is activated throughout the rest of the setup! For that, type

```
conda activate env_TriLMP
```

### 3. Installing TriMEM/TriLMP
TriLMP can be installed using pip as follows (the editable install we were mentioning above): 

```bash
 pip install -e .
```

Build and compile the shared libraries using:
```bash
python setup.py build
```

(MAC users) If you have a problem with this step due to ```OpenMesh```, you might have to change the line
```
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
```
for 
```
cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
```

within the file ```OpenMesh/CMakeLists.txt```.


Note that this folder will be the actual location of the modules with an editable install, i.e., if you change something in the python code here, effects will be immediate. In case you want to change something on the c++ side of the code, make sure to run 
```bash
python3 setup.py build
```
in order to to compile and copy 
the libaries to the `src/trimem` folder as described below.

Finally some shared libraries have to be copied manually from the _skbuild folder. To do so, run

```
. copy_libs.sh
```

In Linux, you may have to manually the copy the libraries by using:

```
cp _skbuild/linux-x86_64-3.11/cmake-build/core.cpython-311-x86_64-linux-gnu.so src/trimem/.
cp _skbuild/linux-x86_64-3.11/cmake-build/libtrimem.so src/trimem/.
cp _skbuild/linux-x86_64-3.11/cmake-build/Build/lib/* src/trimem/.
```

Depending on what changes you do to the C part of the code, you might need to copy these libraries again.

### 4. Installing LAMMPS

TriLMP requires LAMMPS, which has to be installed using python shared libraries and some specific packages. You will have to perform the python install using your ```env_TriLMP``` environment as we detail below. Šarić group members are working on a patched LAMMPS version that is available within the GitHub of the group. You can choose to clone this repository, or instead use the latest stable release from LAMMPS directly.

For example, here we clone the lastest LAMMPS stable release. To do so, go back to the directory where you want to have your install folder and run
```bash
git clone -b stable https://github.com/lammps/lammps.git lammps
```

We can now start manually installing LAMMPS. Go first to the ```lammps``` directory and create the ```build``` folder.
```bash
cd lammps                
mkdir build
cd build
```

Next we have to determine the python executable path. You can do that from the python console
using
```bash
python           
import sys
print(sys.executable)
```
which should print something like 

```bash
/nfs/scistore15/saricgrp/yourusername/.conda/envs/Trienv/bin/python
```
and will be referred to as ```python_path``` in the next bash command. Note that this enables PyLammps in this specific conda environment only! 

Now we can set up the makefiles and build LAMMPS. The command line you see below is a suggestion for the LAMMPS packages you may want to compile for tests (these are suggested here because they have proved useful in the past in the development of works involving TriLMP).

```
cmake -D BUILD_OMP=yes -D BUILD_SHARED_LIBS=yes -D PYTHON_EXECUTABLE="/nfs/scistore15/saricgrp/yourusername/.conda/envs/env_TriLMP/bin/python" -D PKG_PYTHON=yes -D PKG_OPENMP=yes -D PKG_MOLECULE=yes  -D PKG_EXTRA-PAIR=yes -D PKG_EXTRA-FIX=yes -D PKG_EXTRA-COMPUTE=yes -D PKG_RIGID=yes -D PKG_ASPHERE=yes -D PKG_BROWNIAN=yes -D PKG_MC=yes -D PKG_REACTION=yes ../cmake 
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

**IMPORTANT NOTE**

For the LAMMPS compilation to work, you will need OpenMPI and OpenMP libraries. OpenMPI takes care of the distributed memory parallelization, while OpenMP deals with shared memory parallelization. If you use the ```environment.yml``` file to create your conda environment, it includes these libraries through ```mpich``` and ```llvm-openmp```. 

Errors can arise when LAMMPS tries to find the OpenMPI library. Therefore, a couple of hints that may help are:

- *In the cluster/LINUX*: If you try the above compilation and it doesn't work, try loading the openmpi module first by typing ```module load openmpi```
- *In the cluster/LINUX/MAC OS*: If you try the above compilation and it fails, and you are certain that you have an OpenMPI library (you can test this with ```which openmpi``` or ```which mpicc```), you may need to add the ```-D MPI_C_COMPILER=$(which mpicc) -D MPI_CXX_COMPILER=$(which mpicxx)``` flags to the ```cmake``` build. Sometimes, you may need to specify the path to the libraries explicitly through flags like: ```-D MPI_C_COMPILER=path_in_your_system  -D MPI_CXX_COMPILER=path_in_your_system   -D MPI_C_LIBRARIES=path_in_your_system   -D MPI_CXX_LIBRARIES=path_in_your_system```.
  
Finally to make LAMMPS accessible for python, i.e. making and copying the shared libaries, use

```bash
make install-python
```

## 5. Testing your installation

Try running the examples in ```TriLMP_Examples/examples_2026``` to check whether you have installed TriLMP correctly. To run them, use

```
cd TriLMP_Examples/examples_2026/example_yourchoice
python launch_nameexample.py
```

## 6. Updating the code and adding new functionalities

If you want to edit the ```TriLMP``` code to add new simulation functionalities, edit the ```src/trimem/mc/trilmp.py``` file.
