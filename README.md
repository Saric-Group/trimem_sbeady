# Playing around with OpenMesh and the helfrich bending energy.

## Installation

The installation picks the compilers set in the variables $CC$ and $CXX$. It
falls back to a system's gcc installation if existent. On the mpcdf clusters
this is not recommended and it is better to set the compiler explicitly.

```bash
git clone --recurse-submodules https://gitlab.mpcdf.mpg.de/sebak/om-helfrich.git`
pip install ./om-helfrich
```

### Development installation

The above installation creates a temporary build environment that is populated
with all dependencies as specified in `pyproject.toml` and which is deleted
afterwards. However during development, it might be more convenient to directly
call `setup.py`:
```bash
python setup.py install -v --build-type DEBUG
```
This creates a local build (with cache) in `_skbuild` and copies the package
manually to the current python's `side-packages` destination. But installation 
of build-dependencies is not taken care of.

## Usage on the mpcdf clusters

No particular adaptions are necessary when using the intel compilers. However,
when using the gnu compilers (and thus libgomp) there are issues with mixing
with intel's omp implementation used by anaconda-\>numpy-\>mkl. This can be
resolved by setting MKL\_THREADING\_LAYER=GNU in the slurm script.

## mc\_app

This package can be used via the command app `mc_app`. This app is controlled
via some input file. A default input file can be generated by `mc_app config
--conf inp.conf`. A simulation that first runs a minimization and then restarts
sampling with hamiltonian monte carlo can, e.g., be performed with

```bash
# mesh generation
python -c "import meshzoo; \
           import meshio; \
           p,c = meshzoo.icosa_sphere(8); \
           meshio.write_points_cells('input.stl', p, [('triangle', c)])"

# input file
echo -n "\
[DEFAULT]
algorithm = minimize
info = 100
input = input.stl
output_prefix = out/test_
restart_prefix = out/restart_
output_format = vtu
[BONDS]
bond_type = Edge
r = 2
[SURFACEREPULSION]
n_search = cell-list
rlist = 0.2
exclusion_level = 2
refresh = 10
r = 2
lc1 = 0.15
[ENERGY]
kappa_b = 300.0
kappa_a = 1.0e6
kappa_v = 1.0e6
kappa_c = 0.0
kappa_t = 1.0e5
kappa_r = 1.0e3
area_fraction = 1.0
volume_fraction = 0.8
curvature_fraction = 1.0
continuation_delta = 0.0
continuation_lambda = 1.0
[HMC]
num_steps = 10000
traj_steps = 100
step_size = 2.5e-5
momentum_variance = 1.0
thin = 100
flip_ratio = 0.1
flip_type = parallel
initial_temperature = 1.0
cooling_factor = 1.0e-3
start_cooling = 10000
[MINIMIZATION]
maxiter = 2000
out_every = 0" > inp.conf

# prepare output folder
mkdir -p out

# run minimization
mc_app run --conf inp.conf

# sample
sed -i 's/= minimize/= hmc/g' inp.conf
mc_app run --conf inp.conf --restart 0
```
By setting `output_format = xyz`, a plain ascii xyz-coordinate trajectory
is written instead of a series of vtk unstructured-grid files.
