# Triangulated Mesh Generator

The C code in this directory takes the coordinates corresponding to particles on the surface of a sphere. These coordinates are computed using the in-house code that can place particles on the surface of various objects - here referred to as **shells.c**. Next, it uses functions coming from the in-house MC code for triangulated fluid membranes - here referred to as **MCtriang.c** - to extracts the triangulation that the coordinates give rise to (most notably, the faces of the triangles with the correct orientation). In a nutshell, if you want to use this code, you should:
1. Use **shells.c** $\rightarrow$ Generate $3N$ coordinates on the surface of a sphere; $N$ is the number of vertices your membrane will have.
2. Pass the coordinates to the code in this directory $\rightarrow$ It will give you the faces of the triangulation. Vertex positions + edges + faces make the triangulation.

### What is the point of this code?
You can alternatively use the Python ```trimesh``` library to generate triangulated meshes using the ```trimesh.creation.icosphere()``` function. You can read more about how to do so [here](../HowTo_TriLMP.md). This Python function only allows you to use a specific mesh sizes (discrete $N$ values). The C code in this directory gives you more freedom on choosing the value of $N$.
