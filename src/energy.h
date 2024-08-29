/** \file energy.h
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#ifndef ENERGY_H
#define ENERGY_H
#include <memory>

#include "defs.h"
#include "params.h"
#include "mesh_properties.h"

namespace trimem {

struct BondPotential;
struct SurfaceRepulsion;
struct NeighbourList;

class EnergyManager
{
public:

    // constructors

    EnergyManager(const TriMesh& mesh,
                  const EnergyParams& params);
    EnergyManager(const TriMesh& mesh,
                const EnergyParams& params,
                const VertexProperties& vertex_properties);


  // update reference properties
    void update_reference_properties();
    VertexProperties interpolate_reference_properties() const;

//EnergyManager createEnergyManager() const;

    // update repulsion potential
    void update_repulsion(const TriMesh& mesh);

    // energy and gradient evaluation
    VertexProperties properties(const TriMesh& mesh);
    real energy(const TriMesh& mesh);
    real energy(const VertexProperties& props);
    std::vector<Point> gradient(const TriMesh& mesh);

    // print status information
    void print_info(const TriMesh& mesh);


    // energy parameters
    EnergyParams params;

    // management of reference properties
    VertexProperties initial_props;

    // bond potential
    std::unique_ptr<BondPotential> bonds;

    // repulsion penalty
    std::unique_ptr<SurfaceRepulsion> repulse;

    // neighbour list
    std::unique_ptr<NeighbourList> nlist;
};


class EnergyManagerNSR
{
public:

    // constructors

    EnergyManagerNSR(const TriMesh& mesh,
                  const EnergyParams& params);
    EnergyManagerNSR(const TriMesh& mesh,
                const EnergyParams& params,
                const VertexPropertiesNSR& vertex_properties);


  // update reference properties
    void update_reference_properties();
    VertexPropertiesNSR interpolate_reference_properties() const;

//EnergyManager createEnergyManager() const;


    // energy and gradient evaluation
    VertexPropertiesNSR properties(const TriMesh& mesh);
    real energy(const TriMesh& mesh);
    real energy(const VertexPropertiesNSR& props);
    std::vector<Point> gradient(const TriMesh& mesh);
    void gradient_direct(const TriMesh& mesh,std::vector<Point>);

    // print status information
    void print_info(const TriMesh& mesh);

    // energy parameters
    EnergyParams params;

    // management of reference properties
    VertexPropertiesNSR initial_props;

    // bond potential
    std::unique_ptr<BondPotential> bonds;

};



}
#endif
