/** \file flips.cpp
 * \brief Performing edge flips on openmesh.
 */
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <random>

#include "defs.h"

#include "flips.h"
#include "flip_utils.h"
#include "energy.h"
#include "mesh_properties.h"
#include "omp_guard.h"
#include "mesh_util.h"

namespace trimem {

typedef std::chrono::high_resolution_clock myclock;
static std::mt19937 generator_(myclock::now().time_since_epoch().count());

int flip_serial(TriMesh& mesh, EnergyManager& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        std::runtime_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial vertex properties
    VertexProperties props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // acceptance probability distribution
    std::uniform_real_distribution<real> accept(0.0, 1.0);

    // proposal distribution
    std::uniform_int_distribution<int> propose(0, nedges-1);

    int acc = 0;
    for (int i=0; i<nflips; i++)
    {
        int  idx = propose(generator_);
        auto eh  = mesh.edge_handle(idx);
        if (mesh.is_flip_ok(eh) and !mesh.is_boundary(eh))
        {
            // remove old properties
            auto oprops = edge_vertex_properties(mesh, eh, *(estore.bonds),
                                                 *(estore.repulse));
            props -= oprops;

            // update with new properties
            mesh.flip(eh);
            auto nprops = edge_vertex_properties(mesh, eh, *(estore.bonds),
                                                 *(estore.repulse));
            props += nprops;

            // evaluate energy
            real en = estore.energy(props);
            real de = en - e0;

            // evaluate acceptance probability
            real alpha = de < 0.0 ? 1.0 : std::exp(-de);
            real u     = accept(generator_);
            if (u <= alpha)
            {
                e0 = en;
                acc += 1;
            }
            else
            {
                mesh.flip(eh);
                props -= nprops;
                props += oprops;
            }
        }
    }

    return acc;
}

int flip_parallel_batches(TriMesh& mesh, EnergyManager& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        throw std::range_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial energy and associated vertex properties
    VertexProperties props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // set-up locks on edges
    std::vector<omp_lock_t> l_edges(nedges);
    for (auto& lock: l_edges)
        omp_init_lock(&lock);

    int acc  = 0;
#pragma omp parallel reduction(+:acc)
    {
        int ithread = omp_get_thread_num();
        int nthread = omp_get_num_threads();

        int itime   = myclock::now().time_since_epoch().count();
        std::mt19937 prng((ithread + 1) * itime);
        std::uniform_real_distribution<real> accept(0.0, 1.0);
        std::uniform_int_distribution<int> propose(0, nedges-1);
        int iflips = (int) std::ceil(nflips / nthread);

        for (int i=0; i<iflips; i++)
        {
#pragma omp barrier
            EdgeHandle eh(-1);

            {
                std::vector<OmpGuard> guards;
                int idx = propose(prng);
                eh = mesh.edge_handle(idx);
                guards = test_patch(mesh, idx, l_edges);
#pragma omp barrier
                if (guards.empty())
                {
                    continue;
                }
            }
            // here all locks will have been released

            if (not mesh.is_flip_ok(eh))
                continue;

            // compute differential properties
            auto dprops = edge_vertex_properties(mesh, eh, *(estore.bonds),
                                                 *(estore.repulse));
            mesh.flip(eh);
            dprops -= edge_vertex_properties(mesh, eh, *(estore.bonds),
                                             *(estore.repulse));

            real u = accept(prng);

            // evaluate energy
#pragma omp critical
            {
                props -= dprops;
                real en  = estore.energy(props);
                real de = en - e0;

                // evaluate acceptance probability
                real alpha = de < 0.0 ? 1.0 : std::exp(-de);
                if (u <= alpha)
                {
                    e0 = en;
                    acc += 1;
                }
                else
                {
                    mesh.flip(eh);
                    props += dprops;
                }
            }
        }
    } // parallel

    return acc;
}






/* BEGIN NSR VARIANT */


int flip_serial_nsr(TriMesh& mesh, EnergyManagerNSR& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        std::runtime_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial vertex properties
    VertexPropertiesNSR props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // acceptance probability distribution
    std::uniform_real_distribution<real> accept(0.0, 1.0);

    // proposal distribution
    std::uniform_int_distribution<int> propose(0, nedges-1);

    int acc = 0;
    for (int i=0; i<nflips; i++)
    {
        int  idx = propose(generator_);
        auto eh  = mesh.edge_handle(idx);
        if (mesh.is_flip_ok(eh) and !mesh.is_boundary(eh))
        {
            // remove old properties
            auto oprops = edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                                 );
            props -= oprops;

            // update with new properties
            mesh.flip(eh);
            auto nprops = edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                                 );
            props += nprops;

            // evaluate energy
            real en = estore.energy(props);
            real de = en - e0;

            // evaluate acceptance probability
            real alpha = de < 0.0 ? 1.0 : std::exp(-de);
            real u     = accept(generator_);
            if (u <= alpha)
            {
                e0 = en;
                acc += 1;
            }
            else
            {
                mesh.flip(eh);
                props -= nprops;
                props += oprops;
            }
        }
    }

    return acc;
}

std::vector<std::array<int,4>> flip_parallel_batches_nsr(TriMesh& mesh, EnergyManagerNSR& estore, const real& flip_ratio, const std::vector<int>& ids_notflip)
{
    if (flip_ratio > 1.0)
        throw std::range_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // HARDCODING + CHECKING APPROACH WORKS -- ONLY PROBLEM IS COMPILATION
    // COULD THIS VECTOR BE A PROPERTY OF THE MESH?
    // MMB HARDCODING -- THOSE IDs that should not be flipped
    // notice that this is for a r = 4 membrane with 3 patches
    //std::vector<int> ids_notflip = {9, 104, 109, 356, 370, 384, 386, 406, 408, 409, 428, 430, 1372, 1424, 1476, 1477, 1478, 1480, 1482, 1483, 1484, 1542, 1545, 1568, 1569, 1570, 1571, 1573, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1644, 1647, 1664, 1665, 1666, 1667, 1668, 1674, 1675, 1676, 1677, 1678, 1679, 1700, 123, 141, 144, 471, 472, 546, 547, 549, 556, 557, 558, 559, 1848, 1849, 1850, 1852, 1854, 1856, 1872, 2154, 2155, 2156, 2157, 2159, 2161, 2163, 2164, 2165, 2166, 2167, 2183, 2190, 2191, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2492, 2545, 70, 73, 75, 255, 264, 265, 266, 273, 274, 276, 324, 332, 975, 976, 978, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1037, 1040, 1041, 1042, 1044, 1045, 1046, 1047, 1048, 1054, 1055, 1056, 1057, 1058, 1059, 1061, 1064, 1228, 1232, 1234, 1256, 1261, 1263, 1264};

    // get initial energy and associated vertex properties
    VertexPropertiesNSR props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // set-up locks on edges
    std::vector<omp_lock_t> l_edges(nedges);
    for (auto& lock: l_edges)
        omp_init_lock(&lock);

    int acc  = 0;
    int nattempt = 0;
    std::array<int,4> flip_loc;
    std::vector<std::array<int,4>> flip_ids;


#pragma omp parallel reduction(+:acc)
    {
        int ithread = omp_get_thread_num();
        int nthread = omp_get_num_threads();

        int itime   = myclock::now().time_since_epoch().count();
        std::mt19937 prng((ithread + 1) * itime);
        std::uniform_real_distribution<real> accept(0.0, 1.0);
        std::uniform_int_distribution<int> propose(0, nedges-1);
        int iflips = (int) std::ceil(nflips / nthread);

        for (int i=0; i<iflips; i++)
        {
#pragma omp barrier
            EdgeHandle eh(-1);



            {
                std::vector<OmpGuard> guards;
                int idx = propose(prng);
                eh = mesh.edge_handle(idx);
                guards = test_patch(mesh, idx, l_edges);
#pragma omp barrier
                if (guards.empty())
                {
                    continue;
                }
            }
            // here all locks will have been released

            if (not mesh.is_flip_ok(eh))
                continue;

            // compute differential properties
            auto dprops = edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                                 );
            mesh.flip(eh);
            dprops -= edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                             );

            real u = accept(prng);

            // evaluate energy
#pragma omp critical
            {
                nattempt += 1;
                props -= dprops;
                real en  = estore.energy(props);
                real de = en - e0;

                // evaluate acceptance probability
                real alpha = de < 0.0 ? 1.0 : std::exp(-de);
                if (u <= alpha)
                {   
                    auto f1 = (mesh.from_vertex_handle(mesh.halfedge_handle(eh,1)).idx());
                    auto f2 = (mesh.to_vertex_handle(mesh.halfedge_handle(eh,1)).idx());
                    mesh.flip(eh);
                    auto f3 = (mesh.from_vertex_handle(mesh.halfedge_handle(eh,1)).idx());
                    auto f4 = (mesh.to_vertex_handle(mesh.halfedge_handle(eh,1)).idx());
                    mesh.flip(eh);

                    // if found -- do not flip
                    if( (std::find(ids_notflip.begin(), ids_notflip.end(), f1) != ids_notflip.end()) or (std::find(ids_notflip.begin(), ids_notflip.end(), f2) != ids_notflip.end()) or (std::find(ids_notflip.begin(), ids_notflip.end(), f3) != ids_notflip.end()) or (std::find(ids_notflip.begin(), ids_notflip.end(), f4) != ids_notflip.end()) )
                    {
                        mesh.flip(eh);
                        props += dprops;
                    }

                    // else, can flip
                    else{
                    e0 = en;
                    acc += 1;

                    flip_loc[0]=f1;
                    flip_loc[1]=f2;
                    //mesh.flip(eh);
                    flip_loc[2]=f3;
                    flip_loc[3]=f4;
                    //mesh.flip(eh);
                    flip_ids.push_back(flip_loc);
                    }

                }
                else
                {
                    mesh.flip(eh);
                    props += dprops;
                }
            }
        }
    } // parallel
    flip_loc[0]=acc;
    flip_loc[1]=nflips;
    flip_loc[2]=nattempt;
    flip_loc[3]=0;

    flip_ids.push_back(flip_loc);

    return flip_ids;
}


/* END NSR VARIANT */

}
