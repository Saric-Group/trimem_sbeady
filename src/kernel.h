/** \file kernel.h
 * \brief Compute-kernels for the helfrich energy.
 */
#ifndef KERNEL_H
#define KERNEL_H
#include <omp.h>
#include <memory>

#include "defs.h"

#include "params.h"
#include "mesh_properties.h"
#include "mesh_tether.h"
#include "mesh_repulsion.h"

namespace trimem {

//! energy contributions
real area_penalty(const EnergyParams& params,
                  const VertexProperties& props,
                  const VertexProperties& ref_props)
{
    real d = props.area / ref_props.area - 1.0;
    return params.kappa_a * d * d;
}

Point area_penalty_grad(const EnergyParams& params,
                        const VertexProperties& props,
                        const VertexProperties& ref_props,
                        const Point& d_area)
{
    real d   = props.area / ref_props.area - 1.0;
    real fac = 2.0 * params.kappa_a / ref_props.area * d;
    return fac * d_area;
}

real volume_penalty(const EnergyParams& params,
                    const VertexProperties& props,
                    const VertexProperties& ref_props)
{   
    real d = props.volume / ref_props.volume - 1.0;
    return params.kappa_v * d * d;
}

Point volume_penalty_grad(const EnergyParams& params,
                          const VertexProperties& props,
                          const VertexProperties& ref_props,
                          const Point& d_volume)
{   

    real d = props.volume / ref_props.volume - 1.0;
    real fac = 2.0 * params.kappa_v / ref_props.volume * d;
    return fac * d_volume;

}

real curvature_penalty(const EnergyParams& params,
                       const VertexProperties& props,
                       const VertexProperties& ref_props)
{   

    real d = props.curvature / ref_props.curvature - 1.0;
    return params.kappa_c * d * d;

}

Point curvature_penalty_grad(const EnergyParams& params,
                             const VertexProperties& props,
                             const VertexProperties& ref_props,
                             const Point& d_curvature)
{   

    real d = props.curvature / ref_props.curvature - 1.0;
    real fac = 2.0 * params.kappa_c / ref_props.curvature * d;
    return fac * d_curvature;

}

real tether_penalty(const EnergyParams& params, const VertexProperties& props)
{
    return params.kappa_t * props.tethering;
}

Point tether_penalty_grad(const EnergyParams& params,
                          const VertexProperties& props,
                          const Point& d_tether)
{
    return params.kappa_t * d_tether;
}

real repulsion_penalty(const EnergyParams& params,
                       const VertexProperties& props)
{
    return params.kappa_r * props.repulsion;
}

Point repulsion_penalty_grad(const EnergyParams& params,
                             const VertexProperties& props,
                             const Point& d_repulsion)
{
    return params.kappa_r * d_repulsion;
}

real helfrich_energy(const EnergyParams& params, const VertexProperties& props)
{
    return params.kappa_b * props.bending;
}

Point helfrich_energy_grad(const EnergyParams& params,
                           const VertexProperties& props,
                           const Point& d_bending)
{
    return params.kappa_b * d_bending;
}

real trimem_energy(const EnergyParams& params,
                   const VertexProperties& props,
                   const VertexProperties& ref_props)
{
    real energy = 0.0;
    energy += area_penalty(params, props, ref_props);
    // mmb
    if(ref_props.volume !=0){
        energy += volume_penalty(params, props, ref_props);
    }
    if(ref_props.curvature !=0){
    energy += curvature_penalty(params, props, ref_props);
    }
    energy += tether_penalty(params, props);
    energy += repulsion_penalty(params, props);
    energy += helfrich_energy(params, props);
    return energy;
}

Point trimem_gradient(const EnergyParams& params,
                      const VertexProperties& props,
                      const VertexProperties& ref_props,
                      const VertexPropertiesGradient& gprops)
{
    Point grad(0.0);
    grad += area_penalty_grad(params, props, ref_props, gprops.area);
    
    // MMB
    if (ref_props.volume !=0){
        grad += volume_penalty_grad(params, props, ref_props,  gprops.volume);
    }
    // MMB
    if(ref_props.curvature !=0){
        grad += curvature_penalty_grad(params, props, ref_props, gprops.curvature);
    }

    grad += tether_penalty_grad(params, props, gprops.tethering);
    grad += repulsion_penalty_grad(params, props, gprops.repulsion);
    grad += helfrich_energy_grad(params, props, gprops.bending);

    return grad;
}


/* BEGIN NSR VARIANT */


//! energy contributions
real area_penalty_nsr(const EnergyParams& params,
                  const VertexPropertiesNSR& props,
                  const VertexPropertiesNSR& ref_props)
{
    real d = props.area / ref_props.area - 1.0;
    return params.kappa_a * d * d;
}

//! MMB energy contributions
real area_penalty_nsr_mmb(const EnergyParams& params,
                  const VertexPropertiesNSR& props,
                  const VertexPropertiesNSR& ref_props)
{
    return params.kappa_a * props.area;
}

// MMB CHANGE AREA TO MINIMIZATION CONTRIBUTION
Point area_penalty_grad_nsr_mmb(const EnergyParams& params,
                        const VertexPropertiesNSR& props,
                        const VertexPropertiesNSR& ref_props,
                        const Point& d_area)
{
    return params.kappa_a * d_area;
}

Point area_penalty_grad_nsr(const EnergyParams& params,
                        const VertexPropertiesNSR& props,
                        const VertexPropertiesNSR& ref_props,
                        const Point& d_area)
{
    real d   = props.area / ref_props.area - 1.0;
    real fac = 2.0 * params.kappa_a / ref_props.area * d;
    return fac * d_area;
}

real volume_penalty_nsr(const EnergyParams& params,
                    const VertexPropertiesNSR& props,
                    const VertexPropertiesNSR& ref_props)
{
    real d = props.volume / ref_props.volume - 1.0;
    return params.kappa_v * d * d;
}

Point volume_penalty_grad_nsr(const EnergyParams& params,
                          const VertexPropertiesNSR& props,
                          const VertexPropertiesNSR& ref_props,
                          const Point& d_volume)
{
    real d = props.volume / ref_props.volume - 1.0;
    real fac = 2.0 * params.kappa_v / ref_props.volume * d;
    return fac * d_volume;
}

real curvature_penalty_nsr(const EnergyParams& params,
                       const VertexPropertiesNSR& props,
                       const VertexPropertiesNSR& ref_props)
{
    real d = props.curvature / ref_props.curvature - 1.0;
    return params.kappa_c * d * d;
}

Point curvature_penalty_grad_nsr(const EnergyParams& params,
                             const VertexPropertiesNSR& props,
                             const VertexPropertiesNSR& ref_props,
                             const Point& d_curvature)
{
    real d = props.curvature / ref_props.curvature - 1.0;
    real fac = 2.0 * params.kappa_c / ref_props.curvature * d;
    return fac * d_curvature;
}

real tether_penalty_nsr(const EnergyParams& params, const VertexPropertiesNSR& props)
{
    return params.kappa_t * props.tethering;
}

Point tether_penalty_grad_nsr(const EnergyParams& params,
                          const VertexPropertiesNSR& props,
                          const Point& d_tether)
{
    return params.kappa_t * d_tether;
}



real helfrich_energy_nsr(const EnergyParams& params, const VertexPropertiesNSR& props)
{
    return params.kappa_b * props.bending;
}

Point helfrich_energy_grad_nsr(const EnergyParams& params,
                           const VertexPropertiesNSR& props,
                           const Point& d_bending)
{
    return params.kappa_b * d_bending;
}


real trimem_energy_nsr(const EnergyParams& params,
                   const VertexPropertiesNSR& props,
                   const VertexPropertiesNSR& ref_props)
{
    real energy = 0.0;

    // MMB CHANGED
    if(params.area_frac<0){
        // energy contribution from area is sigma * area
        energy += area_penalty_nsr_mmb(params, props, ref_props);
    }
    else{
        // energy contribution from area is sigma * (area-area0)^2
        energy += area_penalty_nsr(params, props, ref_props);
    }

    //MMB CHANGED
    if(ref_props.volume!=0){
        energy += volume_penalty_nsr(params, props, ref_props);
    }
    if(ref_props.curvature !=0){
        energy += curvature_penalty_nsr(params, props, ref_props);
    }
    energy += tether_penalty_nsr(params, props);
    energy += helfrich_energy_nsr(params, props);
    return energy;
}

Point trimem_gradient_nsr(const EnergyParams& params,
                      const VertexPropertiesNSR& props,
                      const VertexPropertiesNSR& ref_props,
                      const VertexPropertiesGradientNSR& gprops)
{

    Point grad(0.0);

    //MMB CHANGE FOR TWO DIFFERENT VERSIONS OF THE AREA
    if (params.area_frac<0){
        // energy contribution from area is sigma * area
        grad += area_penalty_grad_nsr_mmb(params, props, ref_props, gprops.area);
    }
    else{
        // energy contribution from area is sigma * (area-area0)^2
        grad += area_penalty_grad_nsr(params, props, ref_props, gprops.area);
    }

    // MMB CHANGED
    if(ref_props.volume!=0){
    grad += volume_penalty_grad_nsr(params, props, ref_props,  gprops.volume);
    }
    if(ref_props.curvature !=0){
        grad += curvature_penalty_grad_nsr(params, props, ref_props, gprops.curvature);
    }

    grad += tether_penalty_grad_nsr(params, props, gprops.tethering);
    grad += helfrich_energy_grad_nsr(params, props, gprops.bending);


    return grad;
}

/* END NSR VARIANT */



//! evaluate VertexProperties
struct EvaluateProperties
{
    EvaluateProperties(const EnergyParams& params,
                       const TriMesh& mesh,
                       const BondPotential& bonds,
                       const SurfaceRepulsion& repulse,
                       std::vector<VertexProperties>& props) :
        params_(params),
        mesh_(mesh),
        bonds_(bonds),
        repulse_(repulse),
        props_(props) {}

    //parameters
    const EnergyParams& params_;
    const TriMesh& mesh_;
    const BondPotential& bonds_;
    const SurfaceRepulsion& repulse_;

    // result
    std::vector<VertexProperties>& props_;

    void operator() (const int i)
    {
        auto vh = mesh_.vertex_handle(i);
        props_[i] = vertex_properties(mesh_, bonds_, repulse_, vh);
    }
};

//! reduce vector of VertexProperties
struct ReduceProperties
{
    ReduceProperties(const std::vector<VertexProperties>& props) :
        props_(props) {}

    //parameters
    const std::vector<VertexProperties>& props_;

    void operator() (const int i, VertexProperties& contrib)
    {
        contrib += props_[i];
    }
};

struct EvaluatePropertiesGradient
{
    EvaluatePropertiesGradient(const TriMesh& mesh,
                               const BondPotential& bonds,
                               const SurfaceRepulsion& repulse,
                               const std::vector<VertexProperties>& props,
                               std::vector<VertexPropertiesGradient>& gradients)
        :
        mesh_(mesh),
        bonds_(bonds),
        repulse_(repulse),
        props_(props),
        gradients_(gradients) {}

    //parameters
    const TriMesh& mesh_;
    const BondPotential& bonds_;
    const SurfaceRepulsion& repulse_;
    const std::vector<VertexProperties>& props_;

    // result
    std::vector<VertexPropertiesGradient>& gradients_;

    void operator() (const int i)
    {
        auto vh = mesh_.vertex_handle(i);
        vertex_properties_grad(mesh_, bonds_, repulse_, vh, props_, gradients_);
    }
};

struct EvaluateGradient
{
    EvaluateGradient(const EnergyParams& params,
                     const VertexProperties& props,
                     const VertexProperties& ref_props,
                     const std::vector<VertexPropertiesGradient>& gprops,
                     std::vector<Point>& gradient) :
        params_(params),
        props_(props),
        ref_props_(ref_props),
        gprops_(gprops),
        gradient_(gradient) {}

    // parameters
    const EnergyParams& params_;
    const VertexProperties& props_;
    const VertexProperties& ref_props_;
    const std::vector<VertexPropertiesGradient>& gprops_;

    // result
    std::vector<Point>& gradient_;

    void operator() (const int i)
    {
        gradient_[i] += trimem_gradient(params_, props_, ref_props_, gprops_[i]);
    }

};



/* BEGIN NSR VARIANT */




struct EvaluatePropertiesNSR
{
    EvaluatePropertiesNSR(const EnergyParams& params,
                       const TriMesh& mesh,
                       const BondPotential& bonds,
                       std::vector<VertexPropertiesNSR>& props) :
        params_(params),
        mesh_(mesh),
        bonds_(bonds),
        props_(props) {}

    //parameters
    const EnergyParams& params_;
    const TriMesh& mesh_;
    const BondPotential& bonds_;

    // result
    std::vector<VertexPropertiesNSR>& props_;

    void operator() (const int i)
    {
        auto vh = mesh_.vertex_handle(i);
        props_[i] = vertex_properties_nsr(mesh_, bonds_, vh);
    }
};

//! reduce vector of VertexProperties
struct ReducePropertiesNSR
{
    ReducePropertiesNSR(const std::vector<VertexPropertiesNSR>& props) :
        props_(props) {}

    //parameters
    const std::vector<VertexPropertiesNSR>& props_;

    void operator() (const int i, VertexPropertiesNSR& contrib)
    {
        contrib += props_[i];
    }
};

struct EvaluatePropertiesGradientNSR
{
    EvaluatePropertiesGradientNSR(const TriMesh& mesh,
                               const BondPotential& bonds,
                               const std::vector<VertexPropertiesNSR>& props,
                               std::vector<VertexPropertiesGradientNSR>& gradients)
        :
        mesh_(mesh),
        bonds_(bonds),
        props_(props),
        gradients_(gradients) {}

    //parameters
    const TriMesh& mesh_;
    const BondPotential& bonds_;
    const std::vector<VertexPropertiesNSR>& props_;

    // result
    std::vector<VertexPropertiesGradientNSR>& gradients_;

    void operator() (const int i)
    {
        auto vh = mesh_.vertex_handle(i);
        vertex_properties_grad_nsr(mesh_, bonds_, vh, props_, gradients_);
    }
};

struct EvaluateGradientNSR
{
    EvaluateGradientNSR(const EnergyParams& params,
                     const VertexPropertiesNSR& props,
                     const VertexPropertiesNSR& ref_props,
                     const std::vector<VertexPropertiesGradientNSR>& gprops,
                     std::vector<Point>& gradient) :
        params_(params),
        props_(props),
        ref_props_(ref_props),
        gprops_(gprops),
        gradient_(gradient) {}

    // parameters
    const EnergyParams& params_;
    const VertexPropertiesNSR& props_;
    const VertexPropertiesNSR& ref_props_;
    const std::vector<VertexPropertiesGradientNSR>& gprops_;

    // result
    std::vector<Point>& gradient_;

    void operator() (const int i)
    {
        gradient_[i] += trimem_gradient_nsr(params_, props_, ref_props_, gprops_[i]);
    }

};








/* BEGIN NSR VARIANT */








template<class Kernel, class ReductionType>
void parallel_reduction(int n, Kernel& kernel, ReductionType& reduce)
{
#pragma omp declare reduction (tred : ReductionType : omp_out += omp_in) \
  initializer(omp_priv={})

#pragma omp parallel for reduction(tred:reduce)
    for (int i=0; i<n; i++)
    {
        kernel(i, reduce);
    }
}

template<class Kernel>
void parallel_for(int n, Kernel& kernel)
{
#pragma omp parallel for
    for (int i=0; i<n; i++)
      kernel(i);
}

}
#endif
