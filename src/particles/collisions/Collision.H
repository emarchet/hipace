#ifndef HIPACE_COLLISION_H_
#define HIPACE_COLLISION_H_

#include "particles/plasma/MultiPlasma.H"
#include "particles/beam/MultiBeam.H"

#include <AMReX.H>
#include <AMReX_GpuBuffer.H>

#include <string>
#include <vector>

/**
 * \brief Impact ionization of a plasma target species (ion or neutral atom), creating new
 * electrons in a plasma electron species. Pairing follows Perez et al. (2012) with the weight
 * rescaling of Higginson et al. (2020); the cross section is the RBEB model of Kim et al. (2000).
 *
 * Two types are available:
 * - electron_impact: the projectile is a plasma electron; the final state is computed
 *   in the center-of-momentum frame with equal energy sharing between the two electrons.
 * - beam_impact: the projectile is a beam species of any mass and charge (e.g. protons).
 *   The beam is read-only. The target is ionized with Perez's two-step probability (event with
 *   the larger weight, then rejection with the beam weight) over the time the target spends in
 *   the beam slice. The ejected electron is created in the ion rest frame
 *   along the projectile direction, the ion takes the recoil.
 */
class Collision
{
    std::string m_projectile_name = "";
    std::string m_target_name = "";
    std::string m_new_electron_name = "";
    std::string m_collision_type = "";
    std::string m_physical_element ="";
    int m_atomic_number = 0;
    amrex::Gpu::Buffer<amrex::Real> m_binding_energies;
    amrex::Gpu::Buffer<amrex::Real> m_ionization_energies;

public:

    /** Read parameters from the input file and check them against the plasma and beam species
     *
     * \param[in] multi_plasma all plasma species (already initialized)
     * \param[in] multi_beam all beam species
     * \param[in] collision_name name of this impact ionization process
     */
    void ReadParameters (
        const MultiPlasma& multi_plasma,
        const MultiBeam& multi_beam,
        std::string const collision_name);

    /**
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     * \param[in] multi_beam all beams (read-only, used by beam_impact)
     **/
    void doCollision (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        MultiBeam& multi_beam);
    /**
     * \brief Pair projectile and target particles cell by cell (random pairing as in
     * Perez et al. 2012) and apply collision_function to each pair; create the new
     * electrons with ionization_function
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     * \param[in] collision_function check if a collision between two particles produces a new electron,
     *  redistribute the energies and the momenta and update the old particle properties accordingly
     * \param[in] ionization_function initialize newly produced electrons
     **/
    template <class F, class G>
    void pairAndIonize (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        F const& collision_function,
        G const& ionization_function);

    /**
     * \brief Same as pairAndIonize, with a beam species (current beam slice) as projectile.
     * The beam particles are only read.
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     * \param[in] multi_beam all beams
     * \param[in] collision_function decide whether a beam-target pair ionizes the target and
     *  update the target
     * \param[in] ionization_function initialize newly produced electrons
     **/
    template <class F, class G>
    void pairAndIonizeBeam (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        MultiBeam& multi_beam,
        F const& collision_function,
        G const& ionization_function);

    /**
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     **/
    void doElectronImpact (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma);

    /**
     * \param[in] lev MR level
     * \param[in] geom corresponding geometry object
     * \param[in,out] multi_plasma all plasmas
     * \param[in] multi_beam all beams (read-only)
     **/
    void doBeamImpact (
        int lev, const amrex::Geometry& geom,
        MultiPlasma& multi_plasma,
        MultiBeam& multi_beam);
};

#endif // HIPACE_COLLISION_H_