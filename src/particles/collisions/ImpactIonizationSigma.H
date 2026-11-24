#ifndef HIPACE_IMPACT_IONIZATION_SIGMA_H_
#define HIPACE_IMPACT_IONIZATION_SIGMA_H_

#include <AMReX_AmrCore.H>
#include <AMReX_REAL.H>
#include "utils/Constants.H"

// Index mapping for reduced arrays
// Returns the starting index in the energy arrays for a given Z
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int get_energy_index(int Z)
{
    if (Z == 1)  return 0;   // H  -  1 entry  (indices 0)
    if (Z == 7)  return 1;   // N  -  7 entries (indices 1-7)
    if (Z == 2)  return 8;   // He -  2 entries (indices 8-9)
    if (Z == 18) return 10;  // Ar - 18 entries (indices 10-27)
    if (Z == 37) return 28;  // Rb - 37 entries (indices 28-64)
    if (Z == 54) return 65;  // Xe - 54 entries (indices 65-118)

    return 0;
}

namespace coll_ion
{
    using namespace amrex;

    inline constexpr Real alpha = 7.2973525643e-3;
    inline constexpr Real a0    = 5.29177210544e-11; // m
    inline constexpr Real normalization = 4.0 * MathConst::pi * a0*a0 * alpha*alpha*alpha*alpha;

    // Check if the collision energy is sufficient for ionization
    AMREX_GPU_HOST_DEVICE inline
    bool impact_energy (
        Real const ux1, Real const uy1, Real const uz1, Real const g1,
        Real const ux2, Real const uy2, Real const uz2, Real const g2,
        Real const m1, Real const m2, Real const Eion_eV, Real const c2, Real const inv_c2)
    {
        using namespace amrex::literals;

        // Tolerances for numerical checks
        constexpr Real rel_tol  = 1e-10_rt;
        constexpr Real beta_tol = 1e-12_rt;

        Real const Eion  = Eion_eV * PhysConstSI::q_e;  // [J]

        // ------------------------------------------------------------------
        // Early-out if relative motion is negligible
        // ------------------------------------------------------------------
        Real const diffx = ux1 - ux2;
        Real const diffy = uy1 - uy2;
        Real const diffz = uz1 - uz2;
        Real const diffm = std::sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
        Real const summ = std::sqrt(ux1*ux1 + uy1*uy1 + uz1*uz1) + std::sqrt(ux2*ux2 + uy2*uy2 + uz2*uz2);
        // If g = u1 - u2 = 0, do not collide.
        // Or if the relative difference is less than 1.0e-10.
        if (summ <= 0.0_rt) return false;
        if (diffm / summ < rel_tol) return false;

        // ------------------------------------------------------------------
        // Initial lab-frame momenta
        // ------------------------------------------------------------------
        Real const px1 = ux1 * m1 * PhysConstSI::c;
        Real const py1 = uy1 * m1 * PhysConstSI::c;
        Real const pz1 = uz1 * m1 * PhysConstSI::c;

        Real const px2 = ux2 * m2 * PhysConstSI::c;
        Real const py2 = uy2 * m2 * PhysConstSI::c;
        Real const pz2 = uz2 * m2 * PhysConstSI::c;

        // ------------------------------------------------------------------
        // COM velocity
        // ------------------------------------------------------------------
        Real const mass_g = m1 * g1 + m2 * g2;
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mass_g > 0.0_rt, "Mass is zero in ImpactIonizationSigma");

        Real const vcx    = (px1+px2) / mass_g;
        Real const vcy    = (py1+py2) / mass_g;
        Real const vcz    = (pz1+pz2) / mass_g;
        Real const vcms   = vcx*vcx + vcy*vcy + vcz*vcz;

        Real beta2 = vcms * inv_c2;
        if (beta2 < 0.0_rt) {
            if (beta2 > -beta_tol) {
                beta2 = 0.0_rt;
            } else {
                return false;
            }
        }

        if (beta2 >= 1.0_rt - beta_tol) {
            return false;
        }

        Real const gc = 1.0_rt / std::sqrt( 1.0_rt - beta2 );

        // Stable equivalent of (gc - 1)/vcms:
        // (gamma - 1)/v^2 = gamma^2 / ((gamma + 1) c^2)
        Real const boost_v2 = gc*gc / ((gc + 1.0_rt) * c2);

        // ------------------------------------------------------------------
        // COM-frame boost of the initial momenta
        // ------------------------------------------------------------------
        Real const vcDv1 = (vcx*ux1 + vcy*uy1 + vcz*uz1) * PhysConstSI::c / g1;
        Real const vcDv2 = (vcx*ux2 + vcy*uy2 + vcz*uz2) * PhysConstSI::c / g2;

        // Boost particle 1 to COM
        Real p1sx;
        Real p1sy;
        Real p1sz;
        if ( vcms > 0.0_rt )
        {
            Real const lorentz_coeff_1 =
                ( boost_v2 * vcDv1 - gc ) * m1 * g1;

            p1sx = px1 + vcx*lorentz_coeff_1;
            p1sy = py1 + vcy*lorentz_coeff_1;
            p1sz = pz1 + vcz*lorentz_coeff_1;
        }
        else // If vcms = 0, don't do Lorentz-transform.
        {
            p1sx = px1;
            p1sy = py1;
            p1sz = pz1;
        }

        // Boost particle 2 to COM
        Real p2sx;
        Real p2sy;
        Real p2sz;
        if (vcms > 0.0_rt) {
            Real const lorentz_coeff_2 =
                ( boost_v2 * vcDv2 - gc ) * m2 * g2;

            p2sx = px2 + vcx*lorentz_coeff_2;
            p2sy = py2 + vcy*lorentz_coeff_2;
            p2sz = pz2 + vcz*lorentz_coeff_2;
        } else {
            p2sx = px2;
            p2sy = py2;
            p2sz = pz2;
        }

        // ------------------------------------------------------------------
        // Ionization probability and final-state construction
        // ------------------------------------------------------------------
        Real const p1s_sq = p1sx*p1sx + p1sy*p1sy + p1sz*p1sz;
        Real const p2s_sq = p2sx*p2sx + p2sy*p2sy + p2sz*p2sz;

        Real const E1s = std::sqrt(p1s_sq*c2 + m1*m1*c2*c2);
        Real const E2s = std::sqrt(p2s_sq*c2 + m2*m2*c2*c2);

        Real const K1s = E1s - m1*c2;
        Real const K2s = E2s - m2*c2;

        // Available initial kinetic energy in the COM frame [J]
        Real const Ecm = K1s + K2s;

        // Kinetic energy available to the 3 outgoing particles in COM [J]
        Real const Eout = Ecm - Eion;

        return (Eout > 0.0_rt);  // false if energy is not sufficient for ionization
    }

    struct IonizationResult
    {
        Real sigma;
        Real ionization_en;
    };

    // Cross section using RBEB model from Kim et al. (2000)
    /**
     * @brief Relativistic Binary-Encounter-Bethe (RBEB) electron-impact
     *        ionization cross section.
     *
     * Loops over all bound electrons of the ion (from innermost to the
     * outermost occupied shell).  Degenerate sub-shells sharing the same
     * binding energy are grouped and counted together (factor `n`).
     *
     * \param[in] atomic_number             Atomic number of the target element
     * \param[in] zstar                     Current ionization state (number of removed electrons)
     * \param[in] e                         Incident-electron kinetic energy (eV)
     * \param[in] p_binding_energies        Pointer to table of binding energies
     * \param[in] p_ionization_energies     Pointer to table of ionization energies
     * \return Ionization cross section: sigma (m^2) and ionization energy: ionization_en (eV)
     */
    AMREX_GPU_HOST_DEVICE inline
    IonizationResult sigma_E(int atomic_number, int zstar, Real e,
        const Real* p_binding_energies, const Real* p_ionization_energies)
    {
        Real sigma = 0.0_rt;
        Real ionization_en = 0.0;

        int n = 0;      // degeneracy counter

        const int base_idx = 0;
        const Real me_c2 = 510998.95_rt;    // electron rest energy [eV]

        // Convert incident energy to m_e c^2 units
        const Real ep = e / me_c2;
        const Real shift = p_ionization_energies[zstar] - p_binding_energies[atomic_number - zstar - 1];

        // Sum over all bound electrons (k = 0 ... n_bound-1)
        for (int k = 0; k < atomic_number - zstar; k++) {

            int  idx_k   = base_idx + k;
            int  idx_k_p = base_idx + (k + 1);

            // Add current electron to the running group
            n++;

            Real bp = (p_binding_energies[idx_k] + shift) / me_c2;  // binding energy in m_e c^2 units

            // Accumulate degenerate sub-shells (same binding energy)
            if (k < atomic_number - zstar - 1 &&
                p_binding_energies[idx_k] == p_binding_energies[idx_k_p]) {
                continue;
            }

            Real e_red = ep / bp;    // reduced energy  T / B

            if (e_red > 1.0_rt) {
                // Relativistic beta^2 factors
                Real betae2 = 1.0_rt - 1.0_rt / ((1.0_rt + ep) * (1.0_rt + ep));
                Real betab2 = 1.0_rt - 1.0_rt / ((1.0_rt + bp) * (1.0_rt + bp));
                Real betau2 = betab2;   // RBEB: U ~ B

                // Pre-factor  s0 = (r_e / 2c) * n / [B (beta_e^2 + beta_b^2 + beta_u^2)]
                Real s0 = normalization * n /
                          (2.0_rt * bp * (betae2 + betab2 + betau2));

                Real ep2 = 1.0_rt / (1.0_rt + 0.5_rt * ep);
                ep2 *= ep2;

                // RBEB kernel terms
                Real a1c = (1.0_rt + 2.0_rt * ep) / (1.0_rt + e_red) * ep2;
                Real a2c = (e_red - 1.0_rt) * bp * bp * 0.5_rt * ep2;
                Real a3  = std::log(betae2 / (1.0_rt - betae2))
                         - betae2
                         - std::log(2.0_rt * bp);

                Real sk = s0 * (0.5_rt * a3 * (1.0_rt - 1.0_rt / (e_red * e_red))
                              + 1.0_rt - 1.0_rt / e_red
                              + a2c
                              - a1c * std::log(e_red));

                sigma += sk;
            }
            n = 0;   // reset degeneracy for the next shell
        }
        ionization_en = p_ionization_energies[base_idx + zstar];
        return {sigma, ionization_en};
    }

    /**
     * @brief Ionization cross section and mean kinetic energy of the ejected electron for a fast
     *        bare projectile (beam_impact), summed over all bound shells.
     *
     * Semiempirical model of Rudd et al., Rev. Mod. Phys. 64, 441 (1992), Eqs. (40)-(48), per shell
     * with N electrons and binding energy I, in the target rest frame.
     * Rudd's smooth cut-off is replaced by a sharp one at w_c, so that sigma and <W> are closed-form integrals;
     * this is exact when w_c >> v/alpha, i.e. above about 10 MeV per nucleon.
     * Same shell loop, binding-energy shift and degeneracy grouping as sigma_E.
     * Parameters (Rudd Table V): the Kr set for the outer shells of rubidium (Rb+ has the electron
     * configuration of Kr), and the inner-shell set for shells bound more than twice as tightly as
     * the outermost.
     *
     * \param[in] atomic_number             Atomic number of the target element (37)
     * \param[in] zstar                     Current ionization state
     * \param[in] gamma                     Lorentz factor of the projectile in the target rest frame
     * \param[in] me_over_M                 Electron mass over projectile mass
     * \param[in] p_binding_energies        Pointer to table of binding energies
     * \param[in] p_ionization_energies     Pointer to table of ionization energies
     * \param[out] sigma                    Cross section for unit projectile charge (m^2)
     * \param[out] w_mean                   Mean kinetic energy of the ejected electron (eV)
     */
    AMREX_GPU_HOST_DEVICE inline
    void rudd_sigma (int atomic_number, int zstar, Real gamma, Real me_over_M,
                     const Real* p_binding_energies, const Real* p_ionization_energies,
                     Real& sigma, Real& w_mean)
    {
        // Rudd et al. (1992), Table V: A1, B1, C1, D1, E1, A2, B2, C2, D2
        constexpr Real par_ar[9] = {1.20, 8.0, 0.86,  0.00, 0.80, 0.90,  2.7, 0.75,  0.80};
        constexpr Real par_kr[9] = {1.46, 5.7, 0.65, -0.55, 1.00, 1.30, 22.0, 0.95, -1.00};
        constexpr Real par_in[9] = {1.25, 0.5, 1.00,  1.00, 3.00, 1.10,  1.3, 1.00,  0.00};
        const Real me_c2 = 510998.95_rt;        // electron rest energy [eV]
        const Real rydberg = 13.605693122994_rt; // [eV]

        sigma = 0.0_rt;
        w_mean = 0.0_rt;
        const Real beta2 = 1.0_rt - 1.0_rt / (gamma * gamma);
        if (beta2 <= 0.0_rt) return;

        // Maximum energy transfer to a free electron at rest (eV)
        const Real w_max = 2.0_rt * me_c2 * beta2 * gamma * gamma /
            (1.0_rt + 2.0_rt * gamma * me_over_M + me_over_M * me_over_M);

        const Real shift = p_ionization_energies[zstar] - p_binding_energies[atomic_number - zstar - 1];

        Real moment = 0.0_rt;   // sum_k sigma_k <W>_k
        int n = 0;              // degeneracy counter
        for (int k = 0; k < atomic_number - zstar; k++) {
            n++;
            if (k < atomic_number - zstar - 1 &&
                p_binding_energies[k] == p_binding_energies[k + 1]) {
                continue;
            }
            const Real I = p_binding_energies[k] + shift;    // binding energy of the shell (eV)
            // Ar has its own set; Rb and Xe use the Kr set
            const Real* p_outer = (atomic_number == 18) ? par_ar : par_kr;
            const Real* p = (I > 2.0_rt * p_ionization_energies[zstar]) ? par_in : p_outer;

            // Reduced projectile velocity, v^2 = T/I with T = m_e c^2 beta^2 / 2
            const Real v2 = beta2 * me_c2 / (2.0_rt * I);
            const Real v = std::sqrt(v2);

            // Soft (dipole) strength F1 = L1 + H1, with the relativistic Bethe logarithm in H1
            const Real L1 = p[2] * std::pow(v, p[3]) / (1.0_rt + p[4] * std::pow(v, p[3] + 4.0_rt));
            const Real H1 = p[0] * (std::log1p(gamma * gamma * v2) - beta2) / (v2 + p[1] / v2);
            const Real F1 = L1 + H1;
            // Hard (Rutherford) strength F2 = L2 H2 / (L2 + H2)
            const Real L2 = p[7] * std::pow(v, p[8]);
            const Real H2 = p[5] / v2 + p[6] / (v2 * v2);
            const Real F2 = L2 * H2 / (L2 + H2);

            // Reduced cut-off energy
            const Real wc = w_max / I - 2.0_rt * v - rydberg / (4.0_rt * I);

            if (wc > 0.0_rt) {
                // J_n = int_0^wc w^n / (1+w)^3 dw
                const Real q = 1.0_rt / (1.0_rt + wc);
                const Real J0 = 0.5_rt * (1.0_rt - q * q);
                const Real J1 = (1.0_rt - q) - J0;
                const Real J2 = std::log1p(wc) - 2.0_rt * (1.0_rt - q) + J0;
                const Real S = 4.0_rt * MathConst::pi * a0 * a0 * n * (rydberg / I) * (rydberg / I);
                sigma  += S * (F1 * J0 + F2 * J1);
                moment += S * I * (F1 * J1 + F2 * J2);
            }
            n = 0;   // reset degeneracy for the next shell
        }
        w_mean = (sigma > 0.0_rt) ? moment / sigma : 0.0_rt;

        // Neutral Ar and Xe, fast projectile (beta > 0.5): total cross section from the Bethe
        // formula with the constants measured by Rieke and Prepejchal, Phys. Rev. A 6, 1507 (1972),
        //     sigma = 4 pi (hbar/mc)^2 / beta^2 [M^2 (ln(beta^2 gamma^2) - beta^2) + C],
        // 4 pi (hbar/mc)^2 = 1.874e-24 m^2. The Rudd spectrum above still gives w_mean.
        if (zstar == 0 && beta2 > 0.25_rt && sigma > 0.0_rt &&
            (atomic_number == 18 || atomic_number == 54)) {
            // Ar: Rieke & Prepejchal (1972), measured. Xe: Kim et al. (2000), counting ionization
            const Real M2 = (atomic_number == 18) ? 4.22_rt  : 5.157_rt;
            const Real C  = (atomic_number == 18) ? 37.93_rt : 57.96_rt;
            sigma = 1.874e-24_rt / beta2 * (M2 * (std::log(beta2 * gamma * gamma) - beta2) + C);
        }
    }
}

#endif // HIPACE_IMPACT_IONIZATION_SIGMA_H_
