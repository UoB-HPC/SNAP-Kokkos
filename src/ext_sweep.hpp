#include <stdlib.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_kernels.h"
#include "ext_problem.h"
#include "ext_profiler.h"

void sweep_octant(
        const unsigned int timestep, 
        const unsigned int oct, 
        const unsigned int ndiag, 
        const cell* cells,
        const int* num_cells,
        const unsigned int num_groups_todo);

void sweep_cell(
        const int istep,
        const int jstep,
        const int kstep,
        const unsigned int oct,
        const double* restrict l_flux_in,
        double* restrict l_flux_out,
        const cell * restrict cell_index,
        const unsigned int * restrict groups_todo,
        const unsigned int num_groups_todo,
        const unsigned int num_cells);

void compute_sweep_order(int** num_cells, cell** cells);

void perform_sweep(
        unsigned int num_groups_todo);

// Solve the transport equations for a single angle in a single cell for a single group
template <class Device>
struct sweep_cell_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;
    typedef Kokkos::View<const double*, Device> KConstView;
    typedef Kokkos::View<const unsigned int*, Device> KConstUIntView;

    sweep_cell_kernel(
            int nx,
            int ny,
            int nz,
            int cmom,
            const int istep,
            const int jstep,
            const int kstep,
            const unsigned int oct,
            KView mu,
            KView dd_j,
            KView dd_k,
            KView l_flux_in,
            KView l_flux_out,
            KView cell_index,
            KView source,
            KView scat_coeff,
            KView total_cross_section,
            KView time_delta,
            KConstView l_flux_in,
            KConstUIntView groups_todo,
            const unsigned int num_groups_todo,
            const unsigned int num_cells)
                : nx(nx), ny(ny), nz(nz), cmom(cmom), istep(istep), 
                jstep(jstep), kstep(kstep), oct(oct), mu(mu), dd_j(dd_j), 
                dd_k(dd_k), l_flux_in(l_flux_in), l_flux_out(l_flux_out), 
                cell_index(cell_index), source(source), scat_coeff(scat_coeff), 
                total_cross_section(total_cross_section), time_delta(time_delta), 
                l_flux_in(l_flux_in), groups_todo(groups_todo), 
                num_groups_todo(num_groups_todo), num_cells(num_cells) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // Get indexes for angle and group
            const unsigned int i = (istep > 0) ? cell_index[nc] 
                : nx - cell_index[nc].i - 1;
            const unsigned int j = (jstep > 0) ? cell_index[nc+1]
                : ny - cell_index[nc+1] - 1;
            const unsigned int k = (kstep > 0) ? cell_index[nc+2] 
                : nz - cell_index[nc+2] - 1;
            const unsigned int g = groups_todo[tg];

            // Assume transmissive (vacuum boundaries) and that we
            // are sweeping the whole grid so have access to all neighbours
            // This means that we only consider the case for one MPI task
            // at present.

            // Compute angular source
            // Begin with first scattering moment
            double source_term = source(0,i,j,k,g);

            // Add in the anisotropic scattering source moments
            for (unsigned int l = 1; l < cmom; l++)
            {
                source_term += scat_coeff(l,a,oct) * source(l,i,j,k,g);
            }

            double psi = source_term 
                + flux_i(a,g,j,k)*mu(a)*dd_i 
                + flux_j(a,g,i,k)*dd_j(a) 
                + flux_k(a,g,i,j)*dd_k(a);

            // Add contribution from last timestep flux if time-dependant
            if (time_delta(g) != 0.0)
            {
                psi += time_delta(g) * l_flux_in(a,g,i,j,k);
            }

            psi *= denom(a,g,i,j,k);

            // Compute upwind fluxes
            double tmp_flux_i = 2.0*psi - flux_i(a,g,j,k);
            double tmp_flux_j = 2.0*psi - flux_j(a,g,i,k);
            double tmp_flux_k = 2.0*psi - flux_k(a,g,i,j);

            // Time differencing on final flux value
            if (time_delta(g) != 0.0)
            {
                psi = 2.0 * psi - l_flux_in(a,g,i,j,k);
            }

            // Perform the fixup loop
            int num_to_fix = 4;
            // Fixup is a bounded loop as we will worst case fix up 
            // each face and centre value one after each other
            double zeros[4];
#pragma unroll(4)
            for (int fix = 0; fix < 4; fix++)
            {
                // Record which ones are zero
                zeros[0] = (tmp_flux_i < 0.0) ? 0.0 : 1.0;
                zeros[1] = (tmp_flux_j < 0.0) ? 0.0 : 1.0;
                zeros[2] = (tmp_flux_k < 0.0) ? 0.0 : 1.0;
                zeros[3] = (psi < 0.0) ? 0.0 : 1.0;

                if (num_to_fix == zeros[0] + zeros[1] + zeros[2] + zeros[3])
                {
                    break;
                }

                num_to_fix = zeros[0] + zeros[1] + zeros[2] + zeros[3];

                // Recompute cell centre value
                psi = flux_i(a,g,j,k)*mu(a)*dd_i*(1.0+zeros[0]) 
                    + flux_j(a,g,j,k)*dd_j(a)*(1.0+zeros[1]) 
                    + flux_k(a,g,i,j)*dd_k(a)*(1.0+zeros[2]);

                if (time_delta(g) != 0.0)
                {
                    psi += time_delta(g) * l_flux_in(a,g,i,j,k) * (1.0+zeros[3]);
                }
                psi = 0.5*psi + source_term;
                double recalc_denom = total_cross_section(g,i,j,k);
                recalc_denom += mu(a) * dd_i * zeros[0];
                recalc_denom += dd_j(a) * zeros[1];
                recalc_denom += dd_k(a) * zeros[2];
                recalc_denom += time_delta(g) * zeros[3];

                if (recalc_denom > 1.0E-12)
                {
                    psi /= recalc_denom;
                }
                else
                {
                    psi = 0.0;
                }

                // Recompute the edge fluxes with the new centre value
                tmp_flux_i = 2.0 * psi - flux_i(a,g,j,k);
                tmp_flux_j = 2.0 * psi - flux_j(a,g,i,k);
                tmp_flux_k = 2.0 * psi - flux_k(a,g,i,j);
                if (time_delta(g) != 0.0)
                {
                    psi = 2.0*psi - l_flux_in(a,g,i,j,k);
                }
            }

            // Fix up loop is done, just need to set the final values
            tmp_flux_i = tmp_flux_i * zeros[0];
            tmp_flux_j = tmp_flux_j * zeros[1];
            tmp_flux_k = tmp_flux_k * zeros[2];
            psi = psi * zeros[3];

            // Write values to global memory
            flux_i(a,g,j,k) = tmp_flux_i;
            flux_j(a,g,i,k) = tmp_flux_j;
            flux_k(a,g,i,j) = tmp_flux_k;
            l_flux_out(a,g,i,j,k) = psi;
        }

    int nx;
    int ny;
    int nz;
    int cmom;
    const int istep;
    const int jstep;
    const int kstep;
    const unsigned int oct;
    KView mu;
    KView dd_j;
    KView dd_k;
    KView l_flux_in;
    KView l_flux_out;
    KView cell_index;
    KView source;
    KView scat_coeff;
    KView total_cross_section;
    KView time_delta;
    KConstView l_flux_in;
    KConstUIntView groups_todo;
    const unsigned int num_groups_todo;
    const unsigned int num_cells
}
