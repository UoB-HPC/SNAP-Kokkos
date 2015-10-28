#include "ext_shared.hpp"

// Entry point for completing the solve
void ext_solve_(
		double *mu, 
		double *eta, 
		double *xi,
		double *scat_coeff,
		double *weights,
		double *velocity,
		double *xs,
		int *mat,
		double *fixed_source,
		double *gg_cs,
		int *lma);

// Entry point for initialising problem params
void initialise_parameters(
    int *nx_, int *ny_, int *nz_,
    int *ng_, int *nang_, int *noct_,
    int *cmom_, int *nmom_,
    int *ichunk_,
    double *dx_, double *dy_, double *dz_,
    double *dt_,
    int *nmat_,
    int *timesteps_, int *outers_, int *inners_,
    double *epsi_, double *tolr_);

// Allocates buffers on the device
void initialise_host_memory(
		double *mu_in, 
		double *eta_in, 
		double *xi_in,
		double *scat_coeff_in,
		double *weights_in,
		double *velocity_in,
		double *xs_in,
		int *mat_in,
		double *fixed_source_in,
		double *gg_cs_in,
		int *lma_in);

// Do the timestep, outer and inner iterations
void iterate(void);

// Compute the scalar flux from the angular flux
void reduce_angular(void);

// Transposes the scattering coefficient matrix
double* transpose_scat_coeff(double* scat_coeff_in);

// Transposes the scalar flux back to SNAP format
void ext_get_transpose_scalar_flux_(double *scalar);

// Transposes the scalar moments back to SNAP format
void ext_get_transpose_scalar_moments_(double *scalar_moments);

// Transposes the output flux back to SNAP format
void ext_get_transpose_output_flux_(double* output_flux);

template <class Device>
struct reduce_angular_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    reduce_angular_kernel(
            int nx, int ny, int nz, int ng, int nang, int cmom,
            KView angular, KView angular_prev, KView scalar_mom,
            KView weights, KView scalar_flux, KView scat_coeff)
        : nx(nx), ny(ny), nz(nz), ng(ng), nang(nang), cmom(cmom), 
        angular(angular), angular_prev(angular_prev), scalar_mom(scalar_mom), 
        weights(weights), scalar_flux(scalar_flux), scat_coeff(scat_coeff) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // for(unsigned int ind = 0; ind < nx*ny*nz; ++ind)
            //     for (unsigned int g = 0; g < ng; g++)
            //         for (unsigned int a = 0; a < nang; a++)
            const double weight = weights(a);
            const double ang = angular(o,ind,g,a);
            const double ang_p = angular_prev(o,ind,g,a);

            if (time_delta(g) != 0.0)
            {
                scalar_flux[g+ind*ng] += weight * (0.5 * (ang + ang_p));

                for (unsigned int l = 0; l < (cmom-1); l++)
                {
                    scalar_mom[l+g*(cmom-1)+(ng*(cmom-1)*ind)] += 
                        scat_coeff(l+1,a,o) * weight * (0.5 * (ang + ang_p));
                }
            }
            else
            {
                scalar_flux[g+ind*ng] += weight * ang;

                for (unsigned int l = 0; l < (cmom-1); l++)
                {
                    scalar_mom[l+g*(cmom-1)+(ng*(cmom-1)*ind)] += 
                        scat_coeff(l+1,a,o) * weight * ang;
                }

            }
        }
}
