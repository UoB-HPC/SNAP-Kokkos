#include <stdint.h>
#include <math.h>

#include "ext_sweep.h"
#include "ext_macros.h"
#include "ext_problem.h"
#include "ext_profiler.h"

// Calculate the inverted denominator for all the energy groups
void calc_denominator(void)
{
    PROFILED_PARALLEL_FOR(nx*ny*nz*ng*nang,
            calc_denominator_kernel<DEVICE>(
                nang, ng, dd_i, denom, total_cross_section, mu));
}

// Calculate the time delta
void calc_time_delta(void)
{
    PROFILED_PARALLEL_FOR(ng,
            calc_time_delta_kernel<DEVICE>(
                ng, dt, time_delta, velocity));
}

// Calculate the diamond difference coefficients
void calc_dd_coefficients(void)
{
    PROFILED_PARALLEL_FOR(nang,
            calc_dd_coefficients<DEVICE>(
                nang, dx, dy, dz, dd_j, dd_k, eta, xi));

}

// Calculate the total cross section from the spatial mapping
void calc_total_cross_section(void)
{
    PROFILED_PARALLEL_FOR( ,
            calc_total_cross_section_kernel<DEVICE>(
                ng, nx, ny, nz, total_cross_section, xs, mat));
}

void calc_scattering_cross_section(void)
{
    PROFILED_PARALLEL_FOR( ,
            calc_scattering_cross_section_kernel<DEVICE>(
                nmom, nx, ny, nz, ng, scat_cs, gg_cs, mat));
}

// Calculate the outer source
void calc_outer_source(void)
{
    PROFILED_PARALLEL_FOR( ,
            calc_outer_source_kernel<DEVICE>(
                nx, ny, nz, ng, nmom, g2g_souce, fixed_source, 
                gg_cs, mat, scalar_flux, scalar_mom, lma));
}

// Calculate the inner source
void calc_inner_source(void)
{
    PROFILED_PARALLEL_FOR( ,
            calc_inner_source_kernel<DEVICE>(
                nx, ny, nz, ng, source, g2g_source, 
                scat_cs, scalar_flux, source, scalar_mom, lma));
}

void zero_flux_in_out(void)
{
    PROFILED_PARALLEL_FOR( ,
            zero_flux_in_out_kernel<DEVICE>(
                flux_in_len, flux_out_len, flux_in, flux_out));
}

void zero_edge_flux_buffers(void)
{
    int fi_len = nang*ng*ny*nz;
    int fj_len = nang*ng*nx*nz;
    int fk_len = nang*ng*nx*ny;

#define MAX(A,B) (((A) > (B)) ? (A) : (B))
    int max_length = MAX(MAX(fi_len, fj_len), fk_len);

    PROFILED_PARALLEL_FOR( ,
            zero_edge_flux_buffers_kernel<DEVICE>(
                fi_len, fj_len, fk_len, flux_i, flux_j, flux_k));
}

void zero_flux_moments_buffer(void)
{
    PROFILED_PARALLEL_FOR( ,
            zero_flux_moments_buffer_kernel<DEVICE>(
                scalar_mom));
}

void zero_scalar_flux(void)
{
    PROFILED_PARALLEL_FOR( ,
            zero_scalar_flux_kernel<DEVICE>(
                scalar_flux));
}

void zero_g2g_source(void)
{
    PROFILED_PARALLEL_FOR( ,
            zero_g2g_source_kernel<DEVICE>(
                g2g_source));
}

int check_convergence(
        double *old, 
        double *new, 
        double epsi, 
        unsigned int *groups_todo, 
        unsigned int *num_groups_todo, 
        int inner)
{
    START_PROFILING;

    int r = 1;
    int ngt = 0;

    for (unsigned int g = 0; g < ng; g++)
    {
        PROFILED_PARALLEL_FOR(nx*ny*nz,
                check_convergence_kernel<DEVICE>(
                    g, ng, nx, ny, nz, inner, epsi, new, old));
    }

    // Check all inner groups are done in outer convergence test
    if (!inner && ngt != 0)
    {
        r = 0;
    }

    if(inner)
    {
        *num_groups_todo = ngt;
    }
    else
    {
        *num_groups_todo += ngt;
    }

    STOP_PROFILING(__func__, 1);

    return r;
}

void initialise_device_memory(void)
{
    zero_scalar_flux();
    zero_edge_flux_buffers();
    zero_flux_moments_buffer();
    zero_flux_in_out();
    zero_g2g_source();
}

// Copies the value of scalar flux
void store_scalar_flux(double* to)
{
    PROFILED_PARALLEL_FOR( ,
            store_scalar_flux_kernel<DEVICE>(
                ));
}
