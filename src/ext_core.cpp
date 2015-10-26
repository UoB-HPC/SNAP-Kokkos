#include "ext_core.hpp"

void ext_solve_(
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
        int *lma_in)
{
    initialise_host_memory(mu_in, eta_in, xi_in, scat_coeff_in, weights_in, velocity_in,
            xs_in, mat_in, fixed_source_in, gg_cs_in, lma_in);

    initialise_device_memory();
    iterate();

    free(old_outer_scalar);
    free(new_scalar);
    free(old_inner_scalar);
    free(groups_todo);
}

// Argument list:
// nx, ny, nz are the (local to MPI task) dimensions of the grid
// ng is the number of energy groups
// cmom is the "computational number of moments"
// ichunk is the number of yz planes in the KBA decomposition
// dd_i, dd_j(nang), dd_k(nang) is the x,y,z (resp) diamond difference coefficients
// mu(nang) is x-direction cosines
// scat_coef [ec](nang,cmom,noct) - Scattering expansion coefficients
// time_delta [vdelt](ng)              - time-absorption coefficient
// denom(nang,nx,ny,nz,ng) - Sweep denominator, pre-computed/inverted
// weights(nang) - angle weights for scalar reduction
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
        int *lma_in)
{
    START_PROFILING;

    flux_i = (double*)malloc(sizeof(double)*nang*ny*nz*ng);
    flux_j = (double*)malloc(sizeof(double)*nang*nx*nz*ng);
    flux_k = (double*)malloc(sizeof(double)*nang*nx*ny*ng);
    dd_j = (double*)malloc(sizeof(double)*nang);
    dd_k = (double*)malloc(sizeof(double)*nang);
    total_cross_section = (double*)malloc(sizeof(double)*nx*ny*nz*ng);
    scat_cs = (double*)malloc(sizeof(double)*nmom*nx*ny*nz*ng);
    denom = (double*)malloc(sizeof(double)*nang*nx*ny*nz*ng);
    source = (double*)malloc(sizeof(double)*cmom*nx*ny*nz*ng);
    time_delta = (double*)malloc(sizeof(double)*ng);
    groups_todo = (unsigned int*)malloc(sizeof(unsigned int)*ng);
    g2g_source = (double*)malloc(sizeof(double)*cmom*nx*ny*nz*ng);
    scalar_flux = (double*)malloc(sizeof(double)*nx*ny*nz*ng);
    flux_in = (double*)malloc(sizeof(double)*nang*nx*ny*nz*ng*noct);
    flux_out = (double*)malloc(sizeof(double)*nang*nx*ny*nz*ng*noct);
    scalar_mom = (double*)malloc(sizeof(double)*(cmom-1)*nx*ny*nz*ng);
    old_outer_scalar = (double*)malloc(sizeof(double)*nx*ny*nz*ng);
    old_inner_scalar = (double*)malloc(sizeof(double)*nx*ny*nz*ng);
    new_scalar = (double*)malloc(sizeof(double)*nx*ny*nz*ng);

    // Read-only buffers initialised in Fortran code
    mu = mu_in;
    eta = eta_in;
    xi = xi_in;
    weights = weights_in;
    velocity = velocity_in;
    mat = mat_in;
    fixed_source = fixed_source_in;
    gg_cs = gg_cs_in;
    lma = lma_in;
    xs = xs_in;

    STOP_PROFILING(__func__, false);

    scat_coeff = transpose_scat_coeff(scat_coeff_in);
}

// Initialises the problem parameters
void ext_initialise_parameters_(
        int *nx_, int *ny_, int *nz_,
        int *ng_, int *nang_, int *noct_,
        int *cmom_, int *nmom_,
        int *ichunk_,
        double *dx_, double *dy_, double *dz_,
        double *dt_,
        int *nmat_,
        int *timesteps_, int *outers_, int *inners_,
        double *epsi_, double *tolr_)
{
    START_PROFILING;

    // Save problem size information to globals
    nx = *nx_;
    ny = *ny_;
    nz = *nz_;
    ng = *ng_;
    nang = *nang_;
    noct = *noct_;
    cmom = *cmom_;
    nmom = *nmom_;
    ichunk = *ichunk_;
    dx = *dx_;
    dy = *dy_;
    dz = *dz_;
    dt = *dt_;
    nmat = *nmat_;
    timesteps = *timesteps_;
    outers = *outers_;
    inners = *inners_;

    epsi = *epsi_;
    tolr = *tolr_;

    if (nx != ichunk)
    {
        printf("Warning: nx and ichunk are different - expect the answers to be wrong...\n");
    }

    STOP_PROFILING(__func__, false);
}

// Do the timestep, outer and inner iterations
void iterate(void)
{
    unsigned int num_groups_todo;
    int outer_done;

    double t1 = omp_get_wtime();

    // Timestep loop
    for (unsigned int t = 0; t < timesteps; t++)
    {
        unsigned int tot_outers = 0;
        unsigned int tot_inners = 0;
        global_timestep = t;

        // Calculate data required at the beginning of each timestep
        zero_scalar_flux();
        zero_flux_moments_buffer();

        // Outer loop
        outer_done = false;

        for (unsigned int o = 0; o < outers; o++)
        {
            // Reset the inner convergence list
            int inner_done = false;

            for (unsigned int g = 0; g < ng; g++)
            {
                groups_todo[g] = g;
            }

            num_groups_todo = ng;
            tot_outers++;

            calc_total_cross_section();
            calc_scattering_cross_section();
            calc_dd_coefficients();
            calc_time_delta();
            calc_denominator();
            printf("calced\n");

            // Compute the outer source
            calc_outer_source();
            printf("calc outer\n");

            // Save flux
            store_scalar_flux(old_outer_scalar);
            printf("store scalar\n");

            // Inner loop
            for (unsigned int i = 0; i < inners; i++)
            {
                tot_inners++;

                // Compute the inner source
                calc_inner_source();
                printf("calc_inner\n");

                // Save flux
                store_scalar_flux(old_inner_scalar);
                zero_edge_flux_buffers();

#ifdef TIMING
                double t1 = omp_get_wtime();
#endif

                // Sweep
                perform_sweep(num_groups_todo);

#ifdef TIMING
                double t2 = omp_get_wtime();
                printf("sweep took: %lfs\n", t2-t1);
#endif

                // Scalar flux
                reduce_angular();
#ifdef TIMING
                double t3 = omp_get_wtime();
                printf("reductions took: %lfs\n", t3-t2);
#endif

                // Check convergence
                store_scalar_flux(new_scalar);

#ifdef TIMING
                double t4 = omp_get_wtime();
#endif

                inner_done = check_convergence(old_inner_scalar, new_scalar, epsi, groups_todo, &num_groups_todo, true);

#ifdef TIMING
                double t5 = omp_get_wtime();
                printf("inner conv test took %lfs\n",t5-t4);
#endif
                if (inner_done)
                {
                    break;
                }
            }

            // Check convergence
            outer_done = check_convergence(old_outer_scalar, new_scalar, 100.0*epsi, groups_todo, &num_groups_todo, false);

            if (outer_done && inner_done)
            {
                break;
            }
        }

        printf("Time %d -  %d outers, %d inners.\n", t, tot_outers, tot_inners);

        // Exit the time loop early if outer not converged
        if (!outer_done)
        {
            break;
        }
    }

    double t2 = omp_get_wtime();

    printf("Time to convergence: %.3lfs\n", t2-t1);

    if (!outer_done)
    {
        printf("Warning: did not converge\n");
    }

    PRINT_PROFILING_RESULTS;
}

// Compute the scalar flux from the angular flux
void reduce_angular(void)
{
    START_PROFILING;

    zero_scalar_flux();
    zero_flux_moments_buffer();

    double* angular = (global_timestep % 2 == 0) ? flux_out : flux_in;
    double* angular_prev = (global_timestep % 2 == 0) ? flux_in : flux_out;

    int n = nx*ny*nz*nang*ng;
    for(unsigned int o = 0; o < 8; ++o)
    {
        PROFILED_PARALLEL_FOR(n, reduce_angular_kernel);
    }

    STOP_PROFILING(__func__, true);
}

// Copy the scalar flux value back to the host and transpose
void ext_get_transpose_scalar_flux_(double *scalar)
{
    // Transpose the data into the original SNAP format
    for (unsigned int g = 0; g < ng; g++)
    {
        for (unsigned int k = 0; k < nz; k++)
        {
            for (unsigned int j = 0; j < ny; j++)
            {
                for (unsigned int i = 0; i < nx; i++)
                {
                    scalar[i+(nx*j)+(nx*ny*k)+(nx*ny*nz*g)] 
                        = scalar_flux[g+(ng*i)+(ng*nx*j)+(ng*nx*ny*k)];
                }
            }
        }
    }
}

void ext_get_transpose_scalar_moments_(double *scalar_moments)
{
    // Transpose the data into the original SNAP format
    for (unsigned int g = 0; g < ng; g++)
    {
        for (unsigned int l = 0; l < cmom-1; l++)
        {
            for (unsigned int k = 0; k < nz; k++)
            {
                for (unsigned int j = 0; j < ny; j++)
                {
                    for (unsigned int i = 0; i < nx; i++)
                    {
                        scalar_moments[l+((cmom-1)*i)+((cmom-1)*nx*j)+((cmom-1)*nx*ny*k)+((cmom-1)*nx*ny*nz*g)] 
                            = scalar_mom[(l)+(g*(cmom-1))+(ng*(cmom-1)*i)+(ng*(cmom-1)*nx*j)+(ng*(cmom-1)*nx*ny*k)];
                    }
                }
            }
        }
    }
}

// Copy the flux_out buffer back to the host
void ext_get_transpose_output_flux_(double* output_flux)
{
    double *tmp = (global_timestep % 2 == 0) ? flux_out : flux_in;

    // Transpose the data into the original SNAP format
    for (int a = 0; a < nang; a++)
    {
        for (int g = 0; g < ng; g++)
        {
            for (int k = 0; k < nz; k++)
            {
                for (int j = 0; j < ny; j++)
                {
                    for (int i = 0; i < nx; i++)
                    {
                        for (int o = 0; o < noct; o++)
                        {
                            output_flux[a+(nang*i)+(nang*nx*j)+(nang*nx*ny*k)+(nang*nx*ny*nz*o)+(nang*nx*ny*nz*noct*g)] 
                                = tmp[a+(nang*g)+(nang*ng*i)+(nang*ng*nx*j)+(nang*ng*nx*ny*k)+(nang*nx*ny*nz*ng*o)];
                        }
                    }
                }
            }
        }
    }
}

// Transpose the scatter coefficients matrix
double* transpose_scat_coeff(double* scat_coeff_in)
{
    START_PROFILING;

    double* scat_coeff = (double*)malloc(sizeof(double)*nang*cmom*noct);

    for(unsigned int o = 0; o < noct; ++o)
    {
        for(unsigned int l = 0; l < cmom; ++l)
        {
            for(unsigned int a = 0; a < nang; ++a)
            {
                scat_coeff[l+a*cmom+o*(cmom*nang)] = 
                    scat_coeff_in[a+l*nang+o*(cmom*nang)];
            }
        }
    }

    STOP_PROFILING(__func__,true);

    return scat_coeff;
}
