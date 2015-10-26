// Calculate the inverted denominator for all the energy groups
template <class Device> 
struct calc_denominator_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_denominator_kernel(int nang, int ng, double dd_i,
            KView denom, KView total_cross_section,
            KView time_delta, KView mu, KView dd_j, KView dd_k)
        : nang(nang), ng(ng), denom(denom), time_delta(time_delta),
        total_cross_section(total_cross_section), mu(mu), 
        dd_j(dd_j), dd_k(dd_k) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // Reform indices
            // for (int ind = 0; ind < nx*ny*nz; ind++)
            //   for (int g = 0; g < ng; ++g)
            //     for (int a = 0; a < nang; ++a)
            const int ind = index/nang*ng;
            const int g = (index/nang)%ng;
            const int a = index%nang;

            denom[a+g*nang+ind*ng*nang] = 1.0 / (total_cross_section[g+ind*ng] 
                    + time_delta(g) + mu(a)*dd_i + dd_j(a) + dd_k(a));
        }

    int nang;
    int ng;
    double dd_i;
    KView denom;
    KView total_cross_section;
    KView time_delta;
    KView mu;
    KView dd_j;
    KView dd_k;
}

// Calculate the time delta
template <class Device>
struct calc_time_delta_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_time_delta_kernel(int ng, double dt, KView time_delta, KView velocity)
        : ng(ng), dt(dt), time_delta(time_delta), velocity(velocity) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            //for(int g = 0; g < ng; ++g)
            int g = index;
            time_delta(g) = 2.0 / (dt * velocity(g));
        }

    int ng; 
    double dt; 
    KView time_delta; 
    KView velocity;
}

// Calculate the diamond difference coefficients
template <class Device>
struct calc_dd_coefficients_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_dd_coefficients_kernel(int nang, double dx, double dy, double dz,
            KView dd_j, KView dd_k, KView eta, KView xi)
        : nang(nang), dx(dx), dy(dy), dz(dz), dd_j(dd_j), 
        dd_k(dd_k), eta(eta), xi(xi)
    {
        dd_i = 2.0 / dx;
    }

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            //for(int a = 0; a < nang; ++a)
            dd_j(a) = (2.0/dy)*eta(a);
            dd_k(a) = (2.0/dz)*xi(a);
        }

    int nang; 
    double dx; 
    double dy; 
    double dz;
    KView dd_j; 
    KView dd_k; 
    KView eta; 
    KView xi;
}

// Calculate the total cross section from the spatial mapping
template <class Device>
struct calc_total_cross_section_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_total_cross_section_kernel(int ng, int nx, int ny, int nz,
            KView total_cross_section, KView xs, KView mat)
        : ng(ng), nx(nx), ny(ny), nz(nz), xs(xs), mat(mat)
          total_cross_section(total_cross_section) {}


    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // for(int k = 0; k < nz; ++k)
            //     for(int j = 0; j < ny; ++j)
            //         for(int i = 0; i < nx; ++i)
            //             for(int g = 0; g < ng; ++g)
            total_cross_section(g,i,j,k) = xs(mat(i,j,k)-1,g);
        }

    int ng; 
    int nx; 
    int ny; 
    int nz; 
    KView total_cross_section; 
    KView xs; 
    KView mat;
}

template <class Device>
struct calc_scattering_cross_section_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_scattering_cross_section_kernel(int nmom, int nx, int ny, int nz, int ng,
            KView scat_cs, KView gg_cs, KView mat)
        : nmom(nmom), nx(nx), ny(ny), nz(nz), ng(ng), 
        scat_cs(scat_cs), gg_cs(gg_cs), mat(mat) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // for(unsigned int g = 0; g < ng; ++g)
            //     for (unsigned int k = 0; k < nz; k++)
            //         for (unsigned int j = 0; j < ny; j++)
            //             for (unsigned int i = 0; i < nx; i++)
            //                 for (unsigned int l = 0; l < nmom; l++)
            scat_cs(l,i,j,k,g) = gg_cs(mat(i,j,k)-1,l,g,g);
        }

    int nmom;
    int nx;
    int ny;
    int nz;
    int ng;
    KView scat_cs;
    KView gg_cs;
    KView mat;
}

// Calculate the outer source
template <class Device>
struct calc_outer_source_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_outer_source_kernel(int nx, int ny, int nz, int ng, int nmom,
            KView g2g_souce, KView fixed_source, KView gg_cs,
            KView mat, KView scalar_flux, KView scalar_mom,
            KView lma)
        : nx(nx), ny(ny), nz(nz), ng(ng), nmom(nmom), 
        g2g_souce(g2g_souce), fixed_source(fixed_source), 
        gg_cs(gg_cs), mat(mat), scalar_flux(scalar_flux), 
        scalar_mom(scalar_mom), lma(lma) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // for (unsigned int g1 = 0; g1 < ng; g1++)
            //    for(int k = 0; k < nz; ++k)
            //        for(int j = 0; j < ny; ++j)
            //            for(int i = 0; i < nx; ++i)
            g2g_source(0,i,j,k,g1) = fixed_source(i,j,k,g1);

            for (unsigned int g2 = 0; g2 < ng; g2++)
            {
                if (g1 == g2)
                {
                    continue;
                }

                g2g_source(0,i,j,k,g1) += gg_cs(mat(i,j,k)-1,0,g2,g1) * scalar_flux(g2,i,j,k);

                unsigned int mom = 1;
                for (unsigned int l = 1; l < nmom; l++)
                {
                    for (int m = 0; m < lma(l); m++)
                    {
                        // TODO: CHECK WHY THIS CONDITION WAS NECESSARY
                        if(mom < (cmom-1))
                        {
                            g2g_source(mom,i,j,k,g1) += gg_cs(mat(i,j,k)-1,l,g2,g1) 
                                * scalar_mom(g2,mom-1,i,j,k);
                        }

                        mom++;
                    }
                }
            }
        }

    int nx;
    int ny;
    int nz;
    int ng;
    int nmom;
    KView g2g_souce;
    KView fixed_source;
    KView gg_cs;
    KView mat;
    KView scalar_flux;
    KView scalar_mom;
    KView lma;
}

// Calculate the inner source
template <class Device>
struct calc_inner_source_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_inner_source_kernel(int nx, int ny, int nz, int ng,
            KView source, KView g2g_source, KView scat_cs, 
            KView scalar_flux, KView source, KView scalar_mom,
            KView lma)
        : nx(nx), ny(ny), nz(nz), ng(ng), source(source), 
        g2g_source(g2g_source), scat_cs(scat_cs), 
        scalar_flux(scalar_flux), source(source), 
        scalar_mom(scalar_mom), lma(lma) {}


    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            // for (unsigned int g = 0; g < ng; g++)
            //     for(int k = 0; k < nz; ++k)
            //         for(int j = 0; j < ny; ++j)
            //             for(int i = 0; i < nx; ++i)

            source(0,i,j,k,g) = g2g_source(0,i,j,k,g) + scat_cs(0,i,j,k,g) * scalar_flux(g,i,j,k);

            unsigned int mom = 1;
            for (unsigned int l = 1; l < nmom; l++)
            {
                for (int m = 0; m < lma(l); m++)
                {
                    source(mom,i,j,k,g) = g2g_source(mom,i,j,k,g) + scat_cs(l,i,j,k,g) * scalar_mom(g,mom-1,i,j,k);
                    mom++;
                }
            }
        }

    int nx;
    int ny;
    int nz;
    int ng;
    KView source;
    KView g2g_source;
    KView scat_cs;
    KView scalar_flux;
    KView source;
    KView scalar_mom;
    KView lma
}

template <class Device>
struct zero_flux_in_out_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    zero_flux_in_out_kernel(int flux_in_len, int flux_out_len,
            KView flux_in, KView flux_out)
        :flux_in_len(flux_in_len), flux_out_len(flux_out_len), 
        flux_in(flux_in), flux_out(flux_out) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            if(index < flux_in_len)
                flux_in[index] = 0.0;

            if(index < flux_out_len)
                flux_out[i] = 0.0;
        }

    int flux_in_len;
    int flux_out_len;
    KView flux_in;
    KView flux_out
}


template <class Device>
struct zero_edge_flux_buffers_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    zero_edge_flux_buffers_kernel(int fi_len, int fj_len, int fk_len,
            KView flux_i, KView flux_j, KView flux_k)
        : fi_len(fi_len), fj_len(fj_len), fk_len(fk_len), 
        flux_i(flux_i), flux_j(flux_j), flux_k(flux_k) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            if(i < fi_len) flux_i[i] = 0.0;
            if(i < fj_len) flux_j[i] = 0.0;
            if(i < fk_len) flux_k[i] = 0.0;
        }

    int fi_len;
    int fj_len;
    int fk_len;
    KView flux_i;
    KView flux_j;
    KView flux_k
}

template <class Device>
struct zero_flux_moments_buffer_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    zero_flux_buffers_kernel(KView scalar_mom)
        : scalar_mom(scalar_mom) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            //for(int i = 0; i < (cmom-1)*nx*ny*nz*ng; ++i)
            scalar_mom[i] = 0.0;
        }

    KView scalar_mom;
}

template <class Device>
struct zero_scalar_flux_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    zero_scalar_flux_kernel(KView scalar_flux)
        : scalar_flux(scalar_flux) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            //for(int i = 0; i < nx*ny*nz*ng; ++i)
            scalar_flux[i] = 0.0;
        }

    KView scalar_flux;
}

int check_convergence(
        double *old, 
        double *new, 
        double epsi, 
        unsigned int *groups_todo, 
        unsigned int *num_groups_todo, 
        int inner)
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;
    typedef Kokkos::View<int*, Device> KViewInteger;

    int r = 1;
    int ngt = 0;
    //for (unsigned int g = 0; g < ng; g++)

    check_convergence(int g, int ng, int nx, int ny, int nz, 
            bool inner, double epsi, KView new, KView old)
        : ng(ng), nx(nx), ny(ny), nz(nz), 
        inner(inner), epsi(epsi), new(new), old(old) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            if(!*r) break; // Drop all threads for group once tolerance exceeded

            double val = (fabs(old[g+(ng*index)] > tolr))
                ? fabs(new[g+(ng*index)]/old[g+(ng*index)] - 1.0)
                : fabs(new[g+(ng*index)] - old[g+(ng*index)]);

            if (val > epsi)
            {
                if (inner)
                {
                    groups_todo[ngt] = g;
                    ngt += 1;
                }

                *r = 0;
            }
        }

    int g;
    int ng;
    int nx;
    int ny;
    int nz;
    bool inner;
    double epsi;
    KView new;
    KView old;
    KViewInteger r;
}

template <class Device>
struct zero_g2g_source_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    zero_g2g_source_kernel(int g2g_source_len, KView g2g_source)
        : g2g_source_len(g2g_source_len), g2g_source(g2g_source) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            g2g_source[index] = 0.0;
        }

    int g2g_source_len;
    KView g2g_source;
}

// Copies the value of scalar flux
template <class Device>
struct store_scalar_flux_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    store_scalar_flux_kernel(KView to, KView scalar_flux)
        : to(to), scalar_flux(scalar_flux) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            to[i] = scalar_flux[i];
        }

    KView to;
    KView scalar_flux;
}
