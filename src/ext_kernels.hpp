#pragma once 

#include <stdbool.h>
#include "ext_shared.hpp"

void initialise_device_memory(void);
void zero_edge_flux_buffers(void);
void zero_flux_moments_buffer(void);
void zero_flux_in_out(void);
void zero_scalar_flux(void);
void calc_inner_source(void);
void calc_outer_source(void);
void calc_scattering_cross_section(void);
void calc_dd_coefficients(void);
void calc_time_delta(void);
void calc_denominator(void);
void calc_total_cross_section(void);

void store_scalar_flux(double* to);

bool check_convergence(
		double* oldval, 
		double* newval, 
		double epsi, 
		unsigned int *groups_todo, 
		unsigned int *num_groups_todo, 
		bool inner);

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
};

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
};

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
            if(index >= nang) return;

            int a = index;
            dd_j(a) = (2.0/dy)*eta(a);
            dd_k(a) = (2.0/dz)*xi(a);
        }

    int nang; 
    double dx; 
    double dy; 
    double dz;
    double dd_i;
    KView dd_j; 
    KView dd_k; 
    KView eta; 
    KView xi;
};

// Calculate the total cross section from the spatial mapping
template <class Device>
struct calc_total_cross_section_kernel
{
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_total_cross_section_kernel(int ng, int nx, int ny, int nz,
            KView total_cross_section, KView xs, KView mat)
        : ng(ng), nx(nx), ny(ny), nz(nz), xs(xs), mat(mat),
          total_cross_section(total_cross_section) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            if(index > ng*nx*ny*nz) return;

            int c = index / ng;
            int g = index % ng;
            total_cross_section(g,c) = xs(mat(c)-1,g);
        }

    int ng; 
    int nx; 
    int ny; 
    int nz; 
    KView total_cross_section; 
    KView xs; 
    KView mat;
};

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
            if(index >= ng*nz*ny*nx*nmom) return;

            int l = index % nmom;
            int c = (index / nmom) % nx*ny*nz;
            int g = (index / nmom*nx*ny*nz);
            scat_cs(l,c,g) = gg_cs(mat(c)-1,l,g,g);
        }

    int nmom;
    int nx;
    int ny;
    int nz;
    int ng;
    KView scat_cs;
    KView gg_cs;
    KView mat;
};

// Calculate the outer source
template <class Device>
struct calc_outer_source_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_outer_source_kernel(int nx, int ny, int nz, int ng, 
            int nmom, int cmom, KView g2g_souce, KView fixed_source, 
            KView gg_cs, KView mat, KView scalar_flux, KView scalar_mom,
            KView lma)
        : nx(nx), ny(ny), nz(nz), ng(ng), nmom(nmom), 
        g2g_source(g2g_source), fixed_source(fixed_source), 
        gg_cs(gg_cs), mat(mat), scalar_flux(scalar_flux), 
        scalar_mom(scalar_mom), lma(lma) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            if(index >= ng*nz*ny*nx) return;

            int c = index % nx*ny*nz;
            int g1 = index / nx*ny*nz;
            g2g_source(0,c,g1) = fixed_source(c,g1);

            for (unsigned int g2 = 0; g2 < ng; g2++)
            {
                if (g1 == g2)
                {
                    continue;
                }

                g2g_source(0,c,g1) += gg_cs(mat(c)-1,0,g2,g1) * scalar_flux(g2,c);

                unsigned int mom = 1;
                for (unsigned int l = 1; l < nmom; l++)
                {
                    for (int m = 0; m < lma(l); m++)
                    {
                        // TODO: CHECK WHY THIS CONDITION WAS NECESSARY
                        if(mom < (cmom-1))
                        {
                            g2g_source(mom,c,g1) += gg_cs(mat(c)-1,l,g2,g1) 
                                * scalar_mom(g2,mom-1,c);
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
    int cmom;
    KView g2g_source;
    KView fixed_source;
    KView gg_cs;
    KView mat;
    KView scalar_flux;
    KView scalar_mom;
    KView lma;
};

// Calculate the inner source
template <class Device>
struct calc_inner_source_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;

    calc_inner_source_kernel(int nx, int ny, int nz, int ng,
            KView source, KView g2g_source, KView scat_cs, 
            KView scalar_flux, KView scalar_mom,
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
    KView scalar_mom;
    KView lma;
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

template <class Device>
struct check_convergence_kernel
{    
    typedef Device device_type;
    typedef Kokkos::View<double*, Device> KView;
    typedef Kokkos::View<int*, Device> KViewInteger;

    check_convergence(int g, int ng, int nx, int ny, int nz, 
            bool inner, double epsi, KView newval, KView oldval, KViewInteger r)
        : ng(ng), nx(nx), ny(ny), nz(nz),  r(r),
        inner(inner), epsi(epsi), newval(newval), oldval(oldval) {}

    KOKKOS_INLINE_FUNCTION
        void operator() (const int index) const
        {
            if(!*r) break; // Drop all threads for group once tolerance exceeded

            double val = (fabs(oldval[g+(ng*index)] > tolr))
                ? fabs(newval[g+(ng*index)]/oldval[g+(ng*index)] - 1.0)
                : fabs(newval[g+(ng*index)] - oldval[g+(ng*index)]);

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
    KView newval;
    KView oldval;
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
