#pragma once

#define PROFILED_PARALLEL_FOR(n, kernel) \
    START_PROFILING; \
    Kokkos::parallel_for(n, kernel); \
    STOP_PROFILING(__func__, 1);

/*
 *		Array access macros
 */
#define l_flux_out(a,g,i,j,k) l_flux_out[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define l_flux_in(a,g,i,j,k) l_flux_in[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define source(m,i,j,k,g) source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define flux_i(a,g,j,k) flux_i[(a)+(nang*(g))+(nang*ng*(j))+(nang*ng*ny*(k))]
#define flux_j(a,g,i,k) flux_j[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(k))]
#define flux_k(a,g,i,j) flux_k[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))]
#define denom(a,g,i,j,k) denom[(a)+(nang*(g))+(nang*ng*(i))+(nang*ng*nx*(j))+(nang*ng*nx*ny*(k))]
#define dd_j(a) dd_j[(a)]
#define dd_k(a) dd_k[(a)]
#define mu(a) mu[(a)]
#define eta(a) eta[(a)]
#define xi(a) xi[(a)]
#define scat_coeff(m,a,o) scat_coeff[(m)+(a*cmom)+(nang*cmom*(o))]
#define time_delta(g) time_delta[(g)]
#define total_cross_section(g,i,j,k) total_cross_section[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define scalar_flux(g,i,j,k) scalar_flux[(g)+(ng*(i))+(ng*nx*(j))+(ng*nx*ny*(k))]
#define weights(a) weights[(a)]

#define angular(o,ind,g,a) angular[a+g*nang+nang*ng*ind+(nang*nx*ny*nz*ng*(o))]
#define angular_prev(o,ind,g,a) angular_prev[a+g*nang+nang*ng*ind+(nang*nx*ny*nz*ng*(o))]

#define velocity(g) velocity[(g)]

#define mat(i,j,k) mat[(i)+(nx*(j))+(nx*ny*(k))]
#define xs(i,g) xs[(i)+(nmat*(g))]

#define g2g_source(m,i,j,k,g) g2g_source[(m)+(cmom*(i))+(cmom*nx*(j))+(cmom*nx*ny*(k))+(cmom*nx*ny*nz*(g))]
#define fixed_source(i,j,k,g) fixed_source[(i)+(nx*(j))+(nx*ny*(k))+(nx*ny*nz*(g))]
#define gg_cs(m,l,g1,g2) gg_cs[(m)+(nmat*(l))+(nmat*nmom*(g1))+(nmat*nmom*ng*(g2))]
#define lma(m) lma[(m)]
#define scalar_mom(g,m,i,j,k) scalar_mom[(g)+((ng)*(m))+(ng*(cmom-1)*(i))+(ng*(cmom-1)*nx*(j))+(ng*(cmom-1)*nx*ny*(k))]

#define scat_cs(m,i,j,k,g) scat_cs[(m)+(nmom*(i))+(nmom*nx*(j))+(nmom*nx*ny*(k))+(nmom*nx*ny*nz*(g))]

#define flux_i_len nang*nz*ny*ng
#define flux_j_len nang*nx*nz*ng
#define flux_k_len nang*nx*ny*ng
#define dd_j_len nang
#define dd_k_len nang
#define total_cross_section_len nx*ny*nz*ng
#define scat_cs_len nmom*nx*ny*nz*ng
#define denom_len nang*nx*ny*nz*ng
#define source_len cmom*nx*ny*nz*ng
#define time_delta_len ng
#define groups_todo_len ng
#define g2g_source_len cmom*nx*ny*nz*ng
#define scalar_flux_len nx*ny*nz*ng
#define flux_in_len nang*nx*ny*nz*ng*noct
#define flux_out_len nang*nx*ny*nz*ng*noct
#define l_flux_in_len nang*nx*ny*nz*ng
#define l_flux_out_len nang*nx*ny*nz*ng
#define scalar_mom_len (cmom-1)*nx*ny*nz*ng
#define mu_len nang
#define eta_len nang
#define xi_len nang
#define weights_len nang
#define velocity_len ng
#define mat_len nx*ny*nz
#define fixed_source_len nx*ny*nz*ng
#define gg_cs_len nmat*nmom*ng*ng
#define lma_len nmom
#define xs_len nmat*ng
#define scat_coeff_len cmom*nang*noct

