#pragma once

#include <stdbool.h>

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
		double *old, 
		double *new, 
		double epsi, 
		unsigned int *groups_todo, 
		unsigned int *num_groups_todo, 
		bool inner);
