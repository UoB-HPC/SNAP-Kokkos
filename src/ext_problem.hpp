#pragma once

// This file contains a list of the global problem variables
// such as grid size, angles, energy groups, etc.
int nx;
int ny;
int nz;
int ng;
int nang;
int noct;
int cmom;
int nmom;
int nmat;

int ichunk;
int timesteps;

double dt;
double dx;
double dy;
double dz;

int outers;
int inners;

double epsi;
double tolr;

// Data
double* source;
double* flux_in;
double* flux_out;
double* restrict flux_i;
double* restrict flux_j;
double* restrict flux_k;
double* denom;
double dd_i;
double* dd_j;
double* dd_k;
double* mu;
double* eta;
double* xi;
double* scat_coeff;
double* time_delta;
double* total_cross_section;
double* weights;
double* velocity;
double* scalar_flux;
double* xs;
int* mat;
double* fixed_source;
double* gg_cs;
int* lma;
double* g2g_source;
double* scalar_mom;
double* scat_cs;
unsigned int* groups_todo;
double* old_outer_scalar;
double* old_inner_scalar;
double* new_scalar;
     
// Global variable for the timestep
unsigned int global_timestep;
