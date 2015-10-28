#include "ext_sweep.hpp"

// Compute the order of the sweep for the first octant
void compute_sweep_order(int** num_cells, float** cells)
{
    unsigned int nplanes = ichunk + ny + nz - 2;
    *num_cells = (int*)malloc(nplanes*sizeof(int));
    *cells = (int*)malloc(3*nz*ny*ichunk*sizeof(int));
    int* tmp_indices = (int*)malloc(nplanes*sizeof(int));

    for(int ii = 0; ii < nplanes; ++ii)
    {
        (*num_cells)[ii] = 0;
        tmp_indices[ii] = 0;
    }

    // Cells on each plane have equal co-ordinate sum
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;
                (*num_cells)[n]++;
            }
        }
    }

    // Store the cell indexes in the plane array
    for (unsigned int k = 0; k < nz; k++)
    {
        for (unsigned int j = 0; j < ny; j++)
        {
            for (unsigned int i = 0; i < ichunk; i++)
            {
                unsigned int n = i + j + k;

                unsigned int offset = 0;
                for(int l = 0; l < n; ++l)
                {
                    offset += (*num_cells)[l];
                }

                unsigned int ind = tmp_indices[n];
                (*cells)[offset + ind] = i;
                (*cells)[offset + ind+1] = j;
                (*cells)[offset + ind+2] = k;
                tmp_indices[n]++;
            }
        }
    }

    free(tmp_indices);
}

// Sweep over the grid and compute the angular flux
void sweep_octant(
        const unsigned int timestep, 
        const unsigned int oct, 
        const unsigned int ndiag, 
        const float* cells,
        const int* num_cells,
        const unsigned int num_groups_todo)
{
    // Determine the cell step parameters for the given octant
    // Create the list of octant co-ordinates in order

    // This first bit string assumes 3 reflective boundaries
    //int order_3d = 0b000001010100110101011111;

    // This bit string is lexiographically organised
    // This is the order to match the original SNAP
    // However this required all vacuum boundaries
    int order_3d = 0b000001010011100101110111;

    int order_2d = 0b11100100;

    // Use the bit mask to get the right values for starting positions of the sweep
    int xhi = ((order_3d >> (oct * 3)) & 1) ? nx : 0;
    int yhi = ((order_3d >> (oct * 3 + 1)) & 1) ? ny : 0;
    int zhi = ((order_3d >> (oct * 3 + 2)) & 1) ? nz : 0;

    // Set the order you traverse each axis
    int istep = (xhi == nx) ? -1 : 1;
    int jstep = (yhi == ny) ? -1 : 1;
    int kstep = (zhi == nz) ? -1 : 1;

    size_t offset = oct*nang*nx*ny*nz*ng;
    double* l_flux_in = (timestep % 2 == 0 ? flux_in : flux_out) + offset;
    double* l_flux_out = (timestep % 2 == 0 ? flux_out : flux_in) + offset;

    int cells_processed = 0;
    for (unsigned int d = 0; d < ndiag; d++)
    {
        int ncells = num_cells[d];
        int n = ncells*num_groups_todo*nang;
        sweep_cell(istep, jstep, kstep, oct, l_flux_in, l_flux_out,
                &(cells[cells_processed]), groups_todo, num_groups_todo, ncells);
        cells_processed += ncells;
    }
}

// Perform a sweep over the grid for all the octants
void perform_sweep(
        unsigned int num_groups_todo)
{
    // Number of planes in this octant
    unsigned int ndiag = ichunk + ny + nz - 2;

    START_PROFILING;

    // Get the order of cells to enqueue
    float* cells;
    int* num_cells;
    compute_sweep_order(&num_cells, &cells);

    for (int o = 0; o < noct; o++)
    {
        sweep_octant(global_timestep, o, ndiag, cells, num_cells, num_groups_todo);
        zero_edge_flux_buffers();
    }

    free(cells);
    free(num_cells);

    STOP_PROFILING(__func__, true);
}


