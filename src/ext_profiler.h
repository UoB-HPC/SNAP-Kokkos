#include <stdbool.h>

/*
 *		MANUAL PROFILING FOR CUSTOM SCOPE
 *		Not thread safe.
 */

// Compile-time optimised interface
#ifdef ENABLE_PROFILING

    #define _PROFILER_MAX_NAME 1024
    #define START_PROFILING _profiler_start_timer()
    #define STOP_PROFILING(name, tot) _profiler_end_timer(name, tot)
    #define PRINT_PROFILING_RESULTS _profiler_print_results()
    
    typedef struct 
    {
        int calls;
        double time;
        bool count_for_total;
        char name[_PROFILER_MAX_NAME];
    } profile;
    
#pragma omp declare target
    // Internal methods
    void _profiler_start_timer();
    void _profiler_end_timer(const char* kernel_name, bool count_for_total);
    void _profiler_print_results();
#pragma omp end declare target

#else

    #define START_PROFILING ; 
    #define STOP_PROFILING(name, tot) ; 
    #define PRINT_PROFILING_RESULTS ;

#endif

