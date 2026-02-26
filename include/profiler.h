#ifndef PROFILER_H
#define PROFILER_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_PROFILE_POINTS 100
#define MAX_NAME_LEN 64

typedef struct {
    char name[MAX_NAME_LEN];
    struct timespec start_time;
    struct timespec end_time;
    double elapsed_ms;
    int active;
} ProfilePoint;

typedef struct {
    ProfilePoint points[MAX_PROFILE_POINTS];
    int count;
} Profiler;

// Initialize the profiler
void profiler_init(Profiler* p);

// Start timing a named section
void profiler_start(Profiler* p, const char* name);

// End timing a named section
void profiler_end(Profiler* p, const char* name);

// Print all profiling results
void profiler_print_results(Profiler* p);

// Save results to file
void profiler_save_results(Profiler* p, const char* filename);

// Get current time in milliseconds
double get_time_ms(void);

#endif // PROFILER_H
