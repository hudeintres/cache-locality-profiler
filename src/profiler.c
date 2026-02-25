#include "profiler.h"
#include <math.h>

void profiler_init(Profiler* p) {
    p->count = 0;
    for (int i = 0; i < MAX_PROFILE_POINTS; i++) {
        p->points[i].active = 0;
        p->points[i].elapsed_ms = 0.0;
    }
}

void profiler_start(Profiler* p, const char* name) {
    // Find existing or create new profile point
    int idx = -1;
    for (int i = 0; i < p->count; i++) {
        if (strcmp(p->points[i].name, name) == 0) {
            idx = i;
            break;
        }
    }
    
    if (idx == -1) {
        if (p->count >= MAX_PROFILE_POINTS) {
            fprintf(stderr, "Error: Too many profile points\n");
            return;
        }
        idx = p->count++;
        strncpy(p->points[idx].name, name, MAX_NAME_LEN - 1);
        p->points[idx].name[MAX_NAME_LEN - 1] = '\0';
    }
    
    clock_gettime(CLOCK_MONOTONIC, &p->points[idx].start_time);
    p->points[idx].active = 1;
}

void profiler_end(Profiler* p, const char* name) {
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    for (int i = 0; i < p->count; i++) {
        if (strcmp(p->points[i].name, name) == 0 && p->points[i].active) {
            p->points[i].end_time = end_time;
            
            double start_ms = p->points[i].start_time.tv_sec * 1000.0 + 
                             p->points[i].start_time.tv_nsec / 1000000.0;
            double end_ms = end_time.tv_sec * 1000.0 + 
                           end_time.tv_nsec / 1000000.0;
            
            p->points[i].elapsed_ms += (end_ms - start_ms);
            p->points[i].active = 0;
            return;
        }
    }
    
    fprintf(stderr, "Warning: No active profile point named '%s'\n", name);
}

void profiler_print_results(Profiler* p) {
    printf("\n========================================\n");
    printf("         PROFILING RESULTS              \n");
    printf("========================================\n");
    printf("%-30s %15s\n", "Section", "Time (ms)");
    printf("----------------------------------------\n");
    
    double total = 0.0;
    for (int i = 0; i < p->count; i++) {
        printf("%-30s %15.4f\n", p->points[i].name, p->points[i].elapsed_ms);
        total += p->points[i].elapsed_ms;
    }
    
    printf("----------------------------------------\n");
    printf("%-30s %15.4f\n", "TOTAL", total);
    printf("========================================\n");
}

void profiler_save_results(Profiler* p, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return;
    }
    
    fprintf(fp, "section,time_ms\n");
    for (int i = 0; i < p->count; i++) {
        fprintf(fp, "%s,%.6f\n", p->points[i].name, p->points[i].elapsed_ms);
    }
    
    fclose(fp);
    printf("Results saved to %s\n", filename);
}

double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}
