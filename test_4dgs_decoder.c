/*
 * test_4dgs_decoder.c
 *
 * Usage:
 *   ./test_4dgs_decoder ours_cook_spinach.4dgs
 *
 * Compile:
 *   gcc -std=c99 -O2 temporal_4dgs_decoder.c test_4dgs_decoder.c -o test_4dgs_decoder -lm
 */

#include "temporal_4dgs_decoder.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.4dgs>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    printf("=== Loading: %s ===\n\n", path);

    Temporal4DGSScene scene = {0};
    int ret = temporal_4dgs_load_file(path, &scene);
    if (ret != 0) {
        fprintf(stderr, "\nFailed to load scene (error %d)\n", ret);
        return 1;
    }

    printf("\n=== Scene Summary ===\n");
    printf("  Gaussian count   : %u\n",   scene.count);
    printf("  scale_dim        : %u\n",   scene.scale_dim);
    printf("  rotation_dim     : %u\n",   scene.rotation_dim);
    printf("  omega_dim        : %u\n",   scene.omega_dim);
    printf("  tfea_dim         : %u\n",   scene.tfea_dim);
    printf("  has_features_dc  : %u\n",   scene.has_features_dc);
    printf("  has_rgb_dec      : %u\n",   scene.has_rgb_dec);

    /* First Gaussian — xyz */
    if (scene.xyz_f16 && scene.count > 0) {
        float x = temporal_4dgs_half_to_float(scene.xyz_f16[0]);
        float y = temporal_4dgs_half_to_float(scene.xyz_f16[1]);
        float z = temporal_4dgs_half_to_float(scene.xyz_f16[2]);
        printf("\n=== First Gaussian ===\n");
        printf("  xyz              : (%.6f, %.6f, %.6f)\n", x, y, z);
    }

    /* First opacity */
    if (scene.opacity) {
        printf("  opacity          : %.6f\n", scene.opacity[0]);
    }

    /* First scale vector */
    if (scene.scale && scene.scale_dim > 0) {
        printf("  scale            : (");
        for (uint16_t d = 0; d < scene.scale_dim; d++) {
            printf("%.6f%s", scene.scale[d],
                   d < (uint16_t)(scene.scale_dim - 1) ? ", " : "");
        }
        printf(")\n");
    }

    /* First rotation vector */
    if (scene.rotation && scene.rotation_dim > 0) {
        printf("  rotation         : (");
        for (uint16_t d = 0; d < scene.rotation_dim; d++) {
            printf("%.6f%s", scene.rotation[d],
                   d < (uint16_t)(scene.rotation_dim - 1) ? ", " : "");
        }
        printf(")\n");
    }

    /* First tcen / tsca */
    if (scene.tcen)  printf("  tcen[0]          : %.6f\n", scene.tcen[0]);
    if (scene.tsca)  printf("  tsca[0]          : %.6f\n", scene.tsca[0]);

    /* features_dc first 6 values */
    if (scene.has_features_dc && scene.features_dc_f16) {
        printf("  features_dc[0]   : (");
        for (int d = 0; d < 6; d++) {
            printf("%.4f%s",
                   temporal_4dgs_half_to_float(scene.features_dc_f16[d]),
                   d < 5 ? ", " : "");
        }
        printf(")\n");
    }

    printf("\n=== All OK ===\n");

    temporal_4dgs_free_scene(&scene);
    return 0;
}