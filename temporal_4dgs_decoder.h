#ifndef TEMPORAL_4DGS_DECODER_H
#define TEMPORAL_4DGS_DECODER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Temporal4DGSScene
 * All float arrays are fully dequantized / reconstructed floats.
 * f16 arrays are kept as raw uint16_t (half-float) — use
 * temporal_4dgs_half_to_float() to convert.
 * -------------------------------------------------------------------------- */
typedef struct Temporal4DGSScene {
    uint32_t count;          /* N = number of Gaussians */

    /* Raw float16 blocks — size [N*3] and [N*9] */
    uint16_t *xyz_f16;       /* [N, 3] */
    uint16_t *motion_f16;    /* [N, 9] */

    /* Scalar blocks — dequantized float, size [N] each */
    float    *opacity;
    float    *tcen;
    float    *tsca;

    /* VQ-reconstructed float blocks */
    float    *scale;
    uint16_t  scale_dim;

    float    *rotation;
    uint16_t  rotation_dim;

    float    *omega;
    uint16_t  omega_dim;

    float    *tfea;
    uint16_t  tfea_dim;

    /* Optional baked features */
    uint8_t   has_features_dc;
    uint16_t *features_dc_f16;   /* [N, 6] if has_features_dc == 1 */

    /* Optional RGB decoder weights */
    uint8_t   has_rgb_dec;
    uint16_t *rgb_dec_w1_f16;    /* [6, 12]  = 72 values */
    uint16_t *rgb_dec_w2_f16;    /* [3,  6]  = 18 values */
} Temporal4DGSScene;

/* --------------------------------------------------------------------------
 * API
 * -------------------------------------------------------------------------- */

/**
 * Load a raw .4dgs file from disk and decode it into out_scene.
 * Returns 0 on success, non-zero on error.
 *
 * If TEMPORAL_4DGS_ENABLE_ZLIB is defined at compile time, the loader will
 * also transparently decompress .4dgs.gz files.
 */
int temporal_4dgs_load_file(const char *path, Temporal4DGSScene *out_scene);

/**
 * Parse an already-loaded memory buffer.
 * data must remain valid for the duration of this call.
 * Returns 0 on success, non-zero on error.
 */
int temporal_4dgs_parse_memory(const uint8_t *data, size_t size,
                               Temporal4DGSScene *out_scene);

/**
 * Free all heap memory owned by scene.
 * Does NOT free the scene struct itself.
 */
void temporal_4dgs_free_scene(Temporal4DGSScene *scene);

/**
 * Convert an IEEE 754 half-float (binary16) to a 32-bit float.
 */
float temporal_4dgs_half_to_float(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif /* TEMPORAL_4DGS_DECODER_H */