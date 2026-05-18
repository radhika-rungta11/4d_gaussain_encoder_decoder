#ifndef TDGS_TO_SCENE_H
#define TDGS_TO_SCENE_H

#include <stdint.h>
#include <stddef.h>
#include "temporal_4dgs_decoder.h"
#include "splat.h"   /* PackedSplat, BoundingBox, splat_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Matches the typedef in scene.c */
typedef void (*platform_fetch_func_notifier)(void);

/**
 * Low-level conversion.
 * Converts a decoded Temporal4DGSScene into a PackedSplat array.
 *
 * @param scene      Decoded scene from temporal_4dgs_parse_memory()
 * @param out_bounds Filled with the world-space AABB of the output splats
 * @param time_t     Temporal sample point in [0,1] — 0.5 = scene midpoint
 * @return           Heap-allocated PackedSplat[scene->count], or NULL on error.
 *                   Caller must free().
 */
PackedSplat *convert_4dgs_to_packed_splats(const Temporal4DGSScene *scene,
                                            BoundingBox *out_bounds,
                                            float time_t);

/**
 * Must be called before registering on_4dgs_fetched as a fetch callback.
 * Pass a pointer to the splat_t that lives inside g_scene_state.
 */
void set_4dgs_splat_target(splat_t *splat);

/**
 * Optional: called when the splat data is fully loaded and ready to render.
 * Useful for enabling the pipeline only after data is available.
 */
void set_4dgs_ready_callback(platform_fetch_func_notifier cb);

/**
 * Fetch callback — same signature as on_spz_fetched.
 * Register with: g_fetch_fn("spz/ours_cook_spinach.4dgs", on_4dgs_fetched);
 */
void on_4dgs_fetched(const uint8_t *data, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* TDGS_TO_SCENE_H */