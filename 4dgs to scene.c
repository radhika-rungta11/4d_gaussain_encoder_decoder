/*
 * 4dgs_to_scene.c
 *
 * Converts a decoded Temporal4DGSScene into PackedSplat[] and feeds
 * it into the existing splat render pipeline (scene.c / splat.c).
 *
 * Drop this file next to scene.c and add it to your build.
 * Then in scene.c, replace on_spz_fetched calls with on_4dgs_fetched.
 *
 * Integration checklist (see bottom of this file for exact diffs):
 *   1. #include "4dgs_to_scene.h" in scene.c
 *   2. Register: g_fetch_fn("spz/ours_cook_spinach.4dgs", on_4dgs_fetched);
 *   3. Uncomment initialize_splat_pipeline / execute_splat_pipeline in scene.c
 *   4. Add this file to your Makefile / CMakeLists / emcc command
 */

#include "4dgs_to_scene.h"
#include "temporal_4dgs_decoder.h"
#include "splat.h"               /* PackedSplat, BoundingBox, splat_t        */
#include "utils/splat_texture.h" /* create_splat_texture_from_data           */
#include "utils/logger.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* ============================================================
 * Half-float helpers (reuse the decoder's converter)
 * ============================================================ */
static inline float h2f(uint16_t h) {
    return temporal_4dgs_half_to_float(h);
}

/* ============================================================
 * Normalise a float value into a uint8 [0..255]
 * ============================================================ */
static inline uint8_t norm_u8(float v, float lo, float hi) {
    if (hi <= lo) return 0;
    float t = (v - lo) / (hi - lo);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return (uint8_t)(t * 255.0f + 0.5f);
}

/* ============================================================
 * Normalise a float value into a uint16 [0..65535]
 * ============================================================ */
static inline uint16_t norm_u16(float v, float lo, float hi) {
    if (hi <= lo) return 0;
    float t = (v - lo) / (hi - lo);
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return (uint16_t)(t * 65535.0f + 0.5f);
}

/* ============================================================
 * Encode a unit quaternion component [-1..1] → uint8 [0..255]
 * ============================================================ */
static inline uint8_t quat_to_u8(float v) {
    /* map [-1,1] → [0,255] */
    float t = (v + 1.0f) * 0.5f;
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;
    return (uint8_t)(t * 255.0f + 0.5f);
}

/* ============================================================
 * Rotation: 4DGS stores a quaternion [qx, qy, qz, qw] per Gaussian.
 *
 * PackedSplat uses an octahedron-like 3-component encoding:
 *   rot_axis_u, rot_axis_v   (uint8, the dominant-axis dropped component)
 *   rot_angle                (uint8, rotation angle)
 *
 * Simplest lossless-ish approach: encode qx,qy,qz into u,v and store qw
 * magnitude as angle. The shader must mirror this decoding.
 *
 * If your shader decodes rotation differently, adjust here to match.
 * The encoding below stores:
 *   rot_axis_u = quat_to_u8(q.x)
 *   rot_axis_v = quat_to_u8(q.y)
 *   rot_angle  = quat_to_u8(q.z)
 * and the shader reconstructs qw = sqrt(1 - qx²-qy²-qz²) with correct sign
 * (qw sign is encoded in bit 7 of rot_angle — see comment below).
 * ============================================================ */
static void encode_rotation(const float *quat, /* [4]: qx qy qz qw */
                             uint8_t *out_u, uint8_t *out_v,
                             uint8_t *out_angle) {
    float qx = quat[0], qy = quat[1], qz = quat[2], qw = quat[3];

    /* Normalise to be safe */
    float len = sqrtf(qx*qx + qy*qy + qz*qz + qw*qw);
    if (len > 1e-6f) { qx/=len; qy/=len; qz/=len; qw/=len; }

    /*
     * Encode xyz into u,v,angle bytes.
     * We use a "smallest 3" scheme: drop the largest component,
     * encode the other three packed into 8-bit fields.
     * The dropped component's index and sign are recovered from the constraint
     * qx²+qy²+qz²+qw² = 1.
     *
     * For simplicity here (matching most existing Gaussian splat shaders that
     * expect a packed (u,v,angle) triplet decoded as axis-angle):
     *   axis   = normalize(qx, qy, qz)
     *   angle  = 2*acos(|qw|)   in [0, pi]
     *
     * u = axis.x in [-1,1] → [0,255]
     * v = axis.y in [-1,1] → [0,255]
     * angle_u8 encodes the angle/pi in [0,255]
     * The sign of qw is encoded in bit0 of angle_u8 (0=positive, 1=negative).
     */
    float axis_len = sqrtf(qx*qx + qy*qy + qz*qz);
    float ax = 0.0f, ay = 0.0f;
    float angle_norm = 0.0f;

    if (axis_len > 1e-6f) {
        ax = qx / axis_len;
        ay = qy / axis_len;
        /* angle in [0,pi], normalised to [0,1] */
        float angle_rad = 2.0f * acosf(fabsf(qw) < 1.0f ? fabsf(qw) : 1.0f);
        angle_norm = angle_rad / (float)M_PI;
    }

    *out_u = quat_to_u8(ax);
    *out_v = quat_to_u8(ay);

    uint8_t angle_byte = (uint8_t)(angle_norm * 254.0f + 0.5f); /* 0-254 */
    /* store sign of qw in bit 7 */
    if (qw < 0.0f) angle_byte |= 0x80;
    *out_angle = angle_byte;
}

/* ============================================================
 * RGB from features_dc (SH band-0 / DC term).
 *
 * features_dc is float16 [N, 6]:
 *   [0..2] = DC SH coefficients for R, G, B  (scene.c / .spz convention)
 *   [3..5] = second set (e.g. from another SH band or ignored)
 *
 * Gaussian splat DC → linear RGB:
 *   C = 0.5 + SH_C0 * dc    where SH_C0 = 0.28209479177f
 * Then clamp to [0,1] and encode to uint8.
 * ============================================================ */
#define SH_C0 0.28209479177f

static inline uint8_t sh_dc_to_u8(float sh) {
    float c = 0.5f + SH_C0 * sh;
    if (c < 0.0f) c = 0.0f;
    if (c > 1.0f) c = 1.0f;
    return (uint8_t)(c * 255.0f + 0.5f);
}

/* ============================================================
 * Sigmoid — used to convert raw opacity logit → probability
 * ============================================================ */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* ============================================================
 * 4DGS → PackedSplat main conversion
 *
 * Returns a heap-allocated PackedSplat array (caller frees) and
 * fills out_bounds with the world-space bounding box.
 * Returns NULL on failure.
 * ============================================================ */
PackedSplat *convert_4dgs_to_packed_splats(const Temporal4DGSScene *scene,
                                            BoundingBox *out_bounds,
                                            float time_t) {
    if (!scene || !out_bounds) return NULL;
    uint32_t N = scene->count;
    if (N == 0) return NULL;

    /* --------------------------------------------------------
     * 1. Decode all xyz positions to float32 and compute bounds
     * -------------------------------------------------------- */
    float *pos_x = (float *)malloc(N * sizeof(float));
    float *pos_y = (float *)malloc(N * sizeof(float));
    float *pos_z = (float *)malloc(N * sizeof(float));
    if (!pos_x || !pos_y || !pos_z) {
        free(pos_x); free(pos_y); free(pos_z);
        return NULL;
    }

    float bmin_x =  FLT_MAX, bmin_y =  FLT_MAX, bmin_z =  FLT_MAX;
    float bmax_x = -FLT_MAX, bmax_y = -FLT_MAX, bmax_z = -FLT_MAX;

    for (uint32_t i = 0; i < N; i++) {
        /* xyz_f16 is [N,3], row-major */
        float x = h2f(scene->xyz_f16[i*3 + 0]);
        float y = h2f(scene->xyz_f16[i*3 + 1]);
        float z = h2f(scene->xyz_f16[i*3 + 2]);

        /*
         * Apply 4D motion at time t.
         * motion_f16 is [N, 9]: first 3 = velocity (dx,dy,dz),
         * next 6 are higher-order terms (ignored at linear level).
         * 4DGS: pos(t) = xyz + motion[0..2] * (t - tcen)
         */
        if (scene->motion_f16 && scene->tcen) {
            float tcen_i = scene->tcen[i];
            float dt = time_t - tcen_i;
            float vx = h2f(scene->motion_f16[i*9 + 0]);
            float vy = h2f(scene->motion_f16[i*9 + 1]);
            float vz = h2f(scene->motion_f16[i*9 + 2]);
            x += vx * dt;
            y += vy * dt;
            z += vz * dt;
        }

        pos_x[i] = x; pos_y[i] = y; pos_z[i] = z;

        if (x < bmin_x) bmin_x = x; if (x > bmax_x) bmax_x = x;
        if (y < bmin_y) bmin_y = y; if (y > bmax_y) bmax_y = y;
        if (z < bmin_z) bmin_z = z; if (z > bmax_z) bmax_z = z;
    }

    /* Add a small epsilon to avoid degenerate box */
    float eps = 1e-4f;
    out_bounds->min.X = bmin_x - eps; out_bounds->max.X = bmax_x + eps;
    out_bounds->min.Y = bmin_y - eps; out_bounds->max.Y = bmax_y + eps;
    out_bounds->min.Z = bmin_z - eps; out_bounds->max.Z = bmax_z + eps;

    /* --------------------------------------------------------
     * 2. Compute scale range for normalisation (log-scale → exp)
     * 4DGS stores log-scale; PackedSplat uses [0..255] per axis.
     * We normalise within the observed range of exp(scale).
     * -------------------------------------------------------- */
    float smin =  FLT_MAX, smax = -FLT_MAX;
    if (scene->scale) {
        uint16_t sdim = scene->scale_dim; /* typically 3 */
        for (uint32_t i = 0; i < N; i++) {
            for (uint16_t d = 0; d < sdim && d < 3; d++) {
                float sv = expf(scene->scale[i*sdim + d]);
                if (sv < smin) smin = sv;
                if (sv > smax) smax = sv;
            }
        }
    }
    if (smax <= smin) { smin = 0.0f; smax = 1.0f; }

    /* --------------------------------------------------------
     * 3. Allocate output
     * -------------------------------------------------------- */
    PackedSplat *packed = (PackedSplat *)calloc(N, sizeof(PackedSplat));
    if (!packed) { free(pos_x); free(pos_y); free(pos_z); return NULL; }

    float bsz_x = bmax_x - bmin_x + 2*eps;
    float bsz_y = bmax_y - bmin_y + 2*eps;
    float bsz_z = bmax_z - bmin_z + 2*eps;

    for (uint32_t i = 0; i < N; i++) {
        PackedSplat *p = &packed[i];

        /* -- Position (uint16, normalised to bounding box) -- */
        p->pos_x = norm_u16(pos_x[i], bmin_x - eps, bmax_x + eps);
        p->pos_y = norm_u16(pos_y[i], bmin_y - eps, bmax_y + eps);
        p->pos_z = norm_u16(pos_z[i], bmin_z - eps, bmax_z + eps);

        /* -- Scale (uint8, normalised to [smin, smax]) -- */
        if (scene->scale) {
            uint16_t sdim = scene->scale_dim;
            float sx = (sdim > 0) ? expf(scene->scale[i*sdim + 0]) : 0.0f;
            float sy = (sdim > 1) ? expf(scene->scale[i*sdim + 1]) : 0.0f;
            float sz = (sdim > 2) ? expf(scene->scale[i*sdim + 2]) : 0.0f;
            p->scale_x = norm_u8(sx, smin, smax);
            p->scale_y = norm_u8(sy, smin, smax);
            p->scale_z = norm_u8(sz, smin, smax);
        }

        /* -- Rotation (quaternion → axis-angle bytes) -- */
        if (scene->rotation && scene->rotation_dim >= 4) {
            float *q = &scene->rotation[i * scene->rotation_dim];
            encode_rotation(q, &p->rot_axis_u, &p->rot_axis_v, &p->rot_angle);
        }

        /* -- Opacity (sigmoid of raw logit → uint8) -- */
        if (scene->opacity) {
            float op = sigmoid(scene->opacity[i]);
            p->a = (uint8_t)(op * 255.0f + 0.5f);
        } else {
            p->a = 255;
        }

        /* -- Colour from features_dc (SH band-0) -- */
        if (scene->has_features_dc && scene->features_dc_f16) {
            float dc_r = h2f(scene->features_dc_f16[i*6 + 0]);
            float dc_g = h2f(scene->features_dc_f16[i*6 + 1]);
            float dc_b = h2f(scene->features_dc_f16[i*6 + 2]);
            p->r = sh_dc_to_u8(dc_r);
            p->g = sh_dc_to_u8(dc_g);
            p->b = sh_dc_to_u8(dc_b);
        } else {
            /* Fallback: mid-grey */
            p->r = p->g = p->b = 128;
        }

        /* -- SH coefficients (degree 0 only for now, no extra coeffs) -- */
        /* sh_coeffs[] stays zero — the DC colour is in r,g,b above.
         * If your shader supports SH degree > 0, pack scene->tfea here. */
        memset(p->sh_coeffs, 0, sizeof(p->sh_coeffs));
    }

    free(pos_x); free(pos_y); free(pos_z);
    return packed;
}

/* ============================================================
 * on_4dgs_fetched — drop-in replacement for on_spz_fetched in scene.c
 *
 * To wire this in, add to scene.c:
 *
 *   #include "4dgs_to_scene.h"
 *
 * and change load_assets() to:
 *
 *   g_fetch_fn("spz/ours_cook_spinach.4dgs", on_4dgs_fetched);
 *
 * and uncomment:
 *   initialize_splat_pipeline(&g_scene_state.splat, g_scene_state.g_camera);
 *   execute_splat_pipeline(swapchain);
 * ============================================================ */

/*
 * g_scene_splat_ptr must be set before on_4dgs_fetched is called.
 * In scene.c, set it to &g_scene_state.splat during init_scene().
 */
static splat_t         *g_splat_target   = NULL;
static platform_fetch_func_notifier g_on_ready = NULL; /* optional callback */

void set_4dgs_splat_target(splat_t *splat) {
    g_splat_target = splat;
}

void set_4dgs_ready_callback(platform_fetch_func_notifier cb) {
    g_on_ready = cb;
}

void on_4dgs_fetched(const uint8_t *data, size_t size) {
    if (!g_splat_target) {
        error("[4dgs] set_4dgs_splat_target() was not called before fetch");
        return;
    }

    print("[4dgs] Received %zu bytes, parsing...", size);

    Temporal4DGSScene scene = {0};
    if (temporal_4dgs_parse_memory(data, size, &scene) != 0) {
        error("[4dgs] Parse failed");
        return;
    }

    print("[4dgs] Parsed %u Gaussians. Converting to PackedSplat...",
          scene.count);

    BoundingBox bounds = {0};
    /*
     * time_t = 0.5 renders the scene at the midpoint of its temporal range.
     * Change this to animate: pass a value in [0,1] driven by your frame timer.
     */
    float time_t = 0.5f;

    PackedSplat *packed =
        convert_4dgs_to_packed_splats(&scene, &bounds, time_t);
    temporal_4dgs_free_scene(&scene);

    if (!packed) {
        error("[4dgs] Conversion to PackedSplat failed");
        return;
    }

    /* Free any previous splat data */
    if (g_splat_target->g_packed_splat) {
        free(g_splat_target->g_packed_splat);
        g_splat_target->g_packed_splat = NULL;
    }

    /* sh_degree=0: colour only (no SH beyond DC). Upgrade when needed. */
    int sh_degree = 0;
    if (create_splat_texture_from_data(&g_splat_target->splat_texture,
                                       packed, scene.count,
                                       sh_degree) != 0) {
        error("[4dgs] create_splat_texture_from_data failed");
        free(packed);
        return;
    }

    g_splat_target->g_packed_splat    = packed;
    g_splat_target->splat_count       = scene.count;
    g_splat_target->splat_bounds      = bounds;
    g_splat_target->splats_initialized = true;
    g_splat_target->uniforms_dirty    = true;

    print("[4dgs] Ready: %u splats loaded into GPU texture", scene.count);

    if (g_on_ready) g_on_ready();
}