#include "scene.h"
#include "camera.h"
#include "utils/logger.h"
#include "utils/spz_loader.h"
#include "utils/splat_texture.h"  
#include <assert.h>
#include <memory.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include "splat.h"
#include "gltf.h"
#include "sokol/sokol_time.h"
#include "4dgs_to_scene.h"         /* <-- NEW */
#include "temporal_4dgs_decoder.h" /* <-- NEW */

#ifdef _OPENMP
#include <omp.h>
#endif

static platform_fetch_func g_fetch_fn = NULL;

static struct {
    bool initialized;
    splat_t splat;
    gltf_t gltf;
    Camera *g_camera;
    double cpu_frame_time_ms;
} g_scene_state = {0};

// Platform-abstracted file loading
void scene_set_fetch_func(platform_fetch_func fn) { g_fetch_fn = fn; }

// Forward declarations
static void on_4dgs_fetched(const uint8_t *data, size_t size);
static void on_gltf_fetched(const uint8_t *data, size_t size);

void load_assets() {
    if (!g_fetch_fn) {
        error("No platform fetch function set. Call scene_set_fetch_func() "
              "first.");
        return;
    }

    // Load the 4DGS scene file
    g_fetch_fn("spz/ours_cook_spinach.4dgs", on_4dgs_fetched);

    // Load GLTF model (keep or comment out if you only want splats)
    g_fetch_fn("gltf/Fur_test.glb", on_gltf_fetched);
}

int init_scene(void) {
    if (g_scene_state.initialized) {
        return 1;
    }

    // Initialize sokol_time for frame timing
    stm_setup();

    // Initialize camera
    g_scene_state.g_camera = camera_create();
    camera_set_radius(g_scene_state.g_camera, 10.0f);
    if (!g_scene_state.g_camera) {
        error("Failed to create camera");
        return 0;
    }

    load_assets();

    // Tell the 4dgs converter where to write the packed splat data
    set_4dgs_splat_target(&g_scene_state.splat);

    // Setup pipelines
    initialize_gltf_pipeline(&g_scene_state.gltf, g_scene_state.g_camera);
    initialize_splat_pipeline(&g_scene_state.splat, g_scene_state.g_camera);

    g_scene_state.initialized = true;

    return 1;
}

void render_scene(sg_swapchain swapchain) {
    // Frame stats record start
    uint64_t cpu_start = stm_now();

    // Render pipelines
    execute_gltf_pipeline(swapchain);
    execute_splat_pipeline(swapchain);

    // Frame stats record end
    uint64_t cpu_end = stm_now();
    g_scene_state.cpu_frame_time_ms = stm_ms(cpu_end - cpu_start);
}

void handle_input(float x, float y) {
    if (g_scene_state.g_camera) {
        camera_handle_input(g_scene_state.g_camera, x, y);
        g_scene_state.g_camera->is_camera_changed = true;
    }
}

void handle_touch_down(float x, float y) {
    if (g_scene_state.g_camera) {
        camera_reset_touch_state(g_scene_state.g_camera);
        camera_handle_input(g_scene_state.g_camera, x, y);
        g_scene_state.g_camera->is_camera_changed = true;
    }
}

void handle_pinch(float factor) {
    if (g_scene_state.g_camera) {
        camera_handle_pinch(g_scene_state.g_camera, factor);
        g_scene_state.g_camera->is_camera_changed = true;
    }
}

void handle_touch_up(void) {
    if (g_scene_state.g_camera) {
        camera_reset_touch_state(g_scene_state.g_camera);
        g_scene_state.g_camera->is_camera_changed = true;
    }
}

// Mark uniforms as dirty when splat data changes
void mark_uniforms_dirty(void) { g_scene_state.splat.uniforms_dirty = true; }

// Frame timing access functions
double get_cpu_frame_time_ms(void) { return g_scene_state.cpu_frame_time_ms; }

bool is_scene_initialized(void) { return g_scene_state.initialized; }

void cleanup_scene(void) {
    if (!g_scene_state.initialized) {
        return;
    }

    // Clean up camera
    if (g_scene_state.g_camera) {
        camera_destroy(g_scene_state.g_camera);
        g_scene_state.g_camera = NULL;
    }

    // Clean up splat data
    if (g_scene_state.splat.g_packed_splat) {
        free(g_scene_state.splat.g_packed_splat);
        g_scene_state.splat.g_packed_splat = NULL;
    }

    // Cleanup pipelines
    cleanup_splat_pipeline();
    cleanup_gltf_pipeline();

    g_scene_state.initialized = false;
}

// ============================================================
// 4DGS fetch callback — replaces on_spz_fetched
// ============================================================
static void on_4dgs_fetched(const uint8_t *data, size_t size) {
    print("Received 4DGS data: %zu bytes. Parsing...", size);

    Temporal4DGSScene scene = {0};
    if (temporal_4dgs_parse_memory(data, size, &scene) != 0) {
        error("4DGS parse failed.");
        return;
    }

    print("Parsed %u Gaussians. Converting to PackedSplat...", scene.count);

    BoundingBox bounds = {0};

    /*
     * time_t controls which temporal frame to bake.
     * 0.5 = midpoint of the scene (safe default for a single static bake).
     * To animate, change this value each frame and re-call convert + texture.
     */
    float time_t = 0.5f;

    PackedSplat *packed =
        convert_4dgs_to_packed_splats(&scene, &bounds, time_t);

    uint32_t splat_count = scene.count; /* save before free */
    temporal_4dgs_free_scene(&scene);

    if (!packed) {
        error("4DGS → PackedSplat conversion failed.");
        return;
    }

    // Free any previous packed splat data
    if (g_scene_state.splat.g_packed_splat) {
        free(g_scene_state.splat.g_packed_splat);
        g_scene_state.splat.g_packed_splat = NULL;
    }

    /*
     * sh_degree = 0 → colour only (DC term from features_dc).
     * Upgrade to 1/2/3 when your shader supports higher SH bands.
     */
    int sh_degree = 0;

    if (create_splat_texture_from_data(&g_scene_state.splat.splat_texture,
                                       packed, splat_count, sh_degree) != 0) {
        error("create_splat_texture_from_data failed.");
        free(packed);
        return;
    }

    g_scene_state.splat.g_packed_splat     = packed;
    g_scene_state.splat.splat_count        = splat_count;
    g_scene_state.splat.splat_bounds       = bounds;
    g_scene_state.splat.splats_initialized = true;
    g_scene_state.splat.uniforms_dirty     = true;

    print("4DGS ready: %u splats uploaded to GPU.", splat_count);
}

// ============================================================
// GLTF fetch callback — unchanged from original
// ============================================================
static void on_gltf_fetched(const uint8_t *data, size_t size) {
    cgltf_options options = {0};
    cgltf_data *parsed_data = NULL;
    cgltf_result result = cgltf_parse(&options, data, size, &parsed_data);

    if (result != cgltf_result_success) {
        error("Failed to parse GLB: %d", result);
        return;
    }

    result = cgltf_load_buffers(&options, parsed_data, NULL);
    if (result != cgltf_result_success) {
        error("Failed to load GLB buffers: %d", result);
        cgltf_free(parsed_data);
        return;
    }

    g_scene_state.gltf.shaders.metallic =
        sg_make_shader(gltf_shader_desc(sg_query_backend()));

    g_scene_state.gltf.point_light =
        (light_params_t){.light_pos[0]    = 10.0f,
                         .light_pos[1]    = 10.0f,
                         .light_pos[2]    = 10.0f,
                         .light_range     = 200.0f,
                         .light_color[0]  = 1.0f,
                         .light_color[1]  = 1.5f,
                         .light_color[2]  = 2.0f,
                         .light_intensity = 700.0f};

    gltf_set_context(&g_scene_state.gltf);

    if (gltf_parse_buffers(parsed_data) == 0) {
        error("Cannot parse GLB buffers");
        cgltf_free(parsed_data);
        return;
    }

    if (gltf_parse_images(parsed_data) == 0) {
        error("Cannot parse GLB images");
        cgltf_free(parsed_data);
        return;
    }

    if (gltf_parse_materials(parsed_data) == 0) {
        error("Cannot parse GLTF materials");
        cgltf_free(parsed_data);
        return;
    }

    if (gltf_parse_meshes(parsed_data) == 0) {
        error("Cannot parse GLTF meshes");
        cgltf_free(parsed_data);
        return;
    }

    if (gltf_parse_nodes(parsed_data) == 0) {
        error("Cannot parse GLTF nodes");
        cgltf_free(parsed_data);
        return;
    }

    cgltf_free(parsed_data);
}
