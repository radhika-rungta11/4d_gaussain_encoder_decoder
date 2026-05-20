#include "splat.h"
#include "camera.h"
#include "utils/radix_sort.h" 
#include "utils/logger.h"
#include "sokol/sokol_time.h"
#include "utils/splat_texture.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

splat_t *m_splat = NULL;
static Camera *g_camera = NULL;

// Temp buffer for radix sort (double buffering)
static DepthIndexPair *g_radix_temp = NULL;

void Calculate_Depth() {
    HMM_Vec3 camera_forward;
    const HMM_Vec3 *camera_pos = &g_camera->position;
    HMM_Mat4 view = camera_get_view_matrix(g_camera);
    camera_forward = HMM_V3(-view.Elements[0][2], -view.Elements[1][2],
                            -view.Elements[2][2]);
    camera_forward = HMM_NormV3(camera_forward);

    const float inv_65535 = 1.0f / 65535.0f;

    HMM_Vec3 bounds_min = m_splat->splat_bounds.min;
    HMM_Vec3 bounds_size =
        HMM_Sub(m_splat->splat_bounds.max, m_splat->splat_bounds.min);

    for (int i = 0; i < m_splat->splat_count; i++) {
        PackedSplat local_splat = m_splat->g_packed_splat[i];

        HMM_Vec3 norm_pos;
        norm_pos.X = (float)local_splat.pos_x * inv_65535;
        norm_pos.Y = (float)local_splat.pos_y * inv_65535;
        norm_pos.Z = (float)local_splat.pos_z * inv_65535;

        // Denormalize to world space using bounding box
        HMM_Vec3 splat_pos;
        splat_pos.X = bounds_min.X + norm_pos.X * bounds_size.X;
        splat_pos.Y = bounds_min.Y + norm_pos.Y * bounds_size.Y;
        splat_pos.Z = bounds_min.Z + norm_pos.Z * bounds_size.Z;

        // Apply model matrix
        HMM_Vec4 splat_pos4 =
            HMM_V4(splat_pos.X, splat_pos.Y, splat_pos.Z, 1.0f);
        HMM_Vec4 world_pos4 = HMM_MulM4V4(m_splat->model, splat_pos4);
        HMM_Vec3 world_pos = HMM_V3(world_pos4.X, world_pos4.Y, world_pos4.Z);

        HMM_Vec3 to_splat = HMM_SubV3(world_pos, *camera_pos);
        float depth_val = HMM_Dot(to_splat, camera_forward);

        // Store directly into the depth-index pair for radix sort
        m_splat->m_depthIndexPair[i].depth = depth_val;
        m_splat->m_depthIndexPair[i].index = (uint32_t)i;
    }
}

void Sort_Splat() {
    if (!m_splat) {
        return;
    }

    if (!m_splat->m_depthIndexPair || !g_radix_temp) {
        print("Failed to create depth index pair or temp buffer");
        return;
    }

    // Use parallel radix sort (descending order for back-to-front)
    radix_sort_depth_descending(m_splat->m_depthIndexPair, g_radix_temp,
                                m_splat->splat_count);

    // Extract sorted indices
    for (int i = 0; i < m_splat->splat_count; i++) {
        m_splat->splat_indices[i] = m_splat->m_depthIndexPair[i].index;
    }

    sg_update_buffer(
        m_splat->g_index_buffer,
        &(sg_range){.ptr = m_splat->splat_indices,
                    .size = m_splat->splat_count * sizeof(uint32_t)});
}

void extract_frustum_planes(HMM_Mat4 view_proj, HMM_Vec4 planes[6]) {
    // Frustum planes from view-projection matrix (Gribb-Hartmann method)

    // Left plane: row4 + row1
    planes[0] = HMM_V4(view_proj.Elements[0][3] + view_proj.Elements[0][0],
                       view_proj.Elements[1][3] + view_proj.Elements[1][0],
                       view_proj.Elements[2][3] + view_proj.Elements[2][0],
                       view_proj.Elements[3][3] + view_proj.Elements[3][0]);

    // Right plane: row4 - row1
    planes[1] = HMM_V4(view_proj.Elements[0][3] - view_proj.Elements[0][0],
                       view_proj.Elements[1][3] - view_proj.Elements[1][0],
                       view_proj.Elements[2][3] - view_proj.Elements[2][0],
                       view_proj.Elements[3][3] - view_proj.Elements[3][0]);

    // Bottom plane: row4 + row2
    planes[2] = HMM_V4(view_proj.Elements[0][3] + view_proj.Elements[0][1],
                       view_proj.Elements[1][3] + view_proj.Elements[1][1],
                       view_proj.Elements[2][3] + view_proj.Elements[2][1],
                       view_proj.Elements[3][3] + view_proj.Elements[3][1]);

    // Top plane: row4 - row2
    planes[3] = HMM_V4(view_proj.Elements[0][3] - view_proj.Elements[0][1],
                       view_proj.Elements[1][3] - view_proj.Elements[1][1],
                       view_proj.Elements[2][3] - view_proj.Elements[2][1],
                       view_proj.Elements[3][3] - view_proj.Elements[3][1]);

    // Near plane: row4 + row3
    planes[4] = HMM_V4(view_proj.Elements[0][3] + view_proj.Elements[0][2],
                       view_proj.Elements[1][3] + view_proj.Elements[1][2],
                       view_proj.Elements[2][3] + view_proj.Elements[2][2],
                       view_proj.Elements[3][3] + view_proj.Elements[3][2]);

    // Far plane: row4 - row3
    planes[5] = HMM_V4(view_proj.Elements[0][3] - view_proj.Elements[0][2],
                       view_proj.Elements[1][3] - view_proj.Elements[1][2],
                       view_proj.Elements[2][3] - view_proj.Elements[2][2],
                       view_proj.Elements[3][3] - view_proj.Elements[3][2]);

    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float length =
            sqrtf(planes[i].X * planes[i].X + planes[i].Y * planes[i].Y +
                  planes[i].Z * planes[i].Z);
        if (length > 0.0f) {
            planes[i].X /= length;
            planes[i].Y /= length;
            planes[i].Z /= length;
            planes[i].W /= length;
        }
    }
}

void Render_Splat(sg_swapchain swapchain) {
    HMM_Mat4 view = camera_get_view_matrix(g_camera);
    float aspect_ratio = (float)swapchain.width / (float)swapchain.height;
    HMM_Mat4 projection = camera_get_projection_matrix(g_camera, aspect_ratio);

    HMM_Mat4 view_proj = HMM_MulM4(projection, view);
    HMM_Vec4 frustum_planes[6];
    extract_frustum_planes(view_proj, frustum_planes);

    for (int i = 0; i < 6; i++) {
        m_splat->vs_params.frustum_planes[i][0] = frustum_planes[i].X;
        m_splat->vs_params.frustum_planes[i][1] = frustum_planes[i].Y;
        m_splat->vs_params.frustum_planes[i][2] = frustum_planes[i].Z;
        m_splat->vs_params.frustum_planes[i][3] = frustum_planes[i].W;
    }

    memcpy(m_splat->vs_params.viewMat, &view, sizeof(float) * 16);
    memcpy(m_splat->vs_params.projMat, &projection, sizeof(float) * 16);
    memcpy(m_splat->vs_params.modelMat, &m_splat->model, sizeof(float) * 16);

    // FIXED: Calculate focal lengths correctly
    float fov_y_rad = g_camera->fov; // In radians
    float focal_y = ((float)swapchain.height * 0.5f) / tanf(fov_y_rad * 0.5f);
    float focal_x = focal_y; // Square pixels

    float fov_radians = g_camera->fov * (HMM_PI / 180.0f);
    float focal_length = 1.0f / tanf(fov_radians / 2.0f);

    // Calculate focal lengths correctl

    m_splat->vs_params.focalDistance = focal_length;
    m_splat->vs_params.focal[0] = focal_x;
    m_splat->vs_params.focal[1] = focal_y;

    m_splat->vs_params.viewport[0] = (float)swapchain.width;
    m_splat->vs_params.viewport[1] = (float)swapchain.height;

    // Splat scale (adjust to control size)
    // m_splat->vs_params.splat_scale = 1.0f;

    if (m_splat->uniforms_dirty) {
        m_splat->vs_params.bounds_min[0] = m_splat->splat_bounds.min.X;
        m_splat->vs_params.bounds_min[1] = m_splat->splat_bounds.min.Y;
        m_splat->vs_params.bounds_min[2] = m_splat->splat_bounds.min.Z;

        m_splat->vs_params.bounds_max[0] = m_splat->splat_bounds.max.X;
        m_splat->vs_params.bounds_max[1] = m_splat->splat_bounds.max.Y;
        m_splat->vs_params.bounds_max[2] = m_splat->splat_bounds.max.Z;

        HMM_Vec3 bound_size =
            HMM_Sub(m_splat->splat_bounds.max, m_splat->splat_bounds.min);

        m_splat->vs_params.bounds_size[0] = bound_size.X;
        m_splat->vs_params.bounds_size[1] = bound_size.Y;
        m_splat->vs_params.bounds_size[2] = bound_size.Z;

        m_splat->vs_params.texture_width = m_splat->splat_texture.width;
        m_splat->vs_params.texture_height = m_splat->splat_texture.height;
        m_splat->vs_params.splats_per_layer =
            m_splat->splat_texture.width * m_splat->splat_texture.height;

        m_splat->vs_params.sh_degree = m_splat->splat_texture.sh_degree;
        m_splat->vs_params.pixels_per_splat =
            get_pixels_per_splat(m_splat->splat_texture.sh_degree);

        m_splat->g_surface_bind.views[VIEW_splat_texture] =
            m_splat->splat_texture.view;
        m_splat->g_surface_bind.samplers[SMP_splat_sampler] =
            m_splat->splat_texture.sampler;

        m_splat->uniforms_dirty = false;
    }

    m_splat->vs_params.time = (float)stm_sec(stm_now());
    m_splat->g_surface_bind.vertex_buffers[1] = m_splat->g_index_buffer;

    sg_begin_pass(&(sg_pass){
        .action = m_splat->surface_pass_action,
        .swapchain = swapchain,
    });

    sg_apply_pipeline(m_splat->g_surface_pip);
    sg_apply_bindings(&m_splat->g_surface_bind);
    sg_apply_uniforms(UB_splat_vs_params, &SG_RANGE(m_splat->vs_params));
    sg_draw(0, 4, m_splat->splat_count);

    sg_end_pass();
    sg_commit();
}

void initialize_splat_pipeline(splat_t *splat, Camera *camera) {
    if (!splat) {
        print("Failed to create splat pipeline, please initialize the scene");
        return;
    }
    m_splat = splat;
    g_camera = camera;

    m_splat->model = HMM_M4D(1.0f);
    g_camera->is_camera_changed = true;

    // Initialize parallel radix sort (use 4 threads for WASM)
    radix_sort_init(4);

    // CPU Sorting Data
    m_splat->splat_indices =
        (uint32_t *)malloc(m_splat->splat_count * sizeof(uint32_t));
    m_splat->splat_depths =
        (float *)malloc(m_splat->splat_count * sizeof(float));
    m_splat->m_depthIndexPair =
        (DepthIndexPair *)malloc(m_splat->splat_count * sizeof(DepthIndexPair));

    // Allocate temp buffer for radix sort double-buffering
    g_radix_temp =
        (DepthIndexPair *)malloc(m_splat->splat_count * sizeof(DepthIndexPair));

    for (int i = 0; i < m_splat->splat_count; i++) {
        m_splat->splat_indices[i] = (uint32_t)i;
    }

    // Buffer initialization
    m_splat->g_index_buffer = sg_make_buffer(&(sg_buffer_desc){
        .usage = {.vertex_buffer = true, .dynamic_update = true},
        .size = m_splat->splat_count * sizeof(uint32_t),
        .label = "Sorted_Index_Buffer"});

    float vertices[] = {
        -1.0f, -1.0f, // Bottom-left
        1.0f,  -1.0f, // Bottom-right
        -1.0f, 1.0f,  // Top-left
        1.0f,  1.0f   // Top-right
    };

    m_splat->g_surface_bind.vertex_buffers[0] =
        sg_make_buffer(&(sg_buffer_desc){.usage = {.vertex_buffer = true},
                                         .data = SG_RANGE(vertices),
                                         .label = "quad_vertices"});

    sg_shader shd = sg_make_shader(splat_shader_desc(sg_query_backend()));

    // Pipeline
    m_splat->g_surface_pip = sg_make_pipeline(&(sg_pipeline_desc){
        .shader = shd,
        .primitive_type = SG_PRIMITIVETYPE_TRIANGLE_STRIP,
        .colors[0] =
            {
                .blend = {.enabled = true,
                          .src_factor_rgb = SG_BLENDFACTOR_SRC_ALPHA,
                          .dst_factor_rgb = SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                          .src_factor_alpha = SG_BLENDFACTOR_ONE,
                          .dst_factor_alpha =
                              SG_BLENDFACTOR_ONE_MINUS_SRC_ALPHA},
            },
        .index_type = SG_INDEXTYPE_NONE,
        .layout =
            {.attrs =
                 {[ATTR_splat_position] = {.format = SG_VERTEXFORMAT_FLOAT2,
                                           .buffer_index = 0},
                  [ATTR_splat_sorted_index] = {.format = SG_VERTEXFORMAT_UINT,
                                               .buffer_index = 1}},
             .buffers = {[0] = {.stride = 8,
                                .step_func = SG_VERTEXSTEP_PER_VERTEX},
                         [1] = {.stride = 4,
                                .step_func = SG_VERTEXSTEP_PER_INSTANCE}}},

        .depth = {.write_enabled = false, .compare = SG_COMPAREFUNC_ALWAYS},
        .cull_mode = SG_CULLMODE_NONE,
        .label = "splat-pipeline"});

    m_splat->surface_pass_action = (sg_pass_action){
        .colors[0] = {.load_action = SG_LOADACTION_LOAD,
                      .clear_value = {0.0f, 0.0f, 0.0f, 0.0f}}};

    m_splat->uniforms_dirty = true;

    print("Surface Pipeline Ready (with parallel radix sort)");
}

void execute_splat_pipeline(sg_swapchain swapchain) {
    if (!m_splat || !m_splat->splats_initialized)
        return;

    HMM_Mat4 translation = HMM_Translate(HMM_V3(0.0, 0.0, 0.0));
    HMM_Mat4 rotation = HMM_Rotate_RH(3.14f, HMM_V3(0.0f, 0.0f, 1.0f));
    HMM_Mat4 scale = HMM_Scale(HMM_V3(3.0f, 3.0f, 3.0f));

    // Model = T * R * S
    m_splat->model = HMM_Mul(translation, HMM_Mul(rotation, scale));

    Calculate_Depth();
    Sort_Splat();
    g_camera->is_camera_changed = false;
    Render_Splat(swapchain);
}

void cleanup_splat_pipeline() {
    // Cleanup radix sort thread pool
    radix_sort_cleanup();

    // Free CPU buffers
    free(m_splat->m_depthIndexPair);
    free(m_splat->splat_indices);
    free(m_splat->splat_depths);
    free(g_radix_temp);
    g_radix_temp = NULL;

    if (m_splat->g_index_buffer.id != SG_INVALID_ID) {
        sg_destroy_buffer(m_splat->g_index_buffer);
    }

    if (m_splat->g_surface_pip.id != SG_INVALID_ID) {
        sg_destroy_pipeline(m_splat->g_surface_pip);
    }

    cleanup_splat_texture(&m_splat->splat_texture);
}
