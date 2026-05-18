#include "splat_texture.h"
#include "utils/logger.h"
#include <memory.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define TEXTURE_WIDTH 1024
#define TEXTURE_HEIGHT 1024
#define MAX_SPLATS_PER_LAYER (TEXTURE_WIDTH * TEXTURE_HEIGHT)

// Calculate number of pixels needed per splat based on SH degree
int get_pixels_per_splat(int sh_degree) {
    // Base data: 16 bytes = 1 pixel
    // SH coefficients:
    // - Degree 0: 0 bytes (0 additional pixels)
    // - Degree 1: 9 bytes (1 additional pixel = 16 bytes with padding)
    // - Degree 2: 24 bytes (2 additional pixels = 32 bytes with padding)
    // - Degree 3: 45 bytes (3 additional pixels = 48 bytes with padding)
    switch (sh_degree) {
    case 0:
        return 1;
    case 1:
        return 2;
    case 2:
        return 3;
    case 3:
        return 4;
    default:
        return 1;
    }
}

static uint32_t *convert_splats_to_texture_data(PackedSplat *splats,
                                                uint32_t splat_count,
                                                int texture_width,
                                                int texture_height,
                                                int num_layers, int sh_degree) {
    int pixels_per_splat = get_pixels_per_splat(sh_degree);
    size_t pixels_per_layer = (size_t)texture_width * texture_height;
    size_t total_pixels = pixels_per_layer * num_layers;
    size_t total_bytes = total_pixels * 4 * sizeof(uint32_t);

    uint32_t *texture_data = (uint32_t *)malloc(total_bytes);
    if (!texture_data) {
        error("Failed to allocate %zu bytes for texture data", total_bytes);
        return NULL;
    }

    memset(texture_data, 0, total_bytes);

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024) if (splat_count > 5000)
#endif
    for (uint32_t i = 0; i < splat_count; i++) {
        PackedSplat *splat = &splats[i];
        size_t base_index = (size_t)i * pixels_per_splat * 4;

        // Pixel 0: Position, rotation, scale, base color
        texture_data[base_index + 0] =
            ((uint32_t)splat->pos_x << 16) | splat->pos_y;
        texture_data[base_index + 1] = ((uint32_t)splat->pos_z << 16) |
                                       ((uint32_t)splat->rot_axis_u << 8) |
                                       splat->rot_axis_v;
        texture_data[base_index + 2] = ((uint32_t)splat->rot_angle << 24) |
                                       ((uint32_t)splat->scale_x << 16) |
                                       ((uint32_t)splat->scale_y << 8) |
                                       splat->scale_z;
        texture_data[base_index + 3] = ((uint32_t)splat->r << 24) |
                                       ((uint32_t)splat->g << 16) |
                                       ((uint32_t)splat->b << 8) | splat->a;

        // Pack SH coefficients into additional pixels
        // Pack SH coefficients into additional pixels
        if (sh_degree > 0) {
            int num_sh_coeffs = 0;
            if (sh_degree == 1)
                num_sh_coeffs = 9;
            else if (sh_degree == 2)
                num_sh_coeffs = 24;
            else if (sh_degree == 3)
                num_sh_coeffs = 45;

            // Each pixel has 4 uint32s, each uint32 holds 4 bytes = 16 coeffs
            // per pixel
            for (int c = 0; c < num_sh_coeffs; c++) {
                int pixel_idx = c / 16; // Which additional pixel (0, 1, 2, ...)
                int uint_in_pixel =
                    (c % 16) / 4;         // Which uint32 within pixel (0-3)
                int byte_in_uint = c % 4; // Which byte within uint32 (0-3)

                size_t offset =
                    base_index + (1 + pixel_idx) * 4 + uint_in_pixel;
                uint8_t coeff = splat->sh_coeffs[c];
                texture_data[offset] |= ((uint32_t)coeff << (byte_in_uint * 8));
            }
        }
    }

    return texture_data;
}

static void calculate_texture_dimensions(uint32_t splat_count, int *width,
                                         int *height, int *num_layers,
                                         int pixels_per_splat) {
    uint32_t required_pixels = splat_count * pixels_per_splat;

    int selected_width = 0, selected_height = 0, selected_layers = 0;
    size_t min_waste = SIZE_MAX;

    int sizes[] = {256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int texture_width = sizes[i];
        int texture_height = sizes[i];
        int pixels_per_layer = texture_width * texture_height;
        int layers =
            (int)((required_pixels + pixels_per_layer - 1) / pixels_per_layer);
        size_t total_pixels = (size_t)pixels_per_layer * layers;
        size_t waste = total_pixels - required_pixels;

        if (waste < min_waste) {
            selected_width = texture_width;
            selected_height = texture_height;
            selected_layers = layers;
            min_waste = waste;
        }

        if (waste < (float)pixels_per_layer * 0.1f) {
            break;
        }
    }

    if (selected_width == 0) {
        selected_width = TEXTURE_WIDTH;
        selected_height = TEXTURE_HEIGHT;
        selected_layers =
            (splat_count + MAX_SPLATS_PER_LAYER - 1) / MAX_SPLATS_PER_LAYER;
    }

    *width = selected_width;
    *height = selected_height;
    *num_layers = selected_layers;
}

int create_splat_texture_from_data(splat_texture_t *texture,
                                   PackedSplat *splats, uint32_t splat_count,
                                   int sh_degree) {
    int pixels_per_splat = get_pixels_per_splat(sh_degree);
    int width, height, num_layers;
    calculate_texture_dimensions(splat_count, &width, &height, &num_layers,
                                 pixels_per_splat);

    texture->width = width;
    texture->height = height;
    texture->num_layers = num_layers;
    texture->sh_degree = sh_degree;

    size_t required_pixels = splat_count * pixels_per_splat;
    size_t allocated_pixels = (size_t)width * height * num_layers;
    float efficiency = (float)required_pixels / allocated_pixels * 100.0f;

    print(
        "Texture: %u splats, %dx%d, %d layers, SH degree %d, %.1f%% efficiency",
        splat_count, width, height, num_layers, sh_degree, efficiency);

    uint32_t *texture_data =
        convert_splats_to_texture_data(splats, splat_count, texture->width,
                                       texture->height, num_layers, sh_degree);

    if (!texture_data) {
        error("Failed to convert splat data");
        return -1;
    }

    size_t total_size = (size_t)texture->width * texture->height * num_layers *
                        sizeof(uint32_t) * 4;

    sg_image_data image_data = {0};
    image_data.mip_levels[0] =
        (sg_range){.ptr = texture_data, .size = total_size};

    texture->texture =
        sg_make_image(&(sg_image_desc){.type = SG_IMAGETYPE_ARRAY,
                                       .width = texture->width,
                                       .height = texture->height,
                                       .num_slices = num_layers,
                                       .pixel_format = SG_PIXELFORMAT_RGBA32UI,
                                       .usage = {.immutable = true},
                                       .data = image_data,
                                       .label = "splat-texture"});

    texture->sampler =
        sg_make_sampler(&(sg_sampler_desc){.min_filter = SG_FILTER_NEAREST,
                                           .mag_filter = SG_FILTER_NEAREST,
                                           .mipmap_filter = SG_FILTER_NEAREST,
                                           .wrap_u = SG_WRAP_CLAMP_TO_EDGE,
                                           .wrap_v = SG_WRAP_CLAMP_TO_EDGE,
                                           .label = "splat-sampler"});

    texture->view =
        sg_make_view(&(sg_view_desc){.texture =
                                         {
                                             .image = texture->texture,
                                         },
                                     .label = "splat-texture-view"});

    free(texture_data);

    if (texture->texture.id == SG_INVALID_ID ||
        texture->sampler.id == SG_INVALID_ID ||
        texture->view.id == SG_INVALID_ID) {
        error("Failed to create texture resources");
        return -1;
    }

    return 0;
}

void cleanup_splat_texture(splat_texture_t *texture) {
    if (texture->view.id != SG_INVALID_ID) {
        sg_destroy_view(texture->view);
        texture->view.id = SG_INVALID_ID;
    }

    if (texture->sampler.id != SG_INVALID_ID) {
        sg_destroy_sampler(texture->sampler);
        texture->sampler.id = SG_INVALID_ID;
    }

    if (texture->texture.id != SG_INVALID_ID) {
        sg_destroy_image(texture->texture);
        texture->texture.id = SG_INVALID_ID;
    }
}
