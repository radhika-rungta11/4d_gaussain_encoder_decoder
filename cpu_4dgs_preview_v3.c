/*
 * cpu_4dgs_preview_v3.c
 *
 * Simple CPU preview renderer for .4dgs files.
 * This is a debug renderer, not the final Gaussian Splatting renderer.
 *
 * Compile:
 *   gcc -std=c99 -O2 temporal_4dgs_decoder.c cpu_4dgs_preview_v3.c -o cpu_4dgs_preview -lm
 *
 * Run:
 *   ./cpu_4dgs_preview ours_cook_spinach.4dgs frames 120 1280 720
 */

#include "temporal_4dgs_decoder.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct Vec3 {
    float x;
    float y;
    float z;
} Vec3;

typedef struct Image {
    int width;
    int height;
    uint8_t *rgb;
} Image;

static int make_dir(const char *path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p \"%s\"", path);
    return system(cmd);
}

static float clampf_local(float x, float a, float b) {
    if (x < a) return a;
    if (x > b) return b;
    return x;
}

static uint8_t clamp_u8(float x) {
    x = clampf_local(x, 0.0f, 255.0f);
    return (uint8_t)(x + 0.5f);
}

static float sigmoidf_safe(float x) {
    if (x < -30.0f) return 0.0f;
    if (x > 30.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

static Image image_create(int width, int height) {
    Image img;
    img.width = width;
    img.height = height;
    img.rgb = (uint8_t *)calloc((size_t)width * (size_t)height * 3u, 1u);
    return img;
}

static void image_destroy(Image *img) {
    if (img && img->rgb) {
        free(img->rgb);
        img->rgb = NULL;
    }
}

static void image_clear(Image *img, uint8_t r, uint8_t g, uint8_t b) {
    if (!img || !img->rgb) return;

    for (int y = 0; y < img->height; ++y) {
        for (int x = 0; x < img->width; ++x) {
            size_t idx = ((size_t)y * img->width + x) * 3u;
            img->rgb[idx + 0] = r;
            img->rgb[idx + 1] = g;
            img->rgb[idx + 2] = b;
        }
    }
}

static void image_blend_pixel(
    Image *img,
    int x,
    int y,
    uint8_t r,
    uint8_t g,
    uint8_t b,
    float alpha
) {
    if (!img || !img->rgb) return;
    if (x < 0 || y < 0 || x >= img->width || y >= img->height) return;

    alpha = clampf_local(alpha, 0.0f, 1.0f);

    size_t idx = ((size_t)y * img->width + x) * 3u;

    img->rgb[idx + 0] = clamp_u8((1.0f - alpha) * img->rgb[idx + 0] + alpha * r);
    img->rgb[idx + 1] = clamp_u8((1.0f - alpha) * img->rgb[idx + 1] + alpha * g);
    img->rgb[idx + 2] = clamp_u8((1.0f - alpha) * img->rgb[idx + 2] + alpha * b);
}

static int image_write_ppm(const Image *img, const char *path) {
    if (!img || !img->rgb) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to write %s: %s\n", path, strerror(errno));
        return -1;
    }

    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    fwrite(img->rgb, 1, (size_t)img->width * img->height * 3u, f);
    fclose(f);

    return 0;
}

static Vec3 get_xyz(const Temporal4DGSScene *scene, uint32_t i) {
    Vec3 p;

    const uint16_t *h = &scene->xyz_f16[(size_t)i * 3u];

    p.x = temporal_4dgs_half_to_float(h[0]);
    p.y = temporal_4dgs_half_to_float(h[1]);
    p.z = temporal_4dgs_half_to_float(h[2]);

    return p;
}

static void compute_bounds(const Temporal4DGSScene *scene, Vec3 *mn, Vec3 *mx) {
    mn->x = 1e30f;
    mn->y = 1e30f;
    mn->z = 1e30f;

    mx->x = -1e30f;
    mx->y = -1e30f;
    mx->z = -1e30f;

    for (uint32_t i = 0; i < scene->count; ++i) {
        Vec3 p = get_xyz(scene, i);

        if (p.x < mn->x) mn->x = p.x;
        if (p.y < mn->y) mn->y = p.y;
        if (p.z < mn->z) mn->z = p.z;

        if (p.x > mx->x) mx->x = p.x;
        if (p.y > mx->y) mx->y = p.y;
        if (p.z > mx->z) mx->z = p.z;
    }
}

static Vec3 rotate_y(Vec3 p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);

    Vec3 out;
    out.x = c * p.x + s * p.z;
    out.y = p.y;
    out.z = -s * p.x + c * p.z;

    return out;
}

static Vec3 rotate_x(Vec3 p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);

    Vec3 out;
    out.x = p.x;
    out.y = c * p.y - s * p.z;
    out.z = s * p.y + c * p.z;

    return out;
}

static void feature_to_rgb(
    const Temporal4DGSScene *scene,
    uint32_t i,
    uint8_t *r,
    uint8_t *g,
    uint8_t *b
) {
    if (scene->has_features_dc && scene->features_dc_f16) {
        const uint16_t *f = &scene->features_dc_f16[(size_t)i * 6u];

        float fr = temporal_4dgs_half_to_float(f[0]);
        float fg = temporal_4dgs_half_to_float(f[1]);
        float fb = temporal_4dgs_half_to_float(f[2]);

        *r = clamp_u8(sigmoidf_safe(fr) * 255.0f);
        *g = clamp_u8(sigmoidf_safe(fg) * 255.0f);
        *b = clamp_u8(sigmoidf_safe(fb) * 255.0f);
    } else {
        float t = (float)i / (float)(scene->count > 1 ? scene->count - 1 : 1);

        *r = clamp_u8(255.0f * t);
        *g = clamp_u8(255.0f * (1.0f - t));
        *b = 180;
    }
}

static int render_frame(
    const Temporal4DGSScene *scene,
    Vec3 center,
    float scene_radius,
    int frame_index,
    int frame_count,
    Image *img
) {
    image_clear(img, 8, 8, 12);

    float t = 0.0f;

    if (frame_count > 1) {
        t = (float)frame_index / (float)(frame_count - 1);
    }

    float angle_y = t * 2.0f * (float)M_PI;
    float angle_x = -0.25f;

    float camera_distance = scene_radius * 2.8f;
    float focal = 0.80f * (float)img->height;

    for (uint32_t i = 0; i < scene->count; ++i) {
        Vec3 p = get_xyz(scene, i);

        p.x -= center.x;
        p.y -= center.y;
        p.z -= center.z;

        p = rotate_y(p, angle_y);
        p = rotate_x(p, angle_x);

        float z = p.z + camera_distance;

        if (z <= 0.01f) {
            continue;
        }

        int screen_x = (int)((p.x / z) * focal + img->width * 0.5f);
        int screen_y = (int)((-p.y / z) * focal + img->height * 0.5f);

        float opacity = 0.6f;

        if (scene->opacity) {
            opacity = sigmoidf_safe(scene->opacity[i]);

            if (opacity < 0.03f) {
                opacity = 0.03f;
            }
        }

        uint8_t r;
        uint8_t g;
        uint8_t b;

        feature_to_rgb(scene, i, &r, &g, &b);

        int radius = 1;

        if (scene->scale && scene->scale_dim >= 3) {
            float sx = scene->scale[(size_t)i * scene->scale_dim + 0];
            float sy = scene->scale[(size_t)i * scene->scale_dim + 1];
            float sz = scene->scale[(size_t)i * scene->scale_dim + 2];

            float avg = (fabsf(sx) + fabsf(sy) + fabsf(sz)) / 3.0f;
            radius = 1 + (int)clampf_local(avg * 0.8f, 0.0f, 3.0f);
        }

        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float d2 = (float)(dx * dx + dy * dy);
                float rr = (float)(radius * radius + 1);
                float a = opacity * expf(-d2 / rr) * 0.25f;

                image_blend_pixel(img, screen_x + dx, screen_y + dy, r, g, b, a);
            }
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(
            stderr,
            "Usage:\n"
            "  %s <scene.4dgs> [output_dir] [frame_count] [width] [height]\n\n"
            "Example:\n"
            "  %s ours_cook_spinach.4dgs frames 120 1280 720\n",
            argv[0],
            argv[0]
        );

        return 1;
    }

    const char *scene_path = argv[1];
    const char *out_dir = argc >= 3 ? argv[2] : "frames";

    int frame_count = argc >= 4 ? atoi(argv[3]) : 120;
    int width = argc >= 5 ? atoi(argv[4]) : 1280;
    int height = argc >= 6 ? atoi(argv[5]) : 720;

    if (frame_count <= 0) frame_count = 120;
    if (width <= 0) width = 1280;
    if (height <= 0) height = 720;

    printf("Loading 4DGS file: %s\n", scene_path);

    Temporal4DGSScene scene;
    memset(&scene, 0, sizeof(scene));

    if (temporal_4dgs_load_file(scene_path, &scene) != 0) {
        fprintf(stderr, "Failed to load scene: %s\n", scene_path);
        return 1;
    }

    printf("Loaded scene with %u Gaussians\n", scene.count);
    printf(
        "scale_dim=%u rotation_dim=%u omega_dim=%u tfea_dim=%u\n",
        scene.scale_dim,
        scene.rotation_dim,
        scene.omega_dim,
        scene.tfea_dim
    );

    Vec3 mn;
    Vec3 mx;

    compute_bounds(&scene, &mn, &mx);

    Vec3 center;

    center.x = (mn.x + mx.x) * 0.5f;
    center.y = (mn.y + mx.y) * 0.5f;
    center.z = (mn.z + mx.z) * 0.5f;

    Vec3 extent;

    extent.x = mx.x - mn.x;
    extent.y = mx.y - mn.y;
    extent.z = mx.z - mn.z;

    float scene_radius = sqrtf(
        extent.x * extent.x +
        extent.y * extent.y +
        extent.z * extent.z
    ) * 0.5f;

    if (scene_radius < 1.0f) {
        scene_radius = 1.0f;
    }

    printf("Bounds min=(%.3f, %.3f, %.3f)\n", mn.x, mn.y, mn.z);
    printf("Bounds max=(%.3f, %.3f, %.3f)\n", mx.x, mx.y, mx.z);
    printf("Center    =(%.3f, %.3f, %.3f)\n", center.x, center.y, center.z);
    printf("Radius    =%.3f\n", scene_radius);

    make_dir(out_dir);

    Image img = image_create(width, height);

    if (!img.rgb) {
        fprintf(stderr, "Failed to allocate image buffer\n");
        temporal_4dgs_free_scene(&scene);
        return 1;
    }

    for (int f = 0; f < frame_count; ++f) {
        char path[1024];

        snprintf(path, sizeof(path), "%s/frame_%04d.ppm", out_dir, f);

        render_frame(&scene, center, scene_radius, f, frame_count, &img);

        if (image_write_ppm(&img, path) != 0) {
            fprintf(stderr, "Failed to write frame %d\n", f);
            image_destroy(&img);
            temporal_4dgs_free_scene(&scene);
            return 1;
        }

        if (f % 10 == 0 || f == frame_count - 1) {
            printf("Wrote %s\n", path);
        }
    }

    image_destroy(&img);
    temporal_4dgs_free_scene(&scene);

    printf("\nDone. Frames written to: %s\n", out_dir);
    printf("Create video with:\n");
    printf(
        "  ffmpeg -y -framerate 30 -i %s/frame_%%04d.ppm -c:v libx264 -pix_fmt yuv420p output_preview.mp4\n",
        out_dir
    );

    return 0;
}