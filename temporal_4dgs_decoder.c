/*
 * temporal_4dgs_decoder.c
 *
 * Decoder for the .4dgs v3 binary format produced by parser.py.
 *
 * Compile (raw .4dgs only):
 *   gcc -std=c99 -O2 temporal_4dgs_decoder.c test_4dgs_decoder.c -o test_4dgs_decoder -lm
 *
 * Compile with zlib (also supports .4dgs.gz):
 *   gcc -std=c99 -O2 -DTEMPORAL_4DGS_ENABLE_ZLIB \
 *       temporal_4dgs_decoder.c test_4dgs_decoder.c -o test_4dgs_decoder -lm -lz
 */

#include "temporal_4dgs_decoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef TEMPORAL_4DGS_ENABLE_ZLIB
#include <zlib.h>
#endif

/* ============================================================
 * Internal cursor — safe bounds-checked reader
 * ============================================================ */
typedef struct {
    const uint8_t *base;
    size_t         size;
    size_t         pos;
    int            error; /* set to 1 on any overread */
} Cursor;

static void cursor_init(Cursor *c, const uint8_t *data, size_t size) {
    c->base  = data;
    c->size  = size;
    c->pos   = 0;
    c->error = 0;
}

static int cursor_read(Cursor *c, void *dst, size_t n) {
    if (c->error) return 0;
    if (c->pos + n > c->size) {
        fprintf(stderr, "[4dgs] Read overrun: need %zu bytes at offset %zu, "
                        "but buffer is only %zu bytes\n", n, c->pos, c->size);
        c->error = 1;
        return 0;
    }
    memcpy(dst, c->base + c->pos, n);
    c->pos += n;
    return 1;
}

/* Typed helpers */
static uint8_t  read_u8 (Cursor *c) { uint8_t  v=0; cursor_read(c,&v,1); return v; }
static uint16_t read_u16(Cursor *c) { uint16_t v=0; cursor_read(c,&v,2); return v; }
static uint32_t read_u32(Cursor *c) { uint32_t v=0; cursor_read(c,&v,4); return v; }
static float    read_f32(Cursor *c) { float    v=0; cursor_read(c,&v,4); return v; }

/* ============================================================
 * half → float
 * ============================================================ */
float temporal_4dgs_half_to_float(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            /* Denormalised */
            exponent = 1;
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        /* Inf / NaN */
        result = sign | 0x7F800000 | (mantissa << 13);
    } else {
        result = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &result, 4);
    return f;
}

/* ============================================================
 * Huffman table (decoding only)
 * ============================================================ */
#define MAX_HUFFMAN_SYMBOLS 65536
#define MAX_HUFFMAN_BITLEN  32

typedef struct {
    uint16_t symbol;
    uint8_t  bitlen;
    uint32_t code_bits;
} HuffEntry;

typedef struct {
    HuffEntry *entries;
    uint16_t   count;
} HuffTable;

/* Read Huffman table from cursor */
static int huff_read(Cursor *c, HuffTable *ht) {
    ht->count   = read_u16(c);
    ht->entries = NULL;
    if (c->error) return 0;

    if (ht->count == 0) return 1; /* empty table is ok */

    ht->entries = (HuffEntry *)malloc(ht->count * sizeof(HuffEntry));
    if (!ht->entries) {
        fprintf(stderr, "[4dgs] OOM allocating Huffman table (%u entries)\n",
                ht->count);
        return 0;
    }

    for (uint16_t i = 0; i < ht->count; i++) {
        ht->entries[i].symbol    = read_u16(c);
        ht->entries[i].bitlen    = read_u8 (c);
        ht->entries[i].code_bits = read_u32(c);
        if (c->error) { free(ht->entries); ht->entries = NULL; return 0; }
    }
    return 1;
}

static void huff_free(HuffTable *ht) {
    free(ht->entries);
    ht->entries = NULL;
    ht->count   = 0;
}

/* Bitstream reader for Huffman decoding */
typedef struct {
    const uint8_t *data;
    size_t         size;   /* bytes */
    size_t         byte_pos;
    int            bit_pos; /* 0..7, MSB first */
} Bitstream;

static void bs_init(Bitstream *bs, const uint8_t *data, size_t size) {
    bs->data     = data;
    bs->size     = size;
    bs->byte_pos = 0;
    bs->bit_pos  = 7;
}

static int bs_read_bit(Bitstream *bs) {
    if (bs->byte_pos >= bs->size) return -1;
    int bit = (bs->data[bs->byte_pos] >> bs->bit_pos) & 1;
    if (bs->bit_pos == 0) { bs->byte_pos++; bs->bit_pos = 7; }
    else                  { bs->bit_pos--; }
    return bit;
}

/**
 * Decode `n` symbols using canonical Huffman (linear scan).
 * For large tables a lookup table would be faster, but correctness first.
 */
static int huff_decode_n(const HuffTable *ht, Bitstream *bs,
                          uint16_t *out, size_t n)
{
    for (size_t sym_idx = 0; sym_idx < n; sym_idx++) {
        uint32_t code      = 0;
        uint8_t  code_bits = 0;
        int      found     = 0;

        /* We accumulate bits and check against every table entry. */
        /* This is O(bitlen * table_size) per symbol — acceptable for decode. */
        uint8_t max_bits = 0;
        for (uint16_t i = 0; i < ht->count; i++) {
            if (ht->entries[i].bitlen > max_bits)
                max_bits = ht->entries[i].bitlen;
        }

        for (uint8_t depth = 1; depth <= max_bits && !found; depth++) {
            int bit = bs_read_bit(bs);
            if (bit < 0) {
                fprintf(stderr, "[4dgs] Bitstream underrun at symbol %zu\n",
                        sym_idx);
                return 0;
            }
            code = (code << 1) | (uint32_t)bit;
            code_bits++;

            for (uint16_t i = 0; i < ht->count; i++) {
                if (ht->entries[i].bitlen  == code_bits &&
                    ht->entries[i].code_bits == code) {
                    out[sym_idx] = ht->entries[i].symbol;
                    found = 1;
                    break;
                }
            }
        }

        if (!found) {
            fprintf(stderr, "[4dgs] Huffman decode failed at symbol %zu "
                            "(code=0x%X, bits=%u)\n", sym_idx, code, code_bits);
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 * Scalar block decoder
 * opacity, tcen, tsca — Huffman + dequantize
 * ============================================================ */
static float *decode_scalar_block(Cursor *c, uint32_t N) {
    float min_val = read_f32(c);
    float max_val = read_f32(c);
    if (c->error) return NULL;

    HuffTable ht = {0};
    if (!huff_read(c, &ht)) return NULL;

    uint32_t data_len = read_u32(c);
    if (c->error) { huff_free(&ht); return NULL; }

    if (c->pos + data_len > c->size) {
        fprintf(stderr, "[4dgs] Scalar block data overrun: need %u bytes\n",
                data_len);
        huff_free(&ht);
        return NULL;
    }
    const uint8_t *encoded = c->base + c->pos;
    c->pos += data_len;

    /* Huffman decode into quantised symbols (0..255) */
    uint16_t *q_vals = (uint16_t *)malloc(N * sizeof(uint16_t));
    if (!q_vals) { huff_free(&ht); return NULL; }

    Bitstream bs;
    bs_init(&bs, encoded, data_len);
    if (!huff_decode_n(&ht, &bs, q_vals, N)) {
        free(q_vals); huff_free(&ht); return NULL;
    }
    huff_free(&ht);

    /* Dequantise */
    float *out = (float *)malloc(N * sizeof(float));
    if (!out) { free(q_vals); return NULL; }

    float range = max_val - min_val;
    for (uint32_t i = 0; i < N; i++) {
        out[i] = min_val + range * ((float)q_vals[i] / 255.0f);
    }
    free(q_vals);
    return out;
}

/* ============================================================
 * VQ block decoder
 * scale, rotation, omega, tfea
 * ============================================================ */
typedef struct {
    float   *data;       /* [N * dim] */
    uint16_t dim;
} VQResult;

static int decode_vq_block(Cursor *c, uint32_t N, VQResult *out) {
    uint8_t  num_layers     = read_u8 (c);
    uint16_t codebook_size  = read_u16(c);
    uint16_t dim            = read_u16(c);
    if (c->error) return 0;

    out->dim  = dim;
    out->data = NULL;

    /* Read codebooks: [num_layers, codebook_size, dim] float16 */
    size_t cb_count = (size_t)num_layers * codebook_size * dim;
    uint16_t *cb_f16 = (uint16_t *)malloc(cb_count * sizeof(uint16_t));
    if (!cb_f16) return 0;
    if (!cursor_read(c, cb_f16, cb_count * sizeof(uint16_t))) {
        free(cb_f16); return 0;
    }

    /* Convert codebooks to float32 for reconstruction */
    float *cb_f32 = (float *)malloc(cb_count * sizeof(float));
    if (!cb_f32) { free(cb_f16); return 0; }
    for (size_t i = 0; i < cb_count; i++)
        cb_f32[i] = temporal_4dgs_half_to_float(cb_f16[i]);
    free(cb_f16);

    /* Read Huffman table */
    HuffTable ht = {0};
    if (!huff_read(c, &ht)) { free(cb_f32); return 0; }

    uint32_t data_len = read_u32(c);
    if (c->error) { free(cb_f32); huff_free(&ht); return 0; }

    if (c->pos + data_len > c->size) {
        fprintf(stderr, "[4dgs] VQ block data overrun: need %u bytes\n",
                data_len);
        free(cb_f32); huff_free(&ht); return 0;
    }
    const uint8_t *encoded = c->base + c->pos;
    c->pos += data_len;

    /* Huffman decode indices: N * num_layers indices */
    size_t n_indices = (size_t)N * num_layers;
    uint16_t *indices = (uint16_t *)malloc(n_indices * sizeof(uint16_t));
    if (!indices) { free(cb_f32); huff_free(&ht); return 0; }

    Bitstream bs;
    bs_init(&bs, encoded, data_len);
    if (!huff_decode_n(&ht, &bs, indices, n_indices)) {
        free(indices); free(cb_f32); huff_free(&ht); return 0;
    }
    huff_free(&ht);

    /* Reconstruct: for each Gaussian, sum contributions from each layer */
    float *result = (float *)calloc((size_t)N * dim, sizeof(float));
    if (!result) { free(indices); free(cb_f32); return 0; }

    for (uint32_t g = 0; g < N; g++) {
        for (uint8_t layer = 0; layer < num_layers; layer++) {
            uint16_t idx = indices[(size_t)g * num_layers + layer];
            if (idx >= codebook_size) {
                fprintf(stderr,
                        "[4dgs] VQ index %u out of range (codebook_size=%u) "
                        "at Gaussian %u layer %u\n",
                        idx, codebook_size, g, layer);
                free(result); free(indices); free(cb_f32); return 0;
            }
            /* codebook layout: [layer, idx, d] */
            const float *entry = cb_f32 +
                ((size_t)layer * codebook_size + idx) * dim;
            float *dst = result + (size_t)g * dim;
            for (uint16_t d = 0; d < dim; d++)
                dst[d] += entry[d];
        }
    }

    free(indices);
    free(cb_f32);
    out->data = result;
    return 1;
}

/* ============================================================
 * Main parse function
 * ============================================================ */
int temporal_4dgs_parse_memory(const uint8_t *data, size_t size,
                               Temporal4DGSScene *out_scene)
{
    if (!data || !out_scene) return -1;
    memset(out_scene, 0, sizeof(*out_scene));

    Cursor c;
    cursor_init(&c, data, size);

    /* --- Header --- */
    char magic[4];
    cursor_read(&c, magic, 4);
    if (c.error) { fprintf(stderr, "[4dgs] File too small for header\n"); return -1; }
    if (memcmp(magic, "4DGS", 4) != 0) {
        fprintf(stderr, "[4dgs] Bad magic: expected '4DGS', got '%.4s'\n", magic);
        return -1;
    }

    uint32_t version = read_u32(&c);
    if (c.error) return -1;
    if (version != 3) {
        fprintf(stderr, "[4dgs] Unsupported version %u (expected 3)\n", version);
        return -1;
    }

    uint32_t N = read_u32(&c);
    if (c.error) return -1;
    if (N == 0) { fprintf(stderr, "[4dgs] N=0 Gaussians\n"); return -1; }
    out_scene->count = N;

    printf("[4dgs] magic=4DGS  version=%u  N=%u\n", version, N);

    /* --- xyz float16 [N,3] --- */
    out_scene->xyz_f16 = (uint16_t *)malloc((size_t)N * 3 * sizeof(uint16_t));
    if (!out_scene->xyz_f16) { fprintf(stderr, "[4dgs] OOM xyz\n"); goto fail; }
    if (!cursor_read(&c, out_scene->xyz_f16, (size_t)N * 3 * sizeof(uint16_t)))
        goto fail;

    /* --- motion float16 [N,9] --- */
    out_scene->motion_f16 = (uint16_t *)malloc((size_t)N * 9 * sizeof(uint16_t));
    if (!out_scene->motion_f16) { fprintf(stderr, "[4dgs] OOM motion\n"); goto fail; }
    if (!cursor_read(&c, out_scene->motion_f16, (size_t)N * 9 * sizeof(uint16_t)))
        goto fail;

    /* --- Scalar blocks --- */
    printf("[4dgs] Decoding opacity...\n");
    out_scene->opacity = decode_scalar_block(&c, N);
    if (!out_scene->opacity) { fprintf(stderr, "[4dgs] Failed opacity\n"); goto fail; }

    printf("[4dgs] Decoding tcen...\n");
    out_scene->tcen = decode_scalar_block(&c, N);
    if (!out_scene->tcen) { fprintf(stderr, "[4dgs] Failed tcen\n"); goto fail; }

    printf("[4dgs] Decoding tsca...\n");
    out_scene->tsca = decode_scalar_block(&c, N);
    if (!out_scene->tsca) { fprintf(stderr, "[4dgs] Failed tsca\n"); goto fail; }

    /* --- VQ blocks --- */
    printf("[4dgs] Decoding scale (VQ)...\n");
    { VQResult vq = {0};
      if (!decode_vq_block(&c, N, &vq)) { fprintf(stderr, "[4dgs] Failed scale\n"); goto fail; }
      out_scene->scale = vq.data; out_scene->scale_dim = vq.dim; }

    printf("[4dgs] Decoding rotation (VQ)...\n");
    { VQResult vq = {0};
      if (!decode_vq_block(&c, N, &vq)) { fprintf(stderr, "[4dgs] Failed rotation\n"); goto fail; }
      out_scene->rotation = vq.data; out_scene->rotation_dim = vq.dim; }

    printf("[4dgs] Decoding omega (VQ)...\n");
    { VQResult vq = {0};
      if (!decode_vq_block(&c, N, &vq)) { fprintf(stderr, "[4dgs] Failed omega\n"); goto fail; }
      out_scene->omega = vq.data; out_scene->omega_dim = vq.dim; }

    printf("[4dgs] Decoding tfea (VQ)...\n");
    { VQResult vq = {0};
      if (!decode_vq_block(&c, N, &vq)) { fprintf(stderr, "[4dgs] Failed tfea\n"); goto fail; }
      out_scene->tfea = vq.data; out_scene->tfea_dim = vq.dim; }

    /* --- Optional features_dc --- */
    if (c.pos < c.size) {
        out_scene->has_features_dc = read_u8(&c);
        if (!c.error && out_scene->has_features_dc == 1) {
            printf("[4dgs] Decoding features_dc...\n");
            size_t fdc_count = (size_t)N * 6;
            out_scene->features_dc_f16 =
                (uint16_t *)malloc(fdc_count * sizeof(uint16_t));
            if (!out_scene->features_dc_f16) goto fail;
            if (!cursor_read(&c, out_scene->features_dc_f16,
                             fdc_count * sizeof(uint16_t)))
                goto fail;
        }
    }

    /* --- Optional rgb_dec --- */
    if (c.pos < c.size) {
        out_scene->has_rgb_dec = read_u8(&c);
        if (!c.error && out_scene->has_rgb_dec == 1) {
            printf("[4dgs] Decoding rgb_dec weights...\n");
            /* w1: [6,12] = 72 values */
            out_scene->rgb_dec_w1_f16 =
                (uint16_t *)malloc(72 * sizeof(uint16_t));
            if (!out_scene->rgb_dec_w1_f16) goto fail;
            if (!cursor_read(&c, out_scene->rgb_dec_w1_f16,
                             72 * sizeof(uint16_t)))
                goto fail;

            /* w2: [3,6] = 18 values */
            out_scene->rgb_dec_w2_f16 =
                (uint16_t *)malloc(18 * sizeof(uint16_t));
            if (!out_scene->rgb_dec_w2_f16) goto fail;
            if (!cursor_read(&c, out_scene->rgb_dec_w2_f16,
                             18 * sizeof(uint16_t)))
                goto fail;
        }
    }

    if (c.error) {
        fprintf(stderr, "[4dgs] Cursor error at end of parse\n");
        goto fail;
    }

    printf("[4dgs] Parse complete. %zu bytes consumed of %zu.\n",
           c.pos, c.size);
    return 0;

fail:
    temporal_4dgs_free_scene(out_scene);
    return -1;
}

/* ============================================================
 * File loader
 * ============================================================ */
int temporal_4dgs_load_file(const char *path, Temporal4DGSScene *out_scene) {
    if (!path || !out_scene) return -1;

    /* Check extension */
    size_t plen = strlen(path);
    int is_gz = (plen > 3 && strcmp(path + plen - 3, ".gz") == 0);

    if (is_gz) {
#ifdef TEMPORAL_4DGS_ENABLE_ZLIB
        /* --- zlib inflate path --- */
        gzFile gz = gzopen(path, "rb");
        if (!gz) {
            fprintf(stderr, "[4dgs] Cannot open gz file: %s\n", path);
            return -1;
        }
        size_t   capacity = 64 * 1024 * 1024; /* 64 MB initial */
        uint8_t *buf      = (uint8_t *)malloc(capacity);
        size_t   used     = 0;
        if (!buf) { gzclose(gz); return -1; }

        int n;
        while ((n = gzread(gz, buf + used, (unsigned)(capacity - used))) > 0) {
            used += (size_t)n;
            if (used == capacity) {
                capacity *= 2;
                uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
                if (!tmp) { free(buf); gzclose(gz); return -1; }
                buf = tmp;
            }
        }
        gzclose(gz);
        int ret = temporal_4dgs_parse_memory(buf, used, out_scene);
        free(buf);
        return ret;
#else
        fprintf(stderr,
                "[4dgs] File '%s' appears to be gzip-compressed, but "
                "TEMPORAL_4DGS_ENABLE_ZLIB is not defined.\n", path);
        return -1;
#endif
    }

    /* --- Raw .4dgs path --- */
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[4dgs] Cannot open file: %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (file_size <= 0) {
        fprintf(stderr, "[4dgs] Empty or unreadable file: %s\n", path);
        fclose(f);
        return -1;
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)file_size);
    if (!buf) {
        fprintf(stderr, "[4dgs] OOM loading file (%ld bytes)\n", file_size);
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)file_size, f) != (size_t)file_size) {
        fprintf(stderr, "[4dgs] Short read: %s\n", path);
        free(buf); fclose(f);
        return -1;
    }
    fclose(f);

    int ret = temporal_4dgs_parse_memory(buf, (size_t)file_size, out_scene);
    free(buf);
    return ret;
}

/* ============================================================
 * Cleanup
 * ============================================================ */
void temporal_4dgs_free_scene(Temporal4DGSScene *scene) {
    if (!scene) return;
    free(scene->xyz_f16);
    free(scene->motion_f16);
    free(scene->opacity);
    free(scene->tcen);
    free(scene->tsca);
    free(scene->scale);
    free(scene->rotation);
    free(scene->omega);
    free(scene->tfea);
    free(scene->features_dc_f16);
    free(scene->rgb_dec_w1_f16);
    free(scene->rgb_dec_w2_f16);
    memset(scene, 0, sizeof(*scene));
}