/* T112 L2-resident 6-regular lattice hop harness.
 * gcc -O3 -march=native -std=c11 bench_l2_graph.c -o bench_l2_graph
 *
 * Nodes are 12-byte packed (6 * uint16). Do not pad nodes to 64 B.
 * Align the base of the table to 64 B. sizeof tight table = 75936;
 * allocation rounded up to 75968.
 */
#define _GNU_SOURCE
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#define NUM_NODES 6328
#define DEGREE    6
#define HOPS      50000000u
#define TABLE_BYTES (NUM_NODES * DEGREE * (int)sizeof(uint16_t))
#define ALLOC_BYTES (((TABLE_BYTES + 63) / 64) * 64)

#if defined(__GNUC__) || defined(__clang__)
typedef struct {
    uint16_t neighbors[DEGREE];
} __attribute__((packed)) Node;
#else
#pragma pack(push, 1)
typedef struct {
    uint16_t neighbors[DEGREE];
} Node;
#pragma pack(pop)
#endif

_Static_assert(sizeof(Node) == 12, "Node struct leaked memory or contains padding");
_Static_assert(sizeof(Node) * NUM_NODES == 75936,
               "Total table size must equal exactly 75936 bytes (74.156 KiB)");
_Static_assert(75936 <= (256 * 1024), "Table overflows 256 KiB L2 cache ceiling");
_Static_assert(TABLE_BYTES == 75936, "TABLE_BYTES drifted from packed layout");

static void *xaligned(size_t align, size_t n) {
#ifdef _MSC_VER
    return _aligned_malloc(n, align);
#else
    return aligned_alloc(align, n);
#endif
}

static void xfree(void *p) {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    free(p);
#endif
}

static uint64_t nsec(void) {
#ifdef _WIN32
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

__attribute__((noinline))
static uint16_t run_hops(const Node *g, const uint8_t *moves, size_t count) {
    uint16_t cur = 0;
    for (size_t i = 0; i < count; ++i) {
        uint16_t nxt = g[cur].neighbors[moves[i]];
        if (nxt >= NUM_NODES) return cur;
        cur = nxt;
#if defined(__GNUC__)
        asm volatile("" : "+r"(cur));
#endif
    }
    return cur;
}

int main(void) {
    Node *graph = xaligned(64, ALLOC_BYTES);
    uint8_t *moves = malloc(HOPS);
    if (!graph || !moves) return 1;
    memset(graph, 0, ALLOC_BYTES);

    for (uint32_t i = 0; i < NUM_NODES; ++i) {
        static const int off[6] = {1, 2, 3, -1, -2, -3};
        for (int d = 0; d < DEGREE; ++d) {
            int v = (int)i + off[d];
            v %= NUM_NODES;
            if (v < 0) v += NUM_NODES;
            graph[i].neighbors[d] = (uint16_t)v;
        }
    }
    for (size_t i = 0; i < HOPS; ++i) moves[i] = (uint8_t)(i % DEGREE);

    printf("[+] packed table  %d bytes (%.4f KiB)\n", TABLE_BYTES, TABLE_BYTES / 1024.0);
    printf("[+] alloc (64-B)  %d bytes  lines=%.2f  nodes/line=%.2f\n",
           ALLOC_BYTES, TABLE_BYTES / 64.0, 64.0 / 12.0);
    printf("[+] padded-node trap would be %d bytes (%.1f KiB) -- L2 miss. not used.\n",
           NUM_NODES * 64, NUM_NODES * 64 / 1024.0);

    (void)run_hops(graph, moves, 100000);

    uint64_t t0 = nsec();
    uint16_t last = run_hops(graph, moves, HOPS);
    uint64_t t1 = nsec();
    double ns_hop = (double)(t1 - t0) / (double)HOPS;
    printf("[+] hops %u  %.4f s  %.2f ns/hop  last=%u\n",
           HOPS, (t1 - t0) / 1e9, ns_hop, (unsigned)last);

    xfree(graph);
    free(moves);
    return 0;
}
