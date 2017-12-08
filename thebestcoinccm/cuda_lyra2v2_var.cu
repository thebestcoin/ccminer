// Copyright (c) 2017 TheBestCoin developers
//
// Based on original work by sp and DJM34.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version. See COPYING for more details.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>


#include "cuda_helper.h"
#include "cuda_vector.h"
#include "cuda_utils.h"


#include "cuda_lyra2v2_var.h"


#define Nrow 8
#define Ncol 32


#define vectype uint28
__device__ vectype  *DMatrix;


#define TPB52 256
#define TPB50 64


#define u64type uint2
#define memshift 3


__device__ __forceinline__ void Gfunc_v35_VAR(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{

    a += b; d = eorswap32 (a, d);
    c += d; b ^= c; b = ROR24(b);
    a += b; d ^= a; d = ROR16(d);
    c += d; b ^= c; b = ROR2(b, 63);

}

__device__ __forceinline__ void round_lyra_v35_VAR(vectype* s)
{

    Gfunc_v35_VAR(s[0].x, s[1].x, s[2].x, s[3].x);
    Gfunc_v35_VAR(s[0].y, s[1].y, s[2].y, s[3].y);
    Gfunc_v35_VAR(s[0].z, s[1].z, s[2].z, s[3].z);
    Gfunc_v35_VAR(s[0].w, s[1].w, s[2].w, s[3].w);

    Gfunc_v35_VAR(s[0].x, s[1].y, s[2].z, s[3].w);
    Gfunc_v35_VAR(s[0].y, s[1].z, s[2].w, s[3].x);
    Gfunc_v35_VAR(s[0].z, s[1].w, s[2].x, s[3].y);
    Gfunc_v35_VAR(s[0].w, s[1].x, s[2].y, s[3].z);

}




__device__ __forceinline__ void reduceDuplex50_VAR(vectype state[4], uint32_t thread)
{
    const uint32_t ps1 = (Nrow * Ncol * memshift * thread);
    const uint32_t ps2 = (memshift * (Ncol - 1) + memshift * Ncol + Nrow * Ncol * memshift * thread);
#if __CUDA_ARCH__ != 500
    uint28 tmp[3];
#endif

    for (int i = 0; i < Ncol; i++)
    {
#if __CUDA_ARCH__ == 500

        const uint32_t s1 = ps1 + i*memshift;
        const uint32_t s2 = ps2 - i*memshift;

#pragma unroll
        for (int j = 0; j < 3; j++)
            state[j] ^= __ldg4(&(DMatrix + s1)[j]);

        round_lyra_v35_VAR(state);

#pragma unroll
        for (int j = 0; j < 3; j++)
            (DMatrix + s2)[j] = __ldg4(&(DMatrix + s1)[j]) ^ state[j];
#else
        const uint32_t s1 = ps1 + i*memshift;
        const uint32_t s2 = ps2 - i*memshift;
        tmp[0] = __ldg4(&(DMatrix + s1)[0]);
        tmp[1] = __ldg4(&(DMatrix + s1)[1]);
        tmp[2] = __ldg4(&(DMatrix + s1)[2]);
        state[0] ^= tmp[0];
        state[1] ^= tmp[1];
        state[2] ^= tmp[2];

        round_lyra_v35_VAR(state);

#pragma unroll
        for (int j = 0; j < 3; j++)
            (DMatrix + s2)[j] = tmp[j] ^ state[j];
#endif

    }
}
__device__ void reduceDuplexRowSetupV2_VAR(const int rowIn, const int rowInOut, const int rowOut, vectype state[4], uint32_t thread)
{

    int i, j;
    vectype state2[3],state1[3];

    const uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
    const uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
    const uint32_t ps3 = (memshift * (Ncol-1) + memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);

    for (i = 0; i < Ncol; i++)
    {
        const uint32_t s1 = ps1 + i*memshift;
        const uint32_t s2 = ps2 + i*memshift;
        const uint32_t s3 = ps3 - i*memshift;

#if __CUDA_ARCH__ == 500

#pragma unroll
        for (j = 0; j < 3; j++)
            state[j] = state[j] ^ (__ldg4(&(DMatrix + s1)[j]) + __ldg4(&(DMatrix + s2)[j]));

        round_lyra_v35_VAR(state);

#pragma unroll
        for (j = 0; j < 3; j++)
            state1[j] = __ldg4(&(DMatrix + s1)[j]);

#pragma unroll
        for (j = 0; j < 3; j++)
            state2[j] = __ldg4(&(DMatrix + s2)[j]);

#pragma unroll
        for (j = 0; j < 3; j++) 
            (DMatrix + s3)[j] =state[j]^state1[j];

#else

#pragma unroll
        for (j = 0; j < 3; j++)
            state1[j] = __ldg4(&(DMatrix + s1)[j]);

#pragma unroll
        for (j = 0; j < 3; j++)
            state2[j] = __ldg4(&(DMatrix + s2)[j]);

#pragma unroll
        for (j = 0; j < 3; j++)
            state[j] ^= state1[j] + state2[j];

        round_lyra_v35_VAR(state);

#pragma unroll
        for (j = 0; j < 3; j++)
            (DMatrix + s3)[j] = state1[j]^ state[j];;

#endif

        ((uint2*)state2)[0] ^= ((uint2*)state)[11];

#pragma unroll
        for (j = 0; j < 11; j++)
            ((uint2*)state2)[j+1] ^= ((uint2*)state)[j];

#pragma unroll
        for (j = 0; j < 3; j++)
            (DMatrix + s2)[j] = state2[j];
    }
}



__device__ void reduceDuplexRowtV2_VAR(const int rowIn, const int rowInOut, const int rowOut, vectype* state, uint32_t thread)
{
    int i,j;
    vectype state2[3];

    const uint32_t ps1 = (memshift * Ncol * rowIn + Nrow * Ncol * memshift * thread);
    const uint32_t ps2 = (memshift * Ncol * rowInOut + Nrow * Ncol * memshift * thread);
    const uint32_t ps3 = (memshift * Ncol * rowOut + Nrow * Ncol * memshift * thread);

    for (i = 0; i < Ncol; i++)
    {
        const uint32_t s1 = ps1 + i*memshift;
        const uint32_t s2 = ps2 + i*memshift;
        const uint32_t s3 = ps3 + i*memshift;

#pragma unroll
        for (j = 0; j < 3; j++)
            state2[j] = __ldg4(&(DMatrix + s2)[j]);

#pragma unroll
        for (j = 0; j < 3; j++)
            state[j] ^= __ldg4(&(DMatrix + s1)[j]) + state2[j];

        round_lyra_v35_VAR(state);

        ((uint2*)state2)[0] ^= ((uint2*)state)[11];
#pragma unroll
        for (j = 0; j < 11; j++)
            ((uint2*)state2)[j + 1] ^= ((uint2*)state)[j];

#if __CUDA_ARCH__ == 500

        if (rowInOut != rowOut)
        {
#pragma unroll
            for ( j = 0; j < 3; j++)
                (DMatrix + s3)[j] ^= state[j];

        }
        if (rowInOut == rowOut)
        {
#pragma unroll
            for (j = 0; j < 3; j++)
                state2[j] ^= state[j];
        }

#else

        if (rowInOut != rowOut)
        {
#pragma unroll
            for (j = 0; j < 3; j++)
                (DMatrix + s3)[j] ^= state[j];
        }
        else
        {
#pragma unroll
            for (j = 0; j < 3; j++)
                state2[j] ^= state[j];
        }

#endif

#pragma unroll
        for (j = 0; j < 3; j++)
            (DMatrix + s2)[j] = state2[j];
    }
}

__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(128, 1)
#endif
void lyra2v2_gpu_hash_32_VAR(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
    const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);


    vectype state[4];

    if (thread < threads)
    {
        const uint28 blake2b_IV[2] =
        {
            0xf3bcc908, 0x6a09e667,
            0x84caa73b, 0xbb67ae85,
            0xfe94f82b, 0x3c6ef372,
            0x5f1d36f1, 0xa54ff53a,
            0xade682d1, 0x510e527f,
            0x2b3e6c1f, 0x9b05688c,
            0xfb41bd6b, 0x1f83d9ab,
            0x137e2179, 0x5be0cd19
        };

        state[2] = ((blake2b_IV)[0]);
        state[3] = ((blake2b_IV)[1]);

        ((uint2*)state)[0] = __ldg(&outputHash[thread]);
        ((uint2*)state)[1] = __ldg(&outputHash[thread + threads]);
        ((uint2*)state)[2] = __ldg(&outputHash[thread + 2 * threads]);
        ((uint2*)state)[3] = __ldg(&outputHash[thread + 3 * threads]);

        state[1] = state[0];

        for (int i = 0; i<12; i++)
            round_lyra_v35_VAR(state);

        ((uint2*)state)[0].x ^= 0x20;
        ((uint2*)state)[1].x ^= 0x20;
        ((uint2*)state)[2].x ^= 0x20;
        ((uint2*)state)[3].x ^= 1;
        ((uint2*)state)[4].x ^= Nrow;
        ((uint2*)state)[5].x ^= Ncol;
        ((uint2*)state)[6].x ^= 0x80;
        ((uint2*)state)[7].y ^= 0x01000000;

        for (int i = 0; i<12; i++)
            round_lyra_v35_VAR(state);

        const uint32_t ps1 = (memshift * (Ncol - 1) + Nrow * Ncol * memshift * thread);

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
        for (int i = 0; i < Ncol; i++)
        {
            const uint32_t s1 = ps1 - memshift * i;
            DMatrix[s1] = state[0];
            DMatrix[s1+1] = state[1];
            DMatrix[s1+2] = state[2];
            round_lyra_v35_VAR(state);
        }

        reduceDuplex50_VAR(state, thread);

        int row = 2; //index of row to be processed
        int prev = 1; //index of prev (last row ever computed/modified)
        int rowa = 0; //index of row* (a previous row, deterministically picked during Setup and randomly picked while Wandering)
        int step = 1; //Visitation step (used during Setup and Wandering phases)
        int window = 2; //Visitation window (used to define which rows can be revisited during Setup)
        int gap = 1; //Modifier to the step, assuming the values 1 or -1
        do {
            reduceDuplexRowSetupV2_VAR(prev, rowa, row, state, thread);


            //updates the value of row* (deterministically picked during Setup))
            rowa = (rowa + step) & (window - 1);
            //update prev: it now points to the last row ever computed
            prev = row;
            //updates row: goes to the next row to be computed
            row++;

            //Checks if all rows in the window where visited.
            if (rowa == 0) {
                step = window + gap; //changes the step: approximately doubles its value
                window *= 2; //doubles the size of the re-visitation window
                gap = -gap; //inverts the modifier to the step
            }
        } while (row < Nrow);

        row = 0;
        step = Nrow / 2 - 1;
        do
        {
            rowa = ((uint2*)state)[0].x & (Nrow - 1);
            reduceDuplexRowtV2_VAR(prev, rowa, row, state, thread);
            prev = row;
            row = (row + step) & (Nrow - 1);
        }
        while (row != 0);

        const uint32_t shift = (memshift * Ncol * rowa + Nrow * Ncol * memshift * thread);

#pragma unroll
        for (int j = 0; j < 3; j++)
            state[j] ^= __ldg4(&(DMatrix + shift)[j]);

        for (int i = 0; i < 12; i++)
            round_lyra_v35_VAR(state);

        outputHash[thread] = ((uint2*)state)[0];
        outputHash[thread + threads] = ((uint2*)state)[1];
        outputHash[thread + 2 * threads] = ((uint2*)state)[2];
        outputHash[thread + 3 * threads] = ((uint2*)state)[3];

    } //thread
}


__host__
void lyra2v2_cpu_init_VAR(int thr_id, uint32_t threads,uint64_t *hash)
{
    cudaMemcpyToSymbol(DMatrix, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
}



__host__
void lyra2v2_cpu_hash_32_VAR(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb)
{
    dim3 grid((threads + tpb - 1) / tpb);
    dim3 block(tpb);

    lyra2v2_gpu_hash_32_VAR<<<grid, block>>>(threads, startNounce, (uint2*)d_outputHash);
}

