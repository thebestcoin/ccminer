// Copyright (c) 2017 TheBestCoin developers
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version. See COPYING for more details.

#pragma once

// Block length required so Blake2's Initialization Vector (IV) is not overwritten (THIS SHOULD NOT BE MODIFIED)
#define BLOCK_LEN_BLAKE2_SAFE_INT64 8                                   // 512 bits (=64 bytes, =8 uint64_t)
#define BLOCK_LEN_BLAKE2_SAFE_BYTES (BLOCK_LEN_BLAKE2_SAFE_INT64 * 8)   // same as above, in bytes

// Default block lenght: 768 bits
#define BLOCK_LEN_INT64 12								// Block length: 768 bits (=96 bytes, =12 uint64_t)
#define BLOCK_LEN_BYTES (BLOCK_LEN_INT64 * 8)			// Block length, in bytes

#define ROW_LEN_INT64 (BLOCK_LEN_INT64 * 32)			// Total length of a row
#define ROW_LEN_BYTES (ROW_LEN_INT64 * 8)				// Number of bytes per row

void lyra2v2_cpu_init_VAR(int thr_id, uint32_t threads, uint64_t *hash);
void lyra2v2_cpu_hash_32_VAR(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb);
