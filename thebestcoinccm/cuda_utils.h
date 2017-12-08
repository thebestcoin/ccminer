// Copyright (c) 2017 TheBestCoin developers
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version. See COPYING for more details.

#pragma once


#include <cuda_runtime.h>
#include "log.h"

inline uint32_t roundown2(uint32_t v)
{
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;

	return (v + 1) >> 1;
}

inline bool CudaCheck(cudaError_t err, const char * FUNC, int LINE)
{
    if (err != cudaSuccess)
    {
        applog(LOG_ERR, "ERROR: Cuda error in func '%s' at line %i : %s", FUNC, LINE, cudaGetErrorString(err));
        return false;
    }
    return true;
}


#define ON_CUDA_ERROR_(expr, action) \
    do \
    { \
        if (!CudaCheck((expr), __FUNCTION__, __LINE__)) \
            { action; } \
    } \
    while (false)

#define ON_CUDA_ERROR_EXIT_(expr, exitval)  ON_CUDA_ERROR_(expr, exit(exitval))
#define ON_CUDA_ERROR_EXIT(exitval)         ON_CUDA_ERROR_(cudaGetLastError(), exit(exitval))

#define ON_CUDA_ERROR_RETURN_(expr, retval)  ON_CUDA_ERROR_(expr, return (retval))
#define ON_CUDA_ERROR_RETURN(retval)         ON_CUDA_ERROR_(cudaGetLastError(), return (retval))

#define ON_CUDA_ERROR_BREAK_(expr)  ON_CUDA_ERROR_(expr, break)
#define ON_CUDA_ERROR_BREAK         ON_CUDA_ERROR_(cudaGetLastError(), break)


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
# define _ALIGN(x) __align__(x)
#elif _MSC_VER
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#endif


