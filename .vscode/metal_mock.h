#pragma once

// Mock Metal keywords
#define kernel
#define device
#define constant
#define threadgroup

namespace metal {
    typedef unsigned int uint;
    typedef float float4 __attribute__((ext_vector_type(4)));
    typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
    typedef unsigned int uint3 __attribute__((ext_vector_type(3)));
    
    inline float rsqrt(float x) { return 0.0f; }
    inline float exp(float x) { return 0.0f; }
    
    inline float4 exp(float4 x) { return x; }
    inline float4 rsqrt(float4 x) { return x; }

    namespace mem_flags {
        constexpr int mem_threadgroup = 0;
    }
    inline void threadgroup_barrier(int flags) {}
}
