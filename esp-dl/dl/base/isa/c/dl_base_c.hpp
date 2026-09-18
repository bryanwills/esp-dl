#pragma once

// Declarations only. This header is on the public include path (via
// dl_base_isa.hpp on plain-C targets) and must not pull in the generated
// dl_compile_config.h or the heavy dl_base_conv_args.hpp. ConvArgsType is
// used by reference/pointer only, so a forward declaration suffices;
// implementations (dl_c_*.cpp) include the full definition themselves.

#include "dl_define_private.hpp"
#include <cstdint>

namespace dl {
namespace base {
struct ConvArgsType;
} // namespace base
} // namespace dl

extern "C" {

void dl_c_s16_conv2d_11cn(DL_S16_BUFFER_TYPE *buffer, int16_t *input, const dl::base::ConvArgsType &args);
void dl_c_s16_conv2d_hwcn(DL_S16_BUFFER_TYPE *buffer, int16_t *input, const dl::base::ConvArgsType &args);
void dl_c_s8_conv2d_11cn(int32_t *buffer, int8_t *input, const dl::base::ConvArgsType &args);
void dl_c_s8_conv2d_hwcn(int32_t *buffer, int8_t *input, const dl::base::ConvArgsType &args);
void dl_c_w8a16_conv2d_11cn(DL_S16_BUFFER_TYPE *buffer, int16_t *input, const dl::base::ConvArgsType &args);
void dl_c_w8a16_conv2d_hwcn(DL_S16_BUFFER_TYPE *buffer, int16_t *input, const dl::base::ConvArgsType &args);

void dl_c_s16_depthwise_conv2d_hwc1(DL_S16_BUFFER_TYPE *buffer, int16_t *input, const dl::base::ConvArgsType &args);
void dl_c_s8_depthwise_conv2d_hwc1(int32_t *buffer, int8_t *input, const dl::base::ConvArgsType &args);

void dl_c_s16_tail_linear(int16_t *output, DL_S16_BUFFER_TYPE *buffer, const dl::base::ConvArgsType &args);
void dl_c_s16_tail_relu(int16_t *output, DL_S16_BUFFER_TYPE *buffer, const dl::base::ConvArgsType &args);
void dl_c_s16_tail_bias_linear(int16_t *output, DL_S16_BUFFER_TYPE *buffer, const dl::base::ConvArgsType &args);
void dl_c_s16_tail_bias_relu(int16_t *output, DL_S16_BUFFER_TYPE *buffer, const dl::base::ConvArgsType &args);

void dl_c_s8_tail_linear(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_relu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_leakyrelu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_prelu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_bias_linear(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_bias_relu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_bias_per_ch_linear(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_bias_per_ch_relu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_bias_per_ch_leakyrelu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_tail_bias_per_ch_prelu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
void dl_c_s8_dw_tail_bias_per_ch_prelu(int8_t *output, int32_t *buffer, const dl::base::ConvArgsType &args);
}
