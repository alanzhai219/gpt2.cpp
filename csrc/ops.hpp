#pragma once

#include <vector>

#include "tensor.hpp"

namespace gpt2 {
namespace ops {

/*
 *  wte: [vocab_size, n_embed]
*/
Tensor token_embed(const Tensor& wte, const std::vector<int>& ids);

/*
 *  wpt: [n_positions, n_embed]
 * */
Tensor position_embed(const Tensor& wpe, size_t start, size_t S);

Tensor add(const Tensor& a, const Tensor& b);
void add_(Tensor& a, const Tensor& b);

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5);

Tensor matmul_2d(const Tensor& a, const Tensor& b);

void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v);

Tensor split_head(const float* x_ptr, size_t S, size_t n_head, size_t head_dim);

Tensor merge_head(const Tensor& x);

Tensor softmax(const Tensor& x);

void gelu_(Tensor& x);

Tensor matmul_3d(const Tensor& a, const Tensor& b);

// Q [H, S, D] x K_cache^T [H, D, T] -> [H, S, T].
// K cache storage is head-major [H, Tmax, D], with valid length T.
Tensor matmul_qk_cache(const Tensor& query,
					   const std::vector<float>& key_cache,
					   size_t num_heads,
					   size_t max_cache_len,
					   size_t head_dim,
					   size_t cache_len);

// Scores [H, S, T] x V_cache [H, T, D] -> [H, S, D].
// V cache storage is head-major [H, Tmax, D], with valid length T.
Tensor matmul_av_cache(const Tensor& scores,
					   const std::vector<float>& value_cache,
					   size_t num_heads,
					   size_t max_cache_len,
					   size_t head_dim,
					   size_t cache_len);

Tensor causal_mask(const Tensor& a, size_t n_past);

Tensor scale(const Tensor& x, const float s);

Tensor transpose_2d(const Tensor& x);
Tensor transpose_3d(const Tensor& x);

Tensor gemv(const Tensor& x, const float* v);

}
}
