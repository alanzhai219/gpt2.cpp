#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <iostream>

#include "modeling.hpp"
#include "ops.hpp"

namespace gpt2 {

void GPT2::init_op() {
    m_llm_context.device = llm_bricks::DeviceType::cpu;
    m_llm_context.backend = llm_bricks::Backend::avx2;
    m_llm_add = std::make_unique<llm_bricks::Add>(m_llm_context);
    m_llm_layer_norm = std::make_unique<llm_bricks::LayerNorm>(m_llm_context);
    m_llm_scale = std::make_unique<llm_bricks::Scale>(m_llm_context);
    m_llm_softmax = std::make_unique<llm_bricks::Softmax>(m_llm_context);
    m_llm_gelu = std::make_unique<llm_bricks::Gelu>(m_llm_context);
}

std::vector<float> GPT2::forward(const std::vector<int>& tokens, size_t n_past) {
    if (tokens.empty()) throw std::invalid_argument("forward: tokens must not be empty");
    if (n_past != m_kv_cache.get_current_cache_len()) {
        throw std::runtime_error("forward: n_past does not match KV cache length");
    }
    const size_t seq = tokens.size();
    const size_t n_embd = m_w.config.n_embd;

    // [S, n_embd]
    Tensor tok_emb = ops::token_embed(m_w.wte, tokens);
    // [S, n_embd]
    Tensor pos_emb = ops::position_embed(m_w.wpe, n_past, seq); 

    // [S, n_embd]
    Tensor x(tok_emb.shape());
    llm_bricks::AddParams add_params(tok_emb, pos_emb, x);
    m_llm_add->set_input(add_params);
    (void)m_llm_add->infer();

    for (size_t layer = 0; layer < m_w.config.n_layer; ++layer) {
        transfomer_layer(layer, x, n_past);
    }
    // [S, n_embd]
    Tensor normalized(x.shape());
    llm_bricks::LayerNormParams final_layer_norm_params(x, m_w.ln_f_w, m_w.ln_f_b, 1.0e-5F, normalized);
    m_llm_layer_norm->set_input(final_layer_norm_params);
    (void)m_llm_layer_norm->infer();
    // select the last row of [S, n_embd] => [n_embd]
    const float* last = normalized.ptr() + (seq - 1) * n_embd;
    // [n_vocab] = [n_vocab, n_emdb] @ [n_embd]
    Tensor logits = ops::gemv(m_w.wte, last);
    return std::vector<float>(logits.ptr(), logits.ptr() + logits.numel());
}

void GPT2::transfomer_layer(size_t layer_id, Tensor& x, size_t n_past) {
    attn(layer_id, x, n_past);
    mlp(layer_id, x, n_past);
}

void GPT2::attn(size_t layer_id, Tensor& x, size_t n_past) {
    const LayerWeights& layer = m_w.layers.at(layer_id);
    const size_t seq = x.dim(0);

    // [S, n_embd]
    Tensor normalized(x.shape());
    llm_bricks::LayerNormParams attn_layer_norm_params(x, layer.ln_1_w, layer.ln_1_b, 1.0e-5F, normalized);
    m_llm_layer_norm->set_input(attn_layer_norm_params);
    (void)m_llm_layer_norm->infer();
    // [S, 3xn_embd] = [S, n_embd] @ [n_embd, 3xn_embd]
    Tensor qkv = ops::matmul_2d(normalized, layer.attn_c_attn_w);
    ops::add_(qkv, layer.attn_c_attn_b);

    Tensor q, k, v;
    // q,k,v: [S, n_embd]
    ops::split_qkv(qkv, q, k, v);
    auto& kv_cache = m_kv_cache.get_layer_cache(layer_id);
    // Store only the new token block once: [S, C] -> cache [H, Tmax, D].
    kv_cache.append(k.ptr(), v.ptr(), seq);

    // Q: [H, S, D]. K/V stay in cache as [H, Tmax, D] and are read by the MatMul operators.
    Tensor query = ops::split_head(q.ptr(), seq, m_w.config.n_head, m_hidden_dim);
    // [H, S, T] = [H, S, D] @ K_cache^T[H, D, T]
    Tensor scores = ops::matmul_qk_cache(query,
                                          kv_cache.get_kcache(),
                                          kv_cache.num_heads(),
                                          kv_cache.max_cache_len(),
                                          kv_cache.head_size(),
                                          kv_cache.get_cache_len());
    Tensor scaled_scores(scores.shape());
    llm_bricks::ScaleParams scale_params(scores, m_scale, scaled_scores);
    m_llm_scale->set_input(scale_params);
    (void)m_llm_scale->infer();
    scores = std::move(scaled_scores);
    scores = ops::causal_mask(scores, n_past);
    Tensor probabilities(scores.shape());
    llm_bricks::SoftmaxParams softmax_params(scores, probabilities);
    m_llm_softmax->set_input(softmax_params);
    (void)m_llm_softmax->infer();
    scores = std::move(probabilities);
    // [H, S, D] = [H, S, T] @ V_cache[H, T, D]
    Tensor attended = ops::matmul_av_cache(scores,
                                            kv_cache.get_vcache(),
                                            kv_cache.num_heads(),
                                            kv_cache.max_cache_len(),
                                            kv_cache.head_size(),
                                            kv_cache.get_cache_len());
    // [S, n_embd]
    Tensor attended_merge = ops::merge_head(attended);
    // [S, n_embd] = [S, n_embd] @ [n_embd, n_embd]
    Tensor projection = ops::matmul_2d(attended_merge, layer.attn_c_proj_w);
    ops::add_(projection, layer.attn_c_proj_b);
    ops::add_(x, projection);
}

void GPT2::mlp(size_t layer_id, Tensor& x, size_t n_past) {
    (void)n_past;
    const LayerWeights& layer = m_w.layers.at(layer_id);
    Tensor ln2(x.shape());
    llm_bricks::LayerNormParams mlp_layer_norm_params(x, layer.ln_2_w, layer.ln_2_b, 1.0e-5F, ln2);
    m_llm_layer_norm->set_input(mlp_layer_norm_params);
    (void)m_llm_layer_norm->infer();
    // [S, 4xn_embd] = [S, n_embd] @ [n_embd, 4xn_embd]
    Tensor hidden = ops::matmul_2d(ln2, layer.mlp_c_fc_w);
    ops::add_(hidden, layer.mlp_c_fc_b);
    llm_bricks::GeluParams gelu_params(hidden);
    m_llm_gelu->set_input(gelu_params);
    (void)m_llm_gelu->infer();
    // [S, n_embd] = [S, 4xn_embd] @ [4xn_embd, n_embd]
    Tensor output = ops::matmul_2d(hidden, layer.mlp_c_proj_w);
    ops::add_(output, layer.mlp_c_proj_b);
    ops::add_(x, output);
}

int GPT2::temperature_search(const std::vector<float>& logits, float temperature,
                             int top_k) {
    if (logits.empty()) throw std::invalid_argument("temperature_search: empty logits");
    if (temperature <= 0.0F || top_k <= 0) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const size_t count = std::min(static_cast<size_t>(top_k), logits.size());
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    ids.resize(count);

    const float max_logit = logits[ids.front()];
    std::vector<double> probabilities;
    probabilities.reserve(count);
    for (int id : ids) probabilities.push_back(std::exp((logits[id] - max_logit) / temperature));
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return ids[distribution(m_rnd)];
}

std::string GPT2::generate(const tk::Tokenizer& tokenizer, const std::string& prompt,
                           int max_tokens, float temperature, int top_k, uint64_t seed) {
    constexpr int end_of_text = 50256;
    m_rnd.seed(seed);
    reset_cache();
    std::vector<int> ids = tokenizer.encode(prompt);
    if (ids.empty()) ids.push_back(end_of_text);
    if (max_tokens > 0) ids.reserve(ids.size() + static_cast<size_t>(max_tokens));

    if (is_profile) {
        t.start();
    }
    std::vector<float> logits = forward(ids, 0);
    if (is_profile) {
        t.stop();
        size_t first_token_latency = t.elapsed();
        std::cout << "[first token latency] " << first_token_latency << " ms\n";
    }
    size_t search_time = 0;
    size_t next_token_time = 0;
    for (int step = 0; step < max_tokens; ++step) {
        if (is_profile) {
            t.start();
        }
        const int next = temperature_search(logits, temperature, top_k);
        if (is_profile) {
            t.stop();
            search_time += t.elapsed();
        }
        ids.push_back(next);
        if (next == end_of_text) {
            break;
        }
        if (is_profile) {
            t.start();
        }
        logits = forward({next}, m_kv_cache.get_current_cache_len());
        if (is_profile) {
            t.stop();
            next_token_time += t.elapsed();
        }
    }
    if (is_profile) {
        std::cout << "[next token latency] " << next_token_time / max_tokens << " ms\n";
        std::cout << "[search token latency] " << search_time / max_tokens << " ms\n";
    }
    return tokenizer.decode(ids);
}

}  // namespace gpt2
