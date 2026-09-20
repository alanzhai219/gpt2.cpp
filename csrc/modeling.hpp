#pragma once

#include <vector>
#include <string>
#include <random>
#include <cstdint>
#include <cmath>

#include "kvcache.hpp"
#include "tensor.hpp"
#include "weights.hpp"
#include "tokenizer.hpp"
#include "timer.hpp"

namespace gpt2 {

class GPT2 {
public:
    explicit GPT2(GPT2Weights m)
        : m_w(std::move(m)),
          m_kv_cache(m_w.config.n_layer,
                     m_w.config.n_embd / m_w.config.n_head,
                     m_w.config.n_head,
                     m_w.config.n_positions) {
        if (m_w.config.n_embd % m_w.config.n_head != 0) {
            throw std::invalid_argument("n_embd must be divisible by n_head");
        }
        m_hidden_dim = m_w.config.n_embd / m_w.config.n_head;
        m_scale = 1.0F / std::sqrt(static_cast<float>(m_hidden_dim));
        const char* benchmark = std::getenv("ENABLE_BENCHMARK");
        is_profile = benchmark != nullptr && std::atoi(benchmark) > 0;
    }

    std::vector<float> forward(const std::vector<int>& tokens, size_t n_past);
    void transfomer_layer(size_t layer_id, Tensor& x, size_t n_past);
    void attn(size_t layer_id, Tensor& x, size_t n_past);
    void mlp(size_t layer_id, Tensor& x, size_t n_past);

    void reset_cache() { m_kv_cache.reset(); }

    int temperature_search(const std::vector<float>& logits, float temperature, int top_k);
    std::string generate(const tk::Tokenizer& token, const std::string& prompt,
                         int max_tokens, float temperature, int top_k, uint64_t seed);
    
private:
    GPT2Weights m_w;
    size_t m_hidden_dim;
    KVCache m_kv_cache;
    float m_scale = 0.0F;
    std::mt19937_64 m_rnd{42};
    bool is_profile = false;
    timer t;
};

}   // namespace gpt2
