#pragma once

#include <cstddef>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace gpt2 {

struct KVCACHE {
    KVCACHE(size_t n_layer = 0, size_t n_cache_len = 0)
      : m_layer(n_layer), m_cache_len(n_cache_len) {
        reset();
    }

    void reset() {
        m_kcache.assign(m_layer, {});
        m_vcache.assign(m_layer, {});
        m_cache_len = 0;
    }

    std::vector<float>& k_get_layer(size_t layer_id) {
        return m_kcache[layer_id];
    }

    std::vector<float>& v_get_layer(size_t layer_id) {
        return m_vcache[layer_id];
    }

    size_t get_cache_len() const {
        return m_cache_len;
    }

    void set_cache_len(size_t cache_len) {
        m_cache_len = cache_len;
    }

    std::vector<std::vector<float>> m_kcache;
    std::vector<std::vector<float>> m_vcache;
    size_t m_layer;
    size_t m_cache_len;
};

// Each layer has its own KV cache, which is a 3D tensor of shape [H, S, D].
// head 0:
//  S0: [D]
//  S1: [D]
//  ...
//  Sn: [D]
// head 1:
//  S0: [D]
//  S1: [D]
//  ...
//  Sn: [D]
struct LayerKVCache {
    LayerKVCache(size_t head_size, size_t num_heads, size_t max_cache_len)
      : m_head_size(head_size),
        m_num_heads(num_heads),
        m_max_cache_len(max_cache_len),
        m_cur_cache_len(0) {
        m_kcache.resize(num_heads * max_cache_len * head_size);
        m_vcache.resize(num_heads * max_cache_len * head_size);
    }

    void reset() {
        m_cur_cache_len = 0;
    }

    // [H, S, D] flattened to [H * S * D]
    const std::vector<float>& get_kcache() const {
        return m_kcache;
    }

    // [H, S, D] flattened to [H * S * D]
    const std::vector<float>& get_vcache() const {
        return m_vcache;
    }

    size_t get_cache_len() const {
        return m_cur_cache_len;
    }

    size_t head_size() const {
        return m_head_size;
    }

    size_t num_heads() const {
        return m_num_heads;
    }

    size_t max_cache_len() const {
        return m_max_cache_len;
    }

    // Append token-major K/V data [S, H * D] into head-major storage [H, Tmax, D].
    void append(const float* new_k, const float* new_v, size_t new_len) {
        if (m_cur_cache_len + new_len > m_max_cache_len) {
            throw std::runtime_error("Exceeded maximum cache length");
        }
        const size_t hidden_size = m_num_heads * m_head_size;
        for (size_t h = 0; h < m_num_heads; ++h) {
            for (size_t s = 0; s < new_len; ++s) {
                const size_t src_offset = s * hidden_size + h * m_head_size;
                const size_t dst_offset = (h * m_max_cache_len + m_cur_cache_len + s) * m_head_size;
                std::memcpy(m_kcache.data() + dst_offset,
                            new_k + src_offset,
                            m_head_size * sizeof(float));
                std::memcpy(m_vcache.data() + dst_offset,
                            new_v + src_offset,
                            m_head_size * sizeof(float));
            }
        }
        m_cur_cache_len += new_len;
    }

private:
    size_t m_head_size;
    size_t m_num_heads;
    size_t m_max_cache_len;
    size_t m_cur_cache_len;
    std::vector<float> m_kcache; // [H, S, D]
    std::vector<float> m_vcache; // [H, S, D]
};

class KVCache {
public:
    KVCache() = default;

        KVCache(size_t num_layers, size_t head_size, size_t num_heads, size_t max_cache_len)
            : m_num_layers(num_layers) {
                m_layer_caches.reserve(num_layers);
                for (size_t layer = 0; layer < num_layers; ++layer) {
                        m_layer_caches.emplace_back(head_size, num_heads, max_cache_len);
                }
    }

    LayerKVCache& get_layer_cache(size_t layer_id) {
        return m_layer_caches[layer_id];
    }

    void reset() {
        for (auto& layer_cache : m_layer_caches) {
            layer_cache.reset();
        }
    }

    size_t get_current_cache_len() const {
        if (m_layer_caches.empty()) return 0;
        return m_layer_caches[0].get_cache_len();
    }

    size_t get_num_layers() const {
        return m_num_layers;
    }

private:
    size_t m_num_layers = 0;
    std::vector<LayerKVCache> m_layer_caches;
};
}