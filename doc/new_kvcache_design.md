我先确认 `KVCACHE` 的构造位置和模型配置中的最大上下文长度，避免给出和现有初始化接口不匹配的 head-major cache 设计。

Searched for regex `KVCACHE|m_kv_cache|n_ctx|n_positions` (`gpt2.cpp`), 80 results

Read [](file:///home/xiuchuan/workspace/gpt2.cpp/csrc/modeling.hpp)

Read [](file:///home/xiuchuan/workspace/gpt2.cpp/csrc/config.hpp)

把 KV cache 改为 head-major 的关键是：**只在新 K/V 写入 cache 时做一次 `[S,C] → [H,S,D]` 重排；之后 attention 直接按 `[H,T,D]` 读取，不再重排历史数据。**

设：

$$
C = H \times D
$$

其中：

- $C=\text{n\_embd}$
- $H=\text{n\_head}$
- $D=\text{head\_dim}$
- $T_{\max}=\text{n\_positions}$

当前配置已给出最大长度 `n_positions`，见 config.hpp。GPT-2 small 对应：

```text
H = 12
D = 64
C = 768
Tmax = 1024
```

---

## 1. 当前 token-major cache 的存取方式

当前一层 K cache 的内存语义是：

```text
K cache: [T, C]
```

按 token 连续：

```text
token 0: [head0 D floats][head1 D floats]...[headH-1 D floats]
token 1: [head0 D floats][head1 D floats]...[headH-1 D floats]
...
```

写入很方便：

```cpp
k_cache.insert(k_cache.end(), k.ptr(), k.ptr() + k.numel());
```

见 modeling.cpp。

但读取并不适合 attention。对某一个 head $h$，它的历史 K 实际分散在每个 token row 内：

```text
K[0, h, :]
K[1, h, :]
K[2, h, :]
...
```

因此当前代码不得不每轮都执行：

```cpp
Tensor key = ops::split_head(k_cache.data(), total, n_head, head_dim);
Tensor value = ops::split_head(v_cache.data(), total, n_head, head_dim);
```

见 modeling.cpp。

这会把整个历史 cache 转成 `[H,T,D]` 副本。

---

## 2. 新布局：每层保存 `[H, Tmax, D]`

建议每个 layer 分别预分配：

```text
K: [H, Tmax, D]
V: [H, Tmax, D]
```

物理内存顺序：

```text
head 0:
  token 0: [D]
  token 1: [D]
  ...
  token Tmax - 1: [D]

head 1:
  token 0: [D]
  token 1: [D]
  ...
```

线性 offset 是：

$$
\text{offset}(h,t,d)=(h \times T_{\max}+t)\times D+d
$$

也就是：

```cpp
size_t offset(size_t head, size_t token, size_t dim) const {
    return (head * max_tokens + token) * head_dim + dim;
}
```

这满足 decode attention 的读取模式：固定一个 `head` 后，历史 token 的 K/V 是紧凑连续的二维矩阵。

```text
key_for_head(h):   [T, D]
value_for_head(h): [T, D]
```

---

## 3. 推荐的 cache 数据结构

建议不要再将 `m_kcache`、`m_vcache` 作为 `vector<vector<float>>` 的裸数组暴露出来。让 cache 自己拥有布局、边界和读写逻辑。

可将 kvcache.hpp 改造成以下结构：

```cpp
#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace gpt2 {

struct LayerKVCache {
    LayerKVCache() = default;

    LayerKVCache(size_t n_head, size_t max_tokens, size_t head_dim)
        : n_head(n_head),
          max_tokens(max_tokens),
          head_dim(head_dim),
          key(n_head * max_tokens * head_dim),
          value(n_head * max_tokens * head_dim) {}

    size_t offset(size_t h, size_t t, size_t d = 0) const {
        return (h * max_tokens + t) * head_dim + d;
    }

    float* key_ptr(size_t h, size_t t = 0) {
        return key.data() + offset(h, t);
    }

    const float* key_ptr(size_t h, size_t t = 0) const {
        return key.data() + offset(h, t);
    }

    float* value_ptr(size_t h, size_t t = 0) {
        return value.data() + offset(h, t);
    }

    const float* value_ptr(size_t h, size_t t = 0) const {
        return value.data() + offset(h, t);
    }

    size_t n_head = 0;
    size_t max_tokens = 0;
    size_t head_dim = 0;
    std::vector<float> key;    // [H, Tmax, D]
    std::vector<float> value;  // [H, Tmax, D]
};

class KVCache {
public:
    KVCache() = default;

    KVCache(size_t n_layer, size_t n_head, size_t max_tokens, size_t head_dim)
        : m_layers(n_layer, LayerKVCache{n_head, max_tokens, head_dim}),
          m_max_tokens(max_tokens) {}

    void reset() {
        // 保留已分配的容量；下一轮 generation 从位置 0 覆盖写入。
        m_cache_len = 0;
    }

    size_t cache_len() const {
        return m_cache_len;
    }

    size_t max_tokens() const {
        return m_max_tokens;
    }

    void set_cache_len(size_t cache_len) {
        if (cache_len > m_max_tokens) {
            throw std::out_of_range("KV cache capacity exceeded");
        }
        m_cache_len = cache_len;
    }

    LayerKVCache& layer(size_t layer_id) {
        return m_layers.at(layer_id);
    }

    const LayerKVCache& layer(size_t layer_id) const {
        return m_layers.at(layer_id);
    }

private:
    std::vector<LayerKVCache> m_layers;
    size_t m_max_tokens = 0;
    size_t m_cache_len = 0;
};

}  // namespace gpt2
```

这里最重要的行为变化是 `reset()`：

```cpp
m_cache_len = 0;
```

而不是当前：

```cpp
m_kcache.assign(m_layer, {});
m_vcache.assign(m_layer, {});
```

当前实现见 kvcache.hpp。新设计会保留分配好的 buffer，下一次生成直接覆盖，从而避免重复分配。

---

## 4. 初始化方式

`GPT2` 已经有下列全部参数：

- `m_w.config.n_layer`
- `m_w.config.n_head`
- `m_w.config.n_positions`
- `m_hidden_dim`

见 modeling.hpp。

因此初始化可以改为：

```cpp
m_hidden_dim = m_w.config.n_embd / m_w.config.n_head;

m_kv_cache = KVCache(
    m_w.config.n_layer,
    m_w.config.n_head,
    m_w.config.n_positions,
    m_hidden_dim);
```

还应在构造时验证：

```cpp
if (m_w.config.n_embd % m_w.config.n_head != 0) {
    throw std::invalid_argument("n_embd must be divisible by n_head");
}
```

---

## 5. 如何存：仅重排“新 token”的 K/V

QKV projection 和 `split_qkv()` 可以先不改：

```text
K new: [S,C]
V new: [S,C]
```

见 modeling.cpp。

当本次输入长度为 $S$、历史长度为 `n_past` 时，把当前 K/V 写入：

```text
cache K: [H,Tmax,D]
             ^ 写入范围 [n_past, n_past + S)
```

实现：

```cpp
void append_kv(LayerKVCache& cache,
               const Tensor& k,   // [S, C]
               const Tensor& v,   // [S, C]
               size_t n_past) {
    const size_t seq = k.dim(0);
    const size_t hidden = k.dim(1);

    if (v.shape() != k.shape()) {
        throw std::invalid_argument("append_kv: K/V shape mismatch");
    }
    if (hidden != cache.n_head * cache.head_dim) {
        throw std::invalid_argument("append_kv: hidden size mismatch");
    }
    if (n_past + seq > cache.max_tokens) {
        throw std::out_of_range("append_kv: context length exceeded");
    }

    for (size_t h = 0; h < cache.n_head; ++h) {
        const size_t hidden_offset = h * cache.head_dim;

        for (size_t s = 0; s < seq; ++s) {
            const float* src_k = k.ptr() + s * hidden + hidden_offset;
            const float* src_v = v.ptr() + s * hidden + hidden_offset;

            float* dst_k = cache.key_ptr(h, n_past + s);
            float* dst_v = cache.value_ptr(h, n_past + s);

            std::copy_n(src_k, cache.head_dim, dst_k);
            std::copy_n(src_v, cache.head_dim, dst_v);
        }
    }
}
```

### 写入前后的位置关系

假设：

```text
H = 2
D = 4
S = 2
n_past = 3
```

输入 `k` 的逻辑形状：

```text
k: [2, 8]

k[0] = [h0:d0..d3][h1:d0..d3]
k[1] = [h0:d0..d3][h1:d0..d3]
```

写入后：

```text
cache K[0, 3, :] = k[0, 0:4]
cache K[1, 3, :] = k[0, 4:8]

cache K[0, 4, :] = k[1, 0:4]
cache K[1, 4, :] = k[1, 4:8]
```

所以只重排本次新产生的 $S\times C$ 个元素一次。

- prefill：写入 $S\times C$，这是必要成本；
- decode：$S=1$，每层仅搬运 $C$ 个 K 和 $C$ 个 V；
- 绝不再搬运整个 $T\times C$ 历史 cache。

---

## 6. 如何取：按 head 直接获取连续行

新的 attention 不应再调用：

```cpp
ops::split_head(k_cache.data(), total, ...);
ops::split_head(v_cache.data(), total, ...);
ops::transpose_3d(key);
```

而是直接得到：

```cpp
const float* k_head = layer_cache.key_ptr(h, 0);    // 逻辑 [total, D]
const float* v_head = layer_cache.value_ptr(h, 0);  // 逻辑 [total, D]
```

对固定 head，`k_head` 指向：

```text
K[h, 0, 0]
```

且接下来 `total * head_dim` 个元素连续：

```text
K[h, 0, :]
K[h, 1, :]
...
K[h, total - 1, :]
```

### Decode 的 QK 计算

decode 时 `seq = 1`，当前 Q 的第 `h` 个 head 位于：

```cpp
const float* q_head = q.ptr() + h * head_dim;
```

然后不转置 K，直接点积：

```cpp
for (size_t t = 0; t < total; ++t) {
    const float* k_token = k_head + t * head_dim;

    float dot = 0.0F;
    for (size_t d = 0; d < head_dim; ++d) {
        dot += q_head[d] * k_token[d];
    }
    scores[t] = dot * scale;
}
```

数学上等价于：

$$
\operatorname{scores}[h,t]
=
\frac{
\sum_{d=0}^{D-1}
Q[h,d]\cdot K[h,t,d]
}{
\sqrt D
}
$$

这里的 $K^T$ 只是矩阵乘法的数学记号，**不应作为真实 Tensor 物化**。

### Softmax 后读取 V

```cpp
std::fill(out_head, out_head + head_dim, 0.0F);

for (size_t t = 0; t < total; ++t) {
    const float* v_token = v_head + t * head_dim;
    const float weight = scores[t];

    for (size_t d = 0; d < head_dim; ++d) {
        out_head[d] += weight * v_token[d];
    }
}
```

数学形式：

$$
O[h,d]=\sum_{t=0}^{T-1}\operatorname{softmax}(\text{scores})[t]\cdot V[h,t,d]
$$

---

## 7. Decode 专用 attention 接口

最实用的第一版可以只优化最常见的 `seq == 1` 路径：

```cpp
void attention_decode(const Tensor& q,              // [1, C]
                      const LayerKVCache& cache,    // K/V [H, Tmax, D]
                      size_t total,
                      float scale,
                      std::vector<float>& scores,   // 至少 T 大小，可复用
                      Tensor& output);              // [1, C]
```

输出直接保持 token-major：

```text
output: [1,C]
```

每个 head 的输出直接写到：

```cpp
float* out_head = output.ptr() + h * head_dim;
```

这一步已经融合了原来的：

```text
split_head(q)
split_head(full K cache)
split_head(full V cache)
transpose_3d(key)
QK matmul
scale
causal mask
softmax
AV matmul
merge_head
```

并保持 `projection` 的输入格式不变：

```text
attention output: [1,C]
attn_c_proj_w:    [C,C]
```

因而可以继续复用现有的：

```cpp
Tensor projection = ops::matmul_2d(attention_output, layer.attn_c_proj_w);
```

---

## 8. Prefill 与 decode 应拆成两个路径

这是设计上很重要的一点。

### Prefill：`seq > 1`

输入 prompt 时，例如：

```text
S = 128
n_past = 0
```

仍需要 causal attention，因为 token $s$ 不能看未来 token：

$$
t \leq n_{\text{past}}+s
$$

这时可先保留当前 `matmul_3d + causal_mask + softmax + matmul_3d`，但：

- K/V 仍写入新的 head-major cache；
- `key` / `value` 不再从完整 cache 重新 `split_head`；
- K 的转置可以由 `matmul_qk_transposed()` 取代；
- 输出可直接写为 `[S,C]`，消除 `merge_head()`。

### Decode：`seq == 1`

```text
S = 1
```

此时 cache 中只有历史 token 与当前 token，完全没有未来 token。因此无须执行 `causal_mask()`。

当前 modeling.cpp 无论 `seq` 是否为 1 都会创建：

- `transpose_3d(key)`
- `scale(scores)` 副本
- `causal_mask(scores)` 副本
- `softmax(scores)` 副本

这正是 decode fused path 应消除的部分。

---

## 9. `KVCACHE` 的 cache length 应该何时更新

当前全模型结束后才调用：

```cpp
m_kv_cache.set_cache_len(n_past + seq);
```

见 modeling.cpp。

这个策略可以保留，原因是：

- 每层处理同一个 token 块；
- 所有层的 cache 都应具有相同有效长度；
- 在 layer 内部写入时，使用传入的 `n_past` 与局部 `seq`；
- forward 成功完成后，统一将全局有效长度更新到 `n_past + seq`。

推荐约束：

```cpp
const size_t total = n_past + seq;
if (total > m_kv_cache.max_tokens()) {
    throw std::out_of_range("forward: context length exceeded");
}
```

这应在 `forward()` 开始处执行。当前 position embedding 会检查最大长度，见 ops.cpp，但 KV cache 也应该独立检查自己的容量，避免今后模型位置编码实现变化后产生越界写。

---

## 10. 存取成本对比

### 当前实现：每次 decode

假设 `seq=1`：

```text
写 K/V:                 2 × C
重排完整 K cache:       T × C
重排完整 V cache:       T × C
转置完整 K cache:       T × C
```

仅 layout copy 总量约为：

$$
3TC+2C
$$

且会随着上下文 $T$ 线性增长。

### head-major cache：每次 decode

```text
写新 K/V: 2 × C
读取 K/V: attention 本身必须做的读取
```

额外 layout copy 是：

$$
2C
$$

与上下文长度 $T$ 无关。

注意 attention 自身仍需读取 $K/V$：

$$
O(H \times T \times D)=O(TC)
$$

这是 attention 的必要计算；优化的目标是去掉额外的历史 K/V 重排和临时 Tensor 流量，而不是错误地宣称 attention 可以不读历史 cache。

---

## 推荐的改造顺序

1. 新增 `LayerKVCache`，预分配 `[H,Tmax,D]`。
2. 将 `GPT2` 中的 cache 初始化改为传递 layer、head、max token、head dim。
3. 新增 `append_kv()`：把本轮 `[S,C]` 的 K/V 一次性写入 `[H,Tmax,D]`。
4. 先实现 `attention_decode()`，只在 `seq == 1` 使用。
5. decode 路径删除完整 cache 的 `split_head()`、`transpose_3d()`、`causal_mask()` 与 `merge_head()`。
6. 最后再改造 prefill 路径，避免让一次大改同时影响首 token 和后续 token 的数值验证。

这样风险最低：旧 prefill 路径继续可用，而 decode 路径会先获得最明显的 OpenVINO 风格收益。