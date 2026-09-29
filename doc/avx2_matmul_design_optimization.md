# AVX2 FP32 MatMul 设计与优化文档

## 1. 文档目标

本文描述一个面向 AVX2/FMA 平台的单精度矩阵乘实现：

```text
A[1000, 800] @ B[800, 1240] -> C[1000, 1240]
```

默认条件：

- 数据类型：`float32`
- 矩阵布局：row-major
- 运算语义：`C = A * B`
- SIMD 指令集：AVX2
- 融合乘加：FMA3
- 第一阶段实现单线程版本，之后再扩展多线程

设计重点包括：

1. 使用 cache blocking 提高 L1/L2/L3 数据复用率。
2. 使用 register blocking 将 C 的小块长期保存在 YMM 寄存器中。
3. 对 B 进行 packing，保证 micro-kernel 连续读取。
4. 使用 `6 x 16` AVX2 micro-kernel 作为主计算路径。
5. 使用专用 kernel 覆盖当前矩阵的 M/N 边缘。
6. 正确处理 K blocking 产生的部分和累加。
7. 保留任意尺寸矩阵的通用 fallback。

---

## 2. 问题定义

矩阵维度为：

```text
M = 1000
K = 800
N = 1240
```

计算公式：

```text
C[i, j] = sum(A[i, k] * B[k, j]), k = 0 ... K-1
```

若矩阵连续存储，则 leading dimension 为：

```text
lda = K = 800
ldb = N = 1240
ldc = N = 1240
```

内存索引为：

```cpp
A[i * lda + k]
B[k * ldb + j]
C[i * ldc + j]
```

总浮点运算量约为：

```text
2 * M * N * K
= 2 * 1000 * 1240 * 800
= 1,984,000,000 FLOPs
```

即约 `1.984 GFLOP`。其中一次乘法和一次加法分别按一次浮点运算计数。

---

## 3. 总体设计

实现采用两层 blocking：

```text
Cache blocking:
    MC x KC x NC

Register blocking:
    MR x NR
```

初始推荐参数：

```cpp
constexpr int MC = 120;
constexpr int KC = 256;
constexpr int NC = 256;
constexpr int MR = 6;
constexpr int NR = 16;
```

整体执行结构：

```text
for jc = 0 .. N step NC
    for pc = 0 .. K step KC
        pack B[pc:pc+kc, jc:jc+nc]

        for ic = 0 .. M step MC
            compute A[ic:ic+mc, pc:pc+kc]
                  x packed_B
                  into C[ic:ic+mc, jc:jc+nc]
```

其中：

```text
mc = min(MC, M - ic)
nc = min(NC, N - jc)
kc = min(KC, K - pc)
```

顶层循环顺序选择 `jc -> pc -> ic`，使一个 packed B panel 能够被多个 M block 复用。

---

## 4. 分块结构

### 4.1 Cache blocking

推荐初始值：

```text
MC = 120
KC = 256
NC = 256
```

这些参数应被视为起点，而不是所有 AVX2 CPU 的最终最优值。

对应数据大小：

```text
A block: MC * KC * 4
         120 * 256 * 4
         = 122,880 bytes, about 120 KiB

B block: KC * NC * 4
         256 * 256 * 4
         = 262,144 bytes, 256 KiB

C block: MC * NC * 4
         120 * 256 * 4
         = 122,880 bytes, about 120 KiB
```

实际运行时并不要求整个 A、B、C block 同时驻留在同一级 cache。micro-panel 和循环顺序负责建立更细粒度的数据复用。

### 4.2 Register blocking

AVX2 的一个 256-bit YMM 寄存器能够存放：

```text
256 / 32 = 8 个 float
```

因此 16 列需要两个 YMM：

```text
B[k, j:j+7]   -> ymm_b0
B[k, j+8:j+15] -> ymm_b1
```

主 kernel 选择：

```text
MR x NR = 6 x 16
```

C accumulator 需要：

```text
6 rows * 2 vectors per row = 12 YMM registers
```

其余寄存器用于：

- 两个 B 向量
- 一个 A broadcast 向量
- 地址和临时值

该设计接近 AVX2 的寄存器容量上限，因此必须检查生成汇编，避免 accumulator 被 spill 到栈内存。

---

## 5. 当前尺寸的边缘分析

### 5.1 M 方向

```text
1000 = 166 * 6 + 4
```

需要：

```text
6 x 16 主 kernel
4 x 16 M-edge kernel
```

从 cache block 角度：

```text
1000 = 8 * 120 + 40
40 = 6 * 6 + 4
```

前 8 个 `MC=120` block 均可完全分成 `6` 行块，最后一个 `mc=40` block 包含 6 个 `6-row` kernel 和 1 个 `4-row` kernel。

### 5.2 N 方向

```text
1240 = 77 * 16 + 8
```

需要：

```text
6 x 8 N-edge kernel
4 x 8 corner kernel
```

从 cache block 角度：

```text
1240 = 4 * 256 + 216
216 = 13 * 16 + 8
```

最后一个 `NC` block 包含 13 个完整 16-column panel 和一个 8-column panel。

### 5.3 K 方向

```text
800 = 3 * 256 + 32
```

K 尾部不需要独立 SIMD kernel。micro-kernel 接收运行时 `kc`，最后一次直接执行：

```text
kc = 32
```

C 的最终结果由四个 K block 的部分和组成：

```text
C = A[:,   0:256] * B[  0:256, :]
  + A[:, 256:512] * B[256:512, :]
  + A[:, 512:768] * B[512:768, :]
  + A[:, 768:800] * B[768:800, :]
```

因此每个 K block 的 micro-kernel 必须执行 `C += partial_sum`，不能覆盖前一个 K block 的结果。

---

## 6. B Packing 设计

### 6.1 Packing 目的

虽然 row-major B 在 N 方向连续，但 cache blocking 后仍建议将 B 重排成 `kc x 16` micro-panel：

```text
panel 0:
    B[k0, j0:j0+15]
    B[k1, j0:j0+15]
    ...
    B[kc-1, j0:j0+15]

panel 1:
    B[k0, j0+16:j0+31]
    ...
```

这样 micro-kernel 对每个 k 都能连续加载两个 YMM，并且 panel 的地址计算规则固定。

### 6.2 Packed B 布局

```text
packed_B[panel][k][16]
```

线性地址为：

```cpp
packed_B[panel * kc * NR + k * NR + j]
```

最后不足 16 列时用零填充。例如最终 8 列：

```text
[valid x 8][zero x 8]
```

不过最终 8 列 kernel 只加载前 8 个元素，因此不会读取第二个向量。

### 6.3 Packing 伪代码

```cpp
for each 16-column panel:
    nr = min(16, remaining columns)

    for k in [0, kc):
        copy nr valid values
        fill [nr, 16) with zero
```

### 6.4 对齐要求

建议使用 64-byte 对齐的 packed buffer：

```cpp
float* packed_B = static_cast<float*>(
    _mm_malloc(buffer_size, 64));
```

每个 packed row 的长度为：

```text
16 * sizeof(float) = 64 bytes
```

因此其两个 8-float 子向量都满足 32-byte 对齐，可以使用：

```cpp
_mm256_load_ps(bp)
_mm256_load_ps(bp + 8)
```

C 行首及任意列地址未必 32-byte 对齐，写回 C 时使用：

```cpp
_mm256_loadu_ps(...)
_mm256_storeu_ps(...)
```

---

## 7. Micro-kernel 设计

### 7.1 Kernel 组合

针对固定尺寸准备以下四种 AVX2 kernel：

```text
6 x 16: 主体
4 x 16: M 尾部
6 x 8 : N 尾部
4 x 8 : 右下角
```

可以使用模板统一生成：

```cpp
micro_kernel<6, 2>  // 6 x 16
micro_kernel<4, 2>  // 4 x 16
micro_kernel<6, 1>  // 6 x 8
micro_kernel<4, 1>  // 4 x 8
```

第二个模板参数表示每行包含多少个 YMM 向量。

### 7.2 核心计算过程

对每个 k：

```cpp
b0 = load(B[k, 0:7]);
b1 = load(B[k, 8:15]);
```

对 A 的每个 register-block 行：

```cpp
a = broadcast(A[i, k]);
acc[i][0] = fmadd(a, b0, acc[i][0]);
acc[i][1] = fmadd(a, b1, acc[i][1]);
```

`broadcast` 将一个标量复制到 8 个 lane：

```text
A[i,k] -> [a, a, a, a, a, a, a, a]
```

FMA 完成：

```text
acc[lane] = a * b[lane] + acc[lane]
```

一个 B 向量被 6 行 A 重复使用，从而提高 B 的寄存器级复用。

### 7.3 建议的模板 kernel

```cpp
template<int RM, int NV>
static inline void micro_kernel_avx2(
    int kc,
    const float* __restrict A,
    int lda,
    const float* __restrict packed_B,
    float* __restrict C,
    int ldc)
{
    static_assert(RM >= 1 && RM <= 6);
    static_assert(NV == 1 || NV == 2);

    __m256 acc[RM][NV];

    for (int i = 0; i < RM; ++i) {
        for (int v = 0; v < NV; ++v) {
            acc[i][v] = _mm256_setzero_ps();
        }
    }

    for (int k = 0; k < kc; ++k) {
        const float* bp = packed_B +
                          static_cast<std::size_t>(k) * 16;

        const __m256 b0 = _mm256_load_ps(bp);

        __m256 b1;
        if constexpr (NV == 2) {
            b1 = _mm256_load_ps(bp + 8);
        }

        for (int i = 0; i < RM; ++i) {
            const __m256 a = _mm256_broadcast_ss(
                A + static_cast<std::size_t>(i) * lda + k);

            acc[i][0] = _mm256_fmadd_ps(a, b0, acc[i][0]);

            if constexpr (NV == 2) {
                acc[i][1] = _mm256_fmadd_ps(a, b1, acc[i][1]);
            }
        }
    }

    for (int i = 0; i < RM; ++i) {
        float* cp = C + static_cast<std::size_t>(i) * ldc;

        __m256 c0 = _mm256_loadu_ps(cp);
        c0 = _mm256_add_ps(c0, acc[i][0]);
        _mm256_storeu_ps(cp, c0);

        if constexpr (NV == 2) {
            __m256 c1 = _mm256_loadu_ps(cp + 8);
            c1 = _mm256_add_ps(c1, acc[i][1]);
            _mm256_storeu_ps(cp + 8, c1);
        }
    }
}
```

### 7.4 实现注意事项

模板代码便于表达设计，但最终性能需要通过汇编确认：

- 模板循环是否完全展开。
- 12 个 accumulator 是否一直保存在 YMM 中。
- 是否出现栈上的 `vmovups` spill/reload。
- K 循环中是否只包含必要的 load、broadcast 和 FMA。
- 编译器是否产生额外的分支。

若模板版本产生 spill，建议将 `6 x 16` 主 kernel 手工展开，并显式定义 12 个 accumulator。

---

## 8. Edge dispatch 设计

每个 packed B panel 对应一个 `nr`：

```text
nr = min(16, nc - panel_offset)
```

每个 A register block 对应一个 `mr`：

```text
mr = min(6, mc - row_offset)
```

分发逻辑：

```text
mr=6, nr=16 -> kernel<6,2>
mr=4, nr=16 -> kernel<4,2>
mr=6, nr=8  -> kernel<6,1>
mr=4, nr=8  -> kernel<4,1>
其他情况     -> scalar 或 masked fallback
```

对于当前固定 shape，所有 M/N 边缘都能命中 AVX2 kernel，不需要标量路径。

仍建议保留通用 fallback，以支持未来任意尺寸的矩阵：

```cpp
for (int i = 0; i < mr; ++i) {
    for (int j = 0; j < nr; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < kc; ++k) {
            sum += A[i * lda + k] * packed_B[k * 16 + j];
        }
        C[i * ldc + j] += sum;
    }
}
```

如果通用小尾部较常见，可以进一步实现：

- `1/2/3/5 x 16` 行尾 kernel
- `nr < 8` 的 maskload/maskstore kernel
- `8 < nr < 16` 的一个完整 YMM 加一个 masked YMM

---

## 9. 顶层算法伪代码

```cpp
clear C
allocate aligned packed_B

for (int jc = 0; jc < N; jc += NC) {
    int nc = min(NC, N - jc);

    for (int pc = 0; pc < K; pc += KC) {
        int kc = min(KC, K - pc);

        pack_B(
            B + pc * ldb + jc,
            kc,
            nc,
            ldb,
            packed_B);

        for (int ic = 0; ic < M; ic += MC) {
            int mc = min(MC, M - ic);

            compute_block(
                A + ic * lda + pc,
                packed_B,
                C + ic * ldc + jc,
                mc,
                nc,
                kc,
                lda,
                ldc);
        }
    }
}

free packed_B
```

调用参数：

```cpp
matmul_avx2(
    1000,
    1240,
    800,
    A,
    800,
    B,
    1240,
    C,
    1240);
```

---

## 10. C 的初始化与累加语义

若接口语义为：

```text
C = A * B
```

则在进入 K blocking 前将 C 清零一次：

```cpp
for (int i = 0; i < M; ++i) {
    std::memset(C + i * ldc, 0, N * sizeof(float));
}
```

之后所有 micro-kernel 执行：

```text
C += partial result
```

不要在每个 `pc` block 开始时把 C 清零，否则只会保留最后一个 K block 的结果。

如果要支持通用 GEMM：

```text
C = alpha * A * B + beta * C
```

可采用以下策略：

1. 第一个 K block 将 accumulator 与 `beta * C` 合并。
2. 后续 K block执行累加。
3. `alpha` 可以乘到 packed A、packed B 或最终 accumulator 上。
4. 对常见的 `alpha=1`、`beta=0/1` 单独建立快速路径。

---

## 11. 编译与运行要求

推荐编译命令：

```bash
g++ -std=c++17 -O3 -mavx2 -mfma -DNDEBUG matmul.cpp -o matmul
```

仅在 binary 只运行于编译机器或相同微架构时使用：

```bash
g++ -std=c++17 -O3 -march=native -DNDEBUG matmul.cpp -o matmul
```

检查汇编：

```bash
objdump -d -M intel matmul | grep -E "vfmadd|vbroadcast|vmov"
```

核心路径中期望看到：

```text
vbroadcastss
vmovaps 或 vmovups
vfmadd231ps / vfmadd213ps
```

还应检查是否出现大量以 `rsp` 或 `rbp` 为基址的向量读写，它们可能意味着 register spill。

---

## 12. 正确性验证

### 12.1 Reference 实现

建议 reference 使用 double 累积，再转换为 float：

```cpp
for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
        double sum = 0.0;

        for (int k = 0; k < K; ++k) {
            sum += static_cast<double>(A[i * K + k]) *
                   static_cast<double>(B[k * N + j]);
        }

        C_ref[i * N + j] = static_cast<float>(sum);
    }
}
```

### 12.2 误差判断

AVX2 路径使用 FMA，乘法和加法只进行一次舍入；普通 reference 可能有不同舍入顺序，因此不要使用逐元素精确相等。

```cpp
bool nearly_equal(float actual, float expected)
{
    constexpr float atol = 1.0e-3f;
    constexpr float rtol = 2.0e-3f;

    return std::abs(actual - expected) <=
           atol + rtol * std::abs(expected);
}
```

最终阈值需要结合输入分布和 K 大小调整。

### 12.3 必测尺寸

除目标尺寸外，应覆盖以下测试：

```text
1 x 1 x 1
4 x 8 x 32
6 x 16 x 256
7 x 17 x 257
1000 x 800 x 1240
1001 x 801 x 1241
M/N/K 中任一维为 0
```

还应测试：

- 全零输入
- 全一输入
- 正负随机数
- 很小数值
- 包含 NaN/Inf 时是否符合预期语义
- `lda/ldb/ldc` 大于逻辑列数的情况

---

## 13. 性能测量

### 13.1 计时原则

- 先 warm up 多次。
- 正式运行多次，报告中位数或最小值。
- 避免把随机初始化和 reference 计算计入 kernel 时间。
- 明确 packing 时间是否包含在总时间中。
- 使用固定 CPU affinity，降低线程迁移影响。
- 对比单线程 BLAS 时确保线程数一致。

GFLOPS 计算：

```text
GFLOPS = 2 * M * N * K / elapsed_seconds / 1e9
```

当前 shape：

```text
GFLOPS = 1.984 / elapsed_seconds
```

例如执行时间为 20 ms：

```text
GFLOPS = 1.984 / 0.020 = 99.2
```

### 13.2 Linux 性能计数器

可使用：

```bash
perf stat -e \
cycles,instructions,branches,branch-misses,\
cache-references,cache-misses \
./matmul
```

进一步根据 CPU 支持情况观察：

- L1 data cache miss
- L2 request/miss
- LLC load/miss
- stalled cycles
- FP arithmetic retired
- TLB miss

重点判断：

1. IPC 是否合理。
2. FMA 单元是否得到充分利用。
3. packing 后 B 的 cache miss 是否降低。
4. 是否因过大的 block 造成 L1/L2 冲突。
5. 是否因 register spill 导致额外 load/store。

---

## 14. 优化路线

### 14.1 第一阶段：建立可靠 baseline

1. 实现朴素 `i-j-k` reference。
2. 实现 cache-blocked 标量版本。
3. 加入 B packing。
4. 加入 `6 x 16` AVX2 kernel。
5. 加入 `4 x 16`、`6 x 8`、`4 x 8` kernel。
6. 对所有边缘尺寸进行正确性验证。

### 14.2 第二阶段：优化 micro-kernel

#### 手工展开主 kernel

若模板生成代码不理想，显式定义：

```text
c00 c01
c10 c11
c20 c21
c30 c31
c40 c41
c50 c51
```

避免二维数组导致编译器保守处理。

#### K-loop 展开

测试展开 2 次或 4 次：

```text
k
k+1
```

目的：

- 降低循环控制开销
- 增加指令级并行度
- 隐藏 load/broadcast latency

必须检查展开后是否增加寄存器压力和 spill。

#### 软件预取

可试验：

```cpp
_mm_prefetch(reinterpret_cast<const char*>(future_bp), _MM_HINT_T0);
```

预取距离必须实测。由于 packed B 已经是连续流式访问，硬件 prefetcher 可能已经足够，手工 prefetch 不一定有收益。

### 14.3 第三阶段：参数调优

建议搜索：

```text
KC: 128, 192, 256, 320, 384
MC: 72, 96, 120, 144, 180
NC: 128, 256, 384, 512
```

调优时固定其他参数，一次只改变一个维度，并同时检查：

- 总执行时间
- packing 时间
- cache miss
- IPC
- 性能稳定性

### 14.4 第四阶段：Pack A

当前 A 的 K 方向访问连续，因此第一版可以只 pack B。但进一步优化时，可以将 A pack 为：

```text
Apack[micro-panel][k][MR]
```

例如：

```text
Apack[k][0:5]
```

潜在收益：

- 降低跨 `lda` 的地址计算开销
- 改善 TLB 行为
- 更容易预取
- 对非连续或转置 A 更友好
- 形成更标准的 Goto/BLIS 风格层次结构

代价：

- 增加 packing 成本
- 增加临时内存
- 对单次、小矩阵调用可能得不偿失

### 14.5 第五阶段：多线程

较自然的并行维度为 M 或 N cache block。

若每个线程处理不同 `ic` block：

- 线程写入不同 C 行，无写冲突。
- 同一个 packed B panel 可只读共享。
- 需要避免每个 `(jc, pc)` 反复创建线程。

推荐使用持久 parallel region 或线程池，而不是在深层循环中频繁进入和退出 OpenMP parallel region。

多线程设计还需考虑：

- 每线程的 packed A buffer
- packed B 的共享和同步
- NUMA first-touch
- false sharing
- 物理核与 SMT 的区别
- 小 block 下任务粒度过细的问题

---

## 15. 风险与常见错误

### 15.1 每个 K block 覆盖 C

错误：

```cpp
store(C, accumulator);
```

正确：

```cpp
store(C, load(C) + accumulator);
```

或者第一个 K block覆盖、后续 block 累加，但此时需要明确传递 first-block 状态。

### 15.2 对齐假设错误

仅 packed buffer 能保证固定对齐。原始 A、B、C 以及带列偏移的地址不一定对齐。除非已经证明地址对齐，否则使用 unaligned load/store。

### 15.3 N-edge 越界访问 C

最后 8 列不能执行 `6 x 16` C load/store，否则会越过逻辑矩阵边界。应调用 `6 x 8` 或 `4 x 8` kernel。

### 15.4 Packed buffer 尺寸不足

buffer 至少需要：

```text
KC * ceil(NC / NR) * NR * sizeof(float)
```

即使 NC 不是 NR 的整数倍，也要为填零后的完整 panel 分配空间。

### 15.5 模板 kernel 发生 spill

源代码中使用 accumulator 数组并不保证编译器将其全部保存在寄存器中。必须检查汇编和性能计数器。

### 15.6 只测试目标尺寸

目标尺寸刚好只有 4-row 和 8-column 尾部，容易掩盖通用边缘错误。需要额外测试 1、2、3、5 行及非 8 倍数列数。

### 15.7 Benchmark 被自动优化删除

必须使用输出结果，例如计算 checksum，防止编译器认为 C 未被使用而删除整个计算。

---

## 16. 推荐工程结构

```text
matmul/
├── include/
│   └── matmul_avx2.h
├── src/
│   ├── matmul_avx2.cpp
│   ├── pack_b.cpp
│   └── kernels_avx2.cpp
├── tests/
│   ├── test_correctness.cpp
│   └── test_edges.cpp
├── benchmarks/
│   └── benchmark_matmul.cpp
└── README.md
```

职责划分：

```text
matmul_avx2.cpp:
    顶层 cache blocking 和接口语义

pack_b.cpp:
    B packing 和 padding

kernels_avx2.cpp:
    6x16、4x16、6x8、4x8 micro-kernel

test_correctness.cpp:
    与 reference 对比

test_edges.cpp:
    任意 M/N/K 边缘验证

benchmark_matmul.cpp:
    warm-up、计时、GFLOPS 和 checksum
```

---

## 17. 最终推荐方案

针对：

```text
A[1000,800] @ B[800,1240]
```

推荐初始实现：

```text
Cache blocking:
    MC = 120
    KC = 256
    NC = 256

Register blocking:
    主体       6 x 16
    M edge     4 x 16
    N edge     6 x 8
    M/N corner 4 x 8

Packing:
    pack B as kc x 16 micro-panels
    64-byte aligned buffer

K edge:
    runtime kc, 最后一次 kc = 32

Write-back:
    C += partial result
```

当前 shape 的 M/N 尾部均能被专用 AVX2 kernel 完整覆盖：

```text
M = 166 * 6 + 4
N = 77 * 16 + 8
K = 3 * 256 + 32
```

建议首先获得一个正确、无越界、无 register spill 的单线程版本，再依次进行 K-loop 展开、blocking 参数搜索、A packing 和多线程优化。性能优化过程中，以生成汇编和硬件计数器为依据，不仅依赖源代码层面的推断。
