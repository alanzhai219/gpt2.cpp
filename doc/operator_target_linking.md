# 算子链接方式与首 Token 延迟

## 现象

在 GPT-2 推理路径中，即使没有创建或调用 `llm_bricks::Gemv`，仅将
`src/micro/gemv/gemv.cpp` 加入 `llm_bricks` 的源文件列表，也会使首 token
延迟从约 118 ms 上升到约 123 ms。

测试配置：

```bash
export ENABLE_BENCHMARK=1
./build/gpt2 generate "Describe the iphone design" weights 100 0.8 40 42
```

多次测试后，两组中位数仍稳定相差约 5 ms，因此这不是单次 CPU 调度、动态
调频或缓存状态引起的随机误差。

`ENABLE_BENCHMARK=1` 只控制首 token 和后续 token 的 wall-clock 计时；它
不是 LinuxPerf 性能采样。相关计时逻辑位于 `GPT2::generate()`。

## 排除项

### 不是 GEMV 的实际计算开销

当前 GPT-2 路径使用 `ops::gemv()` 计算最终 logits。`llm_bricks::Gemv`
对象的构造和 `infer()` 调用均未启用，因此加入 `gemv.cpp` 不会直接执行
`Gemv::infer()`。

### 不主要是共享库加载时间

动态链接器加载 `.so` 的主要工作发生在进程启动阶段。首 token 计时仅覆盖
`forward(ids, 0)`，此时模型权重和算子对象已经初始化完成。

但是，默认的 PLT 惰性绑定可能在某个共享库符号第一次被调用时发生；该调用可能
位于 `forward(ids, 0)` 中。因此不能仅凭 `.so` 在 `main()` 前已加载就完全排除
动态链接开销。应使用 `LD_BIND_NOW=1` 或链接器的 `-Wl,-z,now` 进行 A/B 验证，
详见后面的工具化验证流程。

## 原因

### 共享库会包含全部算子

当 `llm_bricks` 构建为一个 `SHARED` 库时，所有源文件都会进入同一个
`libllm_bricks.so`。即使调用方没有引用 GEMV，`gemv.cpp` 仍会改变共享库的
整体布局，包括：

- `.text` 中函数的位置和对齐；
- 动态符号表、重定位、PLT/GOT 等元数据；
- 页边界与热点代码的指令缓存、uop-cache 映射；
- 位置无关代码（PIC）和共享库符号可抢占带来的约束。

GPT-2 的首 token 会高频执行 `MatMul`、`LayerNorm`、`Gelu`、`Softmax` 等
算子。虽然这些算子的源代码不变，但未使用算子改变共享库布局后，热点函数
的地址、对齐和缓存映射可能变化，从而造成稳定的延迟差异。

### PLT 惰性绑定会污染首次调用延迟

共享库中的外部函数通常通过 PLT/GOT 调用。默认启用惰性绑定时，动态链接器不会在
程序启动阶段解析全部外部函数；某个函数第一次被调用时才会执行符号查找，并将真实
地址写入 GOT。之后对同一函数的调用会直接使用已经解析的地址。

因此，`MatMul::infer()`、`LayerNorm::infer()` 等算子的第一次调用可能在
`forward(ids, 0)` 内触发绑定工作，并计入 first token latency。这与 `.so` 文件
映射发生在进程启动阶段不同。

可通过 `LD_BIND_NOW=1` 把所有绑定提前到启动阶段进行 A/B 验证：若 first token
延迟下降，说明惰性绑定对冷启动延迟有贡献；若差异保持不变，则主要原因不是惰性
绑定。

### 聚合共享库会增加代码和元数据的缓存压力

一个聚合 `libllm_bricks.so` 会包含当前应用不需要的算子代码，以及与之配套的 ELF
元数据，例如 `.dynsym`、`.dynstr`、hash 表、重定位表、PLT/GOT 和 unwind 表。

额外内容不仅增大库文件，也会改变热点函数的地址、页边界和对齐，并可能增加：

- 指令缓存与 uop-cache 冲突；
- 指令页的 TLB 压力；
- 分支目标缓冲区（BTB）的映射冲突；
- 首次执行热点代码时的代码页 fault 数量。

未使用算子本身不需要实际执行，也可能经由这些效应改变 `MatMul`、`LayerNorm` 等
热点路径的延迟。这是“加入 `gemv.cpp` 后首 token 中位数稳定变慢”的主要候选原因。
静态链接或按算子拆分 target 可以让应用只携带需要的对象文件，从而降低该风险。

## 第一层解决办法：静态库

将 `llm_bricks` 改为 `STATIC` 后，`libllm_bricks.a` 会按目标文件保存各个
算子，例如 `gemv.cpp.o`。

最终链接 `gpt2` 时，链接器只会从静态库中抽取满足未解析符号所需的对象文件。
如果 GPT-2 不引用 `llm_bricks::Gemv`，则 `gemv.cpp.o` 不会进入最终可执行
文件。因此，加入未使用 GEMV 不再影响 `gpt2` 的二进制布局。

静态链接的额外收益是：

- 取消 `libllm_bricks.so` 作为运行时依赖；
- 减少跨共享库符号解析和 PIC 相关约束；
- 使最终可执行文件的热点代码布局更紧凑、可重复。

## 最终解决办法：每个算子一个 CMake target

仅使用一个静态库已经可以在最终链接时剔除未使用对象。继续将每个算子定义为
独立静态 target，则项目可以明确声明自己的依赖，并避免构建不需要的算子。

当前算子 target：

| 算子 | CMake target |
| --- | --- |
| Runtime | `llm_bricks::runtime` |
| Add | `llm_bricks::add` |
| Gelu | `llm_bricks::gelu` |
| LayerNorm | `llm_bricks::layer_norm` |
| MatMul | `llm_bricks::matmul` |
| Rope | `llm_bricks::rope` |
| Scale | `llm_bricks::scale` |
| Softmax | `llm_bricks::softmax` |
| Gemv | `llm_bricks::gemv` |

GPT-2 只链接其当前实际使用的算子：

```cmake
target_link_libraries(gpt2 PRIVATE
    llm_bricks::add
    llm_bricks::gelu
    llm_bricks::layer_norm
    llm_bricks::matmul
    llm_bricks::scale
    llm_bricks::softmax
)
```

因此，`rope` 与 `gemv` 不会成为 GPT-2 的构建依赖，也不会进入最终可执行文件。
其他项目根据自身推理图选择链接所需 target，例如：

```cmake
target_link_libraries(other_project PRIVATE
    llm_bricks::gemv
    llm_bricks::rope
)
```

`llm_bricks::llm_bricks` 保留为兼容用的聚合 interface target，适合希望获得
完整算子集合的 benchmark、测试或通用工具。性能敏感的应用应显式链接实际使用的
算子 target。

## 如何通过工具发现并证明

单个工具不能直接证明“代码布局导致性能变化”。可靠结论来自一系列相互独立的
证据：先证明延迟差异可重复，再证明未使用算子没有被执行，最后通过链接和二进制
分析排除其他候选原因。

### 1. 建立可重复的 A/B 构建

分别保存 `SHARED` 和 `STATIC` 的构建目录或可执行文件。两个版本必须使用相同的：

- 编译器、Release 选项和模型权重；
- prompt、随机种子、`max_tokens`、温度和 `top_k`；
- OpenMP 线程数和 CPU core 绑定；
- CPU governor 与其他系统负载条件。

每个版本应独立启动进程至少 20 次，并记录 `[first token latency]`。报告中位数、
最小值和最大值，而不是使用单次结果。可使用 `taskset` 固定核心，使用
`OMP_NUM_THREADS` 固定线程数；如果环境提供 `hyperfine`，可用它管理多次运行。

同时应验证推理结果相同，例如固定 seed 后比较输出文本，或打开已有 logits dump
并对两个版本的 logits 文件执行 `cmp` 或哈希比较。这样可以排除“性能变化来自计算
结果改变”的可能。

### 2. 证明 `Gemv` 没有进入或执行

对于静态库：

```bash
# 静态库包含 GEMV 对象，说明算子本身可以被其他项目使用。
ar t build/thirdparty/llm-bricks/libllm_bricks.a | grep gemv

# 最终 gpt2 不应含有该未使用算子的定义。
nm -C --defined-only build/gpt2 | grep 'llm_bricks::Gemv::infer'
```

第二条命令没有输出，即证明 `gemv.cpp.o` 没被抽取到最终可执行文件。

对于共享库：

```bash
# 共享库会包含 GEMV，即使 gpt2 从未调用它。
nm -D -C build/thirdparty/llm-bricks/libllm_bricks.so | grep 'llm_bricks::Gemv::infer'
```

还可用 `gdb` 在 `llm_bricks::Gemv::infer()` 设置断点后运行推理；断点未命中说明该
函数未执行。该方法只用于验证调用路径，不应在性能测试时启用。

### 3. 证明 STATIC 与 SHARED 的动态依赖不同

```bash
# STATIC 版本不应显示 libllm_bricks.so。
ldd build/gpt2

# 检查 ELF 的直接动态依赖，结论比 ldd 更适合脚本化。
readelf -d build/gpt2 | grep NEEDED
```

`STATIC` 版本中不应出现 `libllm_bricks.so`；`SHARED` 版本会出现它。这证明两种
构建的运行时装载模型不同，但它本身还不能说明延迟差异的具体来源。

### 4. 排除或确认 PLT 惰性绑定

对同一个 `SHARED` 可执行文件分别在默认环境和以下环境中重复基准：

```bash
LD_BIND_NOW=1 ENABLE_BENCHMARK=1 \
./build/gpt2 generate "Describe the iphone design" weights 100 0.8 40 42
```

`LD_BIND_NOW=1` 使动态链接器在程序启动阶段完成符号绑定。如果共享库版本的首
token 延迟明显下降，说明首次调用中的 PLT 惰性绑定有贡献；如果差异保持不变，
主要原因更可能是热点代码布局、PIC 或缓存映射。

仅作诊断时，可使用：

```bash
LD_DEBUG=statistics ./build/gpt2 ...
LD_DEBUG=bindings ./build/gpt2 ...
```

前者显示动态链接统计，后者显示符号绑定。两者会产生额外输出，不能用于测量性能。

### 5. 直接观察代码布局变化

将 `SHARED` 的“含 GEMV”和“不含 GEMV”版本分别保留后，比较热点符号地址：

```bash
nm -n -C libllm_bricks-with-gemv.so | \
        grep -E 'llm_bricks::(MatMul|LayerNorm|Gelu|Softmax|Scale)::infer'
nm -n -C libllm_bricks-without-gemv.so | \
        grep -E 'llm_bricks::(MatMul|LayerNorm|Gelu|Softmax|Scale)::infer'
```

再配合以下命令查看节大小、节地址和对齐：

```bash
readelf -SW libllm_bricks-with-gemv.so
objdump -drwC libllm_bricks-with-gemv.so
```

如果热点符号地址或 `.text` 节布局因未使用的 `gemv.cpp` 改变，同时 benchmark
中位数同步改变，这就是代码布局效应的直接证据。

可进一步在链接选项中临时加入 `-Wl,-Map,<path>/gpt2.map` 或共享库对应的 map
文件，比较 map 中对象文件和符号的位置。链接 map 对定位“哪个对象改变了布局”最
直接。

### 6. 使用 CPU 性能计数器定位影响类别

在不会输出调试日志的 Release 二进制上，用 `perf stat` 比较两组多次运行的：

```bash
perf stat -r 20 -e task-clock,cycles,instructions,cache-misses,branch-misses,\
context-switches,page-faults -- \
        env ENABLE_BENCHMARK=1 ./build/gpt2 generate \
        "Describe the iphone design" weights 100 0.8 40 42
```

解释方式：

- `cycles` 增加而 `instructions` 接近：更支持 I-cache、分支预测或代码布局问题；
- `instructions` 增加：检查是否走了额外计算或不同代码路径；
- `page-faults` 或 `context-switches` 增加：检查内存映射和调度干扰；
- `LD_BIND_NOW=1` 消除差异：更支持惰性绑定而不是代码布局。

性能计数器用于缩小原因范围，不能单独建立因果关系。应与 A/B 中位数、符号检查和
布局比较共同使用。

### 7. 最强的反事实实验

创建一个不被调用、但大小与 `gemv.cpp` 接近的空算子，或只改变 `gemv.cpp` 在共享
库源文件列表中的顺序。

- 若延迟随着“是否存在无用对象”或“对象位置”改变，而与 GEMV 是否执行无关，则支持
    代码布局假设；
- 若 `LD_BIND_NOW=1` 后差异消失，则优先归因于惰性符号绑定；
- 若改为 `STATIC`、并确认最终可执行文件不含 `Gemv::infer()` 后差异消失，则证明
    未使用对象不再影响该应用的最终二进制。

## 构建

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j10
```

## 建议

1. 性能敏感的可执行程序使用静态算子 target，并显式声明所需算子。
2. 新增算子时，不要自动加入所有应用的链接依赖。
3. 对比性能时固定输入、线程数和 CPU 频率策略，并报告多次运行的中位数。
4. 若需要在多个程序间分发动态库，仍可使用共享库，但应按算子或功能域拆分共享库，避免一个未使用算子改变全部热点路径的布局。
