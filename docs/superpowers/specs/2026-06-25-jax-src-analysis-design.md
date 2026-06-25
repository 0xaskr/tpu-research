# JAX 源码解析 — 设计文档

**日期:** 2026-06-25
**状态:** 进行中
**上下文:** 在 tpu-research 项目中新增 `jax-src/` 目录，以 Jupyter notebooks 形式逐层解析 JAX 源码。

---

## 1. 目标

通过可运行的 Jupyter notebooks，从全局架构到模块细节，系统性地解析 JAX 源码。每个 notebook 自包含：源码路径导航、关键函数与数据结构、调用链追踪、可运行验证。

## 2. 目录结构

```
jax-src/
├── 00-jax-architecture.ipynb      # JAX 包整体架构：模块地图、核心概念、入口函数
├── 01-xla-compile-pipeline.ipynb  # JAX → StableHLO → HLO → 二进制 编译链路
├── 02-core-ir-tracing.ipynb       # Jaxpr IR、core.py、trace 机制、线性化
├── 03-pallas-overview.ipynb       # Pallas TPU kernel 编程模型概览
├── 04-mosaic-memory.ipynb         # Mosaic VMEM/SMEM/CMEM/HBM 内存系统
└── 05-tpu-runtime.ipynb           # libtpu 绑定、设备管理、CustomCall 后端
```

数字前缀固定阅读顺序：整体架构 → 编译链路 → 核心 IR → Pallas → Mosaic → Runtime。

## 3. 设计原则

### 3.1 从整体到细节

- 00 建立全局地图，让后续每个笔记本都有清晰的「定位」
- 每个笔记本开头标注它处于整体架构的哪一层
- 跨模块的调用链在各自 notebook 内展开，不跨文件跳转

### 3.2 每个 notebook 自包含

- **源码路径**：标注涉及的关键文件路径和行号
- **调用链**：从公开 API 追踪到内部实现
- **可运行**：每个 cell 能独立执行，输出不依赖执行顺序之外的隐藏状态
- **关联**：结尾指向相关 notebook

### 3.3 与项目代码的关系

- `jax-src/` 是源码分析笔记，回答「JAX 内部怎么做」
- `src/vmem_probe/` 是实战工具，回答「我们怎么用这些知识」
- 两者互补：notebook 里的原理可以解释 vmem_probe 的行为

## 4. 各 Notebook 范围

### 00 — JAX 整体架构

- JAX 包的物理目录结构（`_src/`、`experimental/`、`interpreters/` 等）
- 5 层架构分层：User API → Tracing/IR → Transformations → Compilation → Runtime
- 核心数据结构速览：`Jaxpr`、`ClosedJaxpr`、`WrappedFun`、`Trace`
- 关键入口函数：`jit`、`grad`、`vmap`、`make_jaxpr`、`pallas_call`
- 平台检测与设备分发

### 01 — XLA 编译链路

- `jax.jit` → `_src/api.py` → `_src/dispatch.py` 调用链
- Jaxpr → StableHLO 转换（`_src/interpreters/mlir.py`）
- StableHLO → HLO → TPU 可执行文件（`_src/pallas/mosaic/lowering.py`）
- 编译缓存机制（`_src/compilation_cache.py`）
- `CompilerParams` 如何影响 lowering（`vmem_limit_bytes` 等）

### 02 — 核心 IR & Tracing

- `jax._src.core` 深度分析：`Jaxpr`、`ClosedJaxpr`、`Var`、`Literal`、`AbstractValue`
- trace 机制：`WrappedFun` → `Trace` → `Tracer`
- Jaxpr 的生成：`partial_eval.trace_to_jaxpr_dynamic`
- Jaxpr 的变换：AD (`ad.py`)、batching (`batching.py`)
- 等式的结构：`JaxprEqn`、`Primitive`、`eval_jaxpr`

### 03 — Pallas 概览

- Pallas 是什么：TPU/GPU kernel 编程模型
- 公开 API vs 内部实现：`experimental/pallas/tpu.py` → `_src/pallas/pallas_call.py`
- `pallas_call` 的完整生命周期：kernel tracing → lowering → compilation → execution
- `BlockSpec`、`GridSpec` 设计
- `CompilerParams` 传递路径：Python → backend config JSON → XLA flag

### 04 — Mosaic 内存系统

- `MemorySpace` 枚举：VMEM、VMEM_SHARED、SMEM、CMEM、HBM、SEMAPHORE
- `TpuInfo` dataclass：`vmem_capacity_bytes`、`num_cores`、`hbm_capacity_bytes`
- `get_tpu_info()` 的实现：device kind → 查表 → 返回硬件参数
- VMEM 双缓冲：`pipeline.py` 中的 `BufferedRef`
- scoped VMEM limit 机制：`--xla_tpu_scoped_vmem_limit_kib` + `vmem_limit_bytes`
- 与 `src/vmem_probe/probe_vmem.py` 的对应关系

### 05 — TPU Runtime

- `jax._src.xla_bridge`：后端发现与选择
- libtpu 动态加载：`lib/` 目录结构
- `tpu_custom_call.py`：`CustomCallBackendConfig` 序列化
- `scoped_memory_configs` 的 JSON 格式
- PJRT / IFRT 接口
- `LIBTPU_INIT_ARGS` 环境变量的作用

## 5. 技术依赖

- **运行时**: Python 3.12+, JAX 0.10.x (with TPU), libtpu, jupyter, ipykernel
- **不需要**: GPU、额外数据集、外部服务
- **执行环境**: TPU v6e lite (1 core)，代码兼容其他 TPU 代际

## 6. 非目标（YAGNI）

- **不翻译** JAX 源码为中文——只是解读和分析
- **不修改** JAX 源码做实验——只读分析
- **不覆盖** JAX 的全部模块——聚焦 TPU 相关链路
- **不做** 性能 benchmark——侧重代码理解，非性能测量
- **不重复** 官方文档内容——补充文档没覆盖的内部实现细节
