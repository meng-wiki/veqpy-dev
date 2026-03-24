# Repository Overview

- `veqpy` 是 Python 3.12+ 的磁流体平衡仓库.
- 当前只支持 `numpy` 和 `numba` 两种后端, 不再维护 native C++ 构建链路.
- 当前核心层次是:
  - `veqpy/engine/`: backend-facing array kernels 和导出面
  - `veqpy/model/`: `Grid`, `Profile`, `Geometry`, `Equilibrium`
  - `veqpy/operator/`: `OperatorCase`, packed `layout/codec`, `Operator`
  - `veqpy/solver/`: `Solver`, `SolverConfig`, `SolverRecord`, `SolverResult`
  - `tests/`: `demo.py`, `benchmark.py`, `benchmark/<startup>-<backend>/`

# Current Facts

- `Operator` 是完整的 packed `x -> residual` runtime owner.
- Stage A/B/C/D 当前仍由 [`veqpy/operator/operator.py`](../veqpy/operator/operator.py) 组织.
- packed `layout/codec` 归 `veqpy/operator/`.
- `Solver` 是 solve facade, 不持有 backend 选择逻辑.
- `Solver.solve(...)` 只执行求解并返回 packed `x`; `SolverResult` / `Equilibrium` / coeff 重建由 `solver.result`, `solver.build_equilibrium()`, `solver.build_coeffs()` 提供.
- `Solver` 还提供 `build_equilibrium_history()` 和 `build_coeffs_history()` 作为 history 重建入口.
- `OperatorCase` 当前是可变 runtime case; `SolverRecord` 会复制 case snapshot.
- `Equilibrium` 是单网格 materialized diagnostic snapshot, 不是 solver-side parametric state.
- 当前运行时浮点基线固定为 `np.float64`.
- 当前 profile 权威顺序固定为:
  - `psin`, `F`, `h`, `v`, `k`, `c0`, `c1`, `s1`, `s2`

# Current Hot-Path ABI

- packed state 与 packed residual 的唯一 layout 语言都是 `coeff_index` / `coeff_indices`.
- `Stage-A` 直接从 packed `x` 通过 `coeff_indices` 读取 profile 系数.
- `Stage-D` 直接通过 `coeff_indices` 把 residual block 写回 packed residual.
- runtime 路径里已经不再使用 `coeff_matrix`.
- 当前保留:
  - `Stage-A` 的 per-profile Python loop
  - `Stage-D` 的 residual block registry
- 当前不维护:
  - bulk Stage-A runner
  - engine-level bulk residual runner

engine 边界当前优先使用 packed field bundles:

- `Grid.T_fields`
- `Profile.u_fields`, `Profile.rp_fields`, `Profile.env_fields`
- `Geometry.tb_fields`, `Geometry.R_fields`, `Geometry.Z_fields`, `Geometry.J_fields`, `Geometry.g_fields`
- `Operator.root_fields`, `Operator.residual_fields`

语义化 property 仍然保留, 例如:

- `grid.T`
- `profile.u`
- `geometry.R`

但热 operator 路径应优先直接使用 `*_fields[...]`.

# Optional Semantics

- `None` 在 public/model 层仍然是合法语义, 用于表达真实的可选输入或 inactive topology.
- 例如 `coeffs_by_name[name] is None` 仍表示该 profile 在 packed layout 中不激活.
- 但 hot Numba kernels 当前优先保持单态 ABI.
- 因此 packed profile 路径使用空 `coeff_indices` 数组表达 offset-only, 而不是把 `None` 直接送进 kernel.
- `Ip` / `beta` 是当前唯一明确保留的例外语义点:
  - facade 语义上它们可以被理解为 optional constraints;
  - 但 hot source kernels 仍使用当前 scalar ABI, 因为直接把 `None` 送进 Numba 会引入可测的性能回退.

# Backend Surface

- backend control surface 只有 [`veqpy/engine/__init__.py`](../veqpy/engine/__init__.py).
- 当前真实后端文件是:
  - `numpy_profile.py`
  - `numpy_geometry.py`
  - `numpy_source.py`
  - `numpy_residual.py`
  - `numba_profile.py`
  - `numba_geometry.py`
  - `numba_source.py`
  - `numba_residual.py`
- `VEQPY_BACKEND` 可接受值只有:
  - `numpy`
  - `numba`
- 环境变量未设置时, 默认后端是 `numba`.

# Developer Workflows

- 安装:
  - `py -m pip install -e .`
  - `py -m pip install -e .[dev]`
- 语法快速检查:
  - `py -m compileall veqpy tests`
- 运行最小示例并生成 demo 产物:
  - `py tests/demo.py`
- 运行多模式 benchmark:
  - `py tests/benchmark.py`

# Solver Surface

- `SolverConfig.method` 当前支持:
  - root 路径: `hybr`, `krylov`, `root-lm`, `broyden1`, `broyden2`
  - least-squares 路径: `trf`, `dogbox`, `lm`
- `Solver.solve(...)` 对 root 方法使用 `scipy.optimize.root(...)`.
- `Solver.solve(...)` 对 `trf` / `dogbox` / `lm` 使用 `scipy.optimize.least_squares(...)`.
- 当主方法失败时, 当前 fallback 顺序是:
  - `least_squares/lm`
  - `least_squares/trf`
- `SolverConfig` 当前主要包含三类字段:
  - 求解方法和收敛阈值: `method`, `rtol`, `atol`
  - SciPy 限制字段: `root_maxiter`, `root_maxfev`
  - solve 行为开关: `enable_warmstart`, `enable_homotopy`, `enable_verbose`, `enable_history`
- homotopy 相关策略字段还包括:
  - `homotopy_truncation_tol`
  - `homotopy_truncation_patience`
- `Solver.solve(...)` 支持对上述字段做单次覆盖, 但不接收 `case`.
- 长期替换物理 case 用 `replace_case(...)`.
- 长期替换默认求解配置用 `replace_config(...)`.

# Suggested Checks

- 只改 `README.md` 或 `docs/`:
  - 不强制跑数值脚本.
  - 至少核对路径, 命令, 产物目录仍真实存在.
- 改任意 `veqpy/*.py` 或 `tests/*.py`:
  - 建议先跑 `py -m compileall veqpy tests`
- 改 `veqpy/engine/`, `veqpy/model/`, `veqpy/operator/`, `veqpy/solver/`, 包级 `__init__.py`, packed `layout/codec`, `Operator`, `OperatorCase`, `Solver`, `Equilibrium`:
  - 建议跑 `py -m compileall veqpy tests`
  - 建议跑 `py tests/demo.py`
- 改 source route, residual assembly, solver fallback, homotopy stage policy, benchmark 口径:
  - 额外建议跑 `py tests/benchmark.py`

# High-Risk Areas

- `veqpy/engine/__init__.py`
  - 错误导出会直接破坏 backend surface.
- `veqpy/operator/operator.py`
  - packed runtime ownership, Stage A/B/C/D, residual assembly 当前都在这里收束.
- `veqpy/operator/layout.py`
  - packed index 变化会影响 `x0`, residual, `replace_case(...)`, benchmark 产物和文档.
- `veqpy/operator/codec.py`
  - packed state 和 packed residual 的边界转码都依赖这里.
- `veqpy/solver/solver.py`
  - root / least-squares 路径, fallback, homotopy stage policy 都在这里.
- `veqpy/model/equilibrium.py`
  - snapshot semantics, resample semantics, plotting/comparison 在这里定义.
- `tests/demo.py`
  - 当前最直接的端到端示例入口.
- `tests/benchmark.py`
  - 当前多模式 benchmark 和 physics delta 观察口径入口.

# Quick File Map

- `README.md`
- `TODO.md`
- `pyproject.toml`
- `veqpy/engine/__init__.py`
- `veqpy/operator/operator.py`
- `veqpy/operator/layout.py`
- `veqpy/operator/codec.py`
- `veqpy/solver/solver.py`
- `veqpy/solver/solver_config.py`
- `veqpy/model/equilibrium.py`
- `tests/demo.py`
- `tests/benchmark.py`
- `docs/conventions.md`
- `docs/guardrails.md`
- `docs/veqpy_operators.md`
- `docs/veqpy_equilibrium.md`
