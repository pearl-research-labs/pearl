# Research Iteration Process

This document defines how research findings are captured, iterated upon, and
persisted throughout the Vulkan/OpenCL porting effort.

## Rule 0: RESEARCH-*.md is source of truth

All research findings live in `miner/RESEARCH-*.md` files. These are the
canonical records. Do not let ephemeral chat context override what's documented.

## Iteration Cadence

Each work session must:
1. Read existing RESEARCH-*.md files to re-establish context
2. Update any findings that changed during the session
3. Append an entry to the **Iteration Log** at the bottom of the relevant file
4. Update **Next Actions** checklists (mark done, add new items)

## File Naming Convention

```
RESEARCH-<TOPIC>.md
```

Examples:
- `RESEARCH-VULKAN.md` - Vulkan compute porting research
- `RESEARCH-OPENCL.md` - OpenCL porting research
- `RESEARCH-GEMM.md` - GEMM kernel porting deep-dive (future)
- `RESEARCH-BLAKE3.md` - BLAKE3 on non-CUDA targets (future)

## What to Document

For each topic, capture:
1. **Date-stamped entries** in the Iteration Log
2. **Decisions and rationale** (why one approach over another)
3. **Dead ends** (what didn't work and why)
4. **Performance observations** (comparing CUDA vs target)
5. **Code pointers** to relevant files/line numbers
6. **Open questions** that need further investigation

## Persistence Rules

1. **Write before switching context** - If you discover something important,
   write it to the RESEARCH file before starting the next task.
2. **Don't lose ephemeral findings** - If you learn something in conversation
   that isn't documented elsewhere, add it.
3. **Reference benchmark numbers** - When comparing performance, include the
   exact command, device name, and driver version used.
4. **Link to code** - Reference specific files and line numbers so findings
   can be verified independently.

## Todo Integration

The todo list should reflect research state:
- `research/<component>` items should be `in_progress` while investigating
- Mark `completed` when findings are written to RESEARCH-*.md
- Add new todo items as open questions are identified

## Example Iteration Log Entry

```markdown
## 2026-05-22: Discovered XYZ limitation

Found that Vulkan `VK_KHR_cooperative_matrix` is not supported on NVIDIA
GPUs older than Turing (RTX 20 series). This means we cannot use it as a
WGMMA replacement for the GEMM kernel.

**Decision:** Implement manual tiled matmul in GLSL.

**Files consulted:**
- `miner/pearl-gemm/csrc/gemm/pearl_gemm_kernel.h:42-85` - tile sizing
- `miner/pearl-gemm/csrc/gemm/collective_mainloop.hpp:120-200` - mainloop pattern

**Next Actions:**
- [x] Test cooperative matrix on available HW
- [ ] Prototype manual matmul in GLSL
```
