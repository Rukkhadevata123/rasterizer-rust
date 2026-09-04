# Modern Graphics API Refactor Plan

## Status

Status: proposed after repository-wide reassessment

Target release: **5.0.0** if the public renames and removals below are implemented. The crate currently exposes `RenderContext`, `Renderer`, `RenderQueue`, `RenderCommand`, `RenderState`, and the module tree publicly, so applying this plan without compatibility aliases is a breaking change.

This document replaces the completed pre-4.0 roadmap. It describes an optional architectural modernization of the existing software rasterizer. The goal is to expose a small, accurate, explicit rendering API while retaining the safe synchronous CPU backend and its current performance properties.

No phase is approved merely because it appears here. Each phase has an exit criterion and a decision gate. Optional GPU-like concepts are added only when the CPU implementation has corresponding behavior.

## Repository Coordination and Benchmark Portability

`Plan.md` is tracked as the cross-machine source of truth while this refactor is active. Update it in the same small slice that changes the implementation, including the commit ID, validation performed, and any unresolved decision. After the refactor is complete, archive the plan outside the repository and remove it in an explicit cleanup commit rather than silently ignoring it again.

Correctness tests, deterministic contracts, and required output behavior remain absolute gates. Performance measurements are environment-relative evidence: CPU model, power policy, thermal state, operating system, compiler/toolchain, background load, and worker topology can all change the result. Benchmark CSV files and numbers already present in the repository document the environment in which they were captured; they must not be treated as a universal baseline on another machine, or influence acceptance merely because a new machine is faster or slower in absolute terms.

On every development machine, collect a fresh baseline from the immediately preceding implementation using the same machine, toolchain, release settings, workload, worker counts, and source/output path as the candidate. Prefer adjacent old/new runs and repeat or reverse their order when short or multi-worker scenarios are noisy. Use repository benchmark history to check schema, workloads, hashes, and prior reasoning—not as the performance denominator for a different environment. Record machine metadata and state explicitly when reporting a gate result.

## Reassessment Summary

The original direction was sound, but the implementation order and several proposed types needed correction after comparison with the current code.

The main conclusions are:

1. **Split execution from render-target ownership before recording commands.** `Renderer` currently owns a `Rasterizer`, one `FrameBuffer`, background caching, and a reusable shadow-depth snapshot. A command buffer cannot cleanly target external attachments while this ownership remains bundled.
2. **Do not start with one heterogeneous frame-wide `CommandBuffer<'a>`.** Shadow and main work use different concrete shader and varying types. The renderer relies on static generic dispatch. A borrowed `Vec<EncodedRenderPass<'a>>` cannot hold these pass types without an enum or type erasure, and recording both passes also conflicts with borrowing the shadow target first mutably and then immutably.
3. **Use one typed render pass per command buffer in the first implementation.** Submit the shadow buffer synchronously, release its mutable borrow, then create a read-only shadow view and record/submit the main buffer. This preserves static dispatch and accurately models the existing dependency.
4. **Do not model interleaved framebuffer storage as two independent mutable attachment references.** `FrameBuffer` stores interleaved `Sample { color, depth }` values. The initial pass descriptor must hold one exclusive target borrow plus optional color/depth operations. Separate `ColorTargetView<'a>` and `DepthTargetView<'a>` values would imply aliasing that the storage cannot safely provide.
5. **Separate shader algorithms and draw data before designing bind groups.** `PbrShader` currently combines shader code, frame data, object transforms, shadow bindings, and fallback material state. Recording `set_pipeline` and `set_bind_group` calls before fixing that boundary would only encode the current coupling in a new API.
6. **Treat mirrored winding deliberately.** Scene objects cache the required `FrontFace` from their transform determinant, and recording selects that primitive-state variant per draw. Pipeline caching may later reuse the two variants; mutable per-object data must not be placed in an allegedly immutable pipeline.
7. **Preserve the meaning of existing benchmark columns.** Current pass “preparation” includes queue/shader construction and other pass setup, not just triangle preparation. New recording and submission timings require a versioned CSV schema or additional columns rather than silently changing old columns.
8. **Keep hot reload as a first-class acceptance path.** Main target, supersampling target, shadow target, model resources, mip policy, and window dimensions currently have different reload behavior. New device/resource ownership must not turn every configuration change into a full rebuild.
9. **Freeze the 5.0 public boundary before building typed commands.** The current crate exposes custom `Shader` implementations and most rendering modules. Whether commands are extensible library API or only an internal façade for PBR/shadow changes the command-buffer, pipeline, binding, and visibility design.

## Current Architecture and Constraints

| Area | Current implementation | Constraint on the redesign |
|---|---|---|
| Scene | `RenderContext` owns camera, lights, objects, and selected shadow light | This is scene data, not a GPU context |
| Orchestration | `render_shadow_pass` and `render_main_pass` build shader vectors and draw queues, then execute immediately | Recording must separate construction from execution without duplicating these paths |
| Submission | `RenderQueue` is an ordered draw list; `RenderCommand` contains `shader_index`, geometry, material, state, and sort depth | Reserve “queue” for command-buffer submission |
| Shader dispatch | `Shader<C>` is generic; prepared triangles retain `&S` and copyable fragment context | Static dispatch is a performance and design invariant unless measurement justifies changing it |
| Shader ownership | One `PbrShader` is configured per object; a special identity shader is used for pre-transformed transparent vertices | Pipeline, frame, object, and material responsibilities must be split before bind-group syntax is meaningful |
| Raster state | `RenderState` contains culling, mirrored winding, depth test/compare/write, opaque/alpha blending, and wireframe | Defaults and all state combinations must remain behaviorally identical |
| Target storage | `FrameBuffer` owns interleaved linear RGB and depth samples, including supersamples | API-level attachments need not imply physically separate arrays |
| Parallelism | Triangle preparation is parallel where profitable; rasterization writes exclusive 8-row bands | No locks, atomics, `UnsafeCell`, manual `Sync`, or ordering loss may be introduced |
| Transparency | Transparent triangles are globally sorted by view-space depth with insertion IDs as deterministic ties | Every band must visit transparent primitives in exactly that order |
| Shadow resource | Shadow depth is copied into a reusable `Arc<Vec<f32>>` snapshot before the main pass | A direct read-only view is optional and must be benchmarked against the contiguous snapshot |
| Background | Solid, gradient, and image generation are folded into `clear_with_options` | Constant attachment clearing and generated background content are different operations |
| Post-process | SSAA resolve, exposure, optional ACES, linear-to-sRGB conversion, and `u32` packing happen together | This is a CPU post-process/resolve pass, not a compute shader |
| Application paths | GUI, CLI, and benchmark repeat the same shadow/main/post-process sequence | A shared frame executor is more valuable than an early generic `Surface` trait |
| Public API | Most modules and rendering types are public in `lib.rs` | Renames/removals require a major release or an explicitly experimental namespace |

## Goals

- Expose accurate `RenderDevice`, `CommandEncoder`, `RenderPassEncoder`, `CommandBuffer`, and `GraphicsQueue::submit` concepts.
- Keep submission synchronous and make that fact visible in documentation and return types.
- Separate render execution, targets, reusable resources, pipeline state, and draw bindings.
- Preserve deterministic output, transparent order, safe band ownership, and release performance.
- Make shadow, main, background, resolve/tonemap, and presentation boundaries visible in code and profiling.
- Use Rust types and borrowing to validate resource access where practical; use runtime validation only for state that cannot be expressed cheaply in types.
- End each migration slice with one production execution path, not parallel legacy and modern renderers.

## Non-Goals

- A hardware GPU backend as part of this refactor.
- Vulkan/WebGPU source compatibility.
- Asynchronous submission or execution.
- Fake fences, semaphores, barriers, command pools, image layouts, memory heaps, present modes, or swapchains.
- Runtime shader compilation while shaders remain Rust code.
- General-purpose heterogeneous descriptor arrays before typed bindings prove insufficient.
- A frame graph for the current fixed pass sequence unless later features create meaningful branching or transient-resource reuse.
- Changing rasterization algorithms, shading equations, color management, or asset support while the API boundary is being migrated.

## Design Principles

- Preserve output correctness before pursuing API resemblance.
- Keep `core::rasterizer` unaware of scene models, materials, application configuration, and presentation.
- Keep shader execution statically dispatched in the required phases.
- Prefer a narrow API that represents real behavior over a broad API filled with no-op GPU concepts.
- Keep `FrameBuffer`'s interleaved `Sample` layout until a separate measured storage experiment justifies changing it.
- Preserve one total primitive order per submitted draw phase; band binning may partition pixels, never reorder primitives inside a band.
- Distinguish immutable pipeline configuration from frame-, object-, material-, and draw-frequency data.
- Avoid raw integer resource IDs. If persistent handles become necessary, use typed generation-checked handles.
- Make invalid command state return structured errors. Do not use `Drop` to panic or silently report validation failures.
- Make large module moves only after behavior is stable; do not mix mechanical relocation with semantic rewrites.
- Establish a same-machine baseline immediately before each performance-sensitive subphase.

## Target Architecture

```text
Application / FrameRenderer
  |- RenderScene
  |    |- Camera
  |    |- Lights
  |    `- SceneObjects / Models / Materials
  |
  |- RenderDevice
  |    `- creates descriptors, pipelines, encoders, and optional resources
  |
  |- GraphicsQueue
  |    `- synchronous submit of typed CommandBuffer values
  |         `- SoftwareRasterBackend
  |              |- Rasterizer
  |              `- reusable preparation and band-bin scratch
  |
  |- FrameTargets
  |    |- ShadowTarget
  |    |- MainHdrTarget
  |    `- PresentBuffer
  |
  |- FrameResources
  |    |- background image cache
  |    `- optional reusable shadow snapshot
  |
  `- Output destination
       |- minifb window
       |- PNG writer
       `- benchmark hash
```

Initial frame execution remains explicitly sequential:

```text
record shadow pass
-> submit synchronously
-> obtain ShadowDepthView or snapshot
-> record main pass (clear/background/opaque/masked/transparent)
-> submit synchronously
-> resolve + tonemap + pack PresentBuffer
-> present, save, or hash
```

This is intentionally more limited than a GPU API. It is also implementable with safe borrowing and concrete shader types.

## Terminology and Module Direction

Recommended names (public visibility is defined separately in Phase 11.0):

| Current name | Current role | Target name |
|---|---|---|
| `RenderContext` | Camera, lights, objects, selected shadow light | `RenderScene` |
| `Renderer` | Backend, target, and caches | Remove after splitting responsibilities |
| `RenderQueue` | Ordered high-level draw work | `RenderPhase` |
| `RenderCommand` | One draw and its bindings/state | `DrawPacket` |
| `RenderState` | Raster/depth/blend state plus winding | Split into `GraphicsPipelineState` and an explicitly documented draw override if needed |
| `DepthCompare` | Depth comparison operation | `CompareFunction` |
| `PreparedTriangle` | Clipped screen-space raster input | `RasterPrimitive` if the rename improves internal readability |
| `ClearOptions` | Clear values plus generated background | Attachment operations plus `BackgroundPass` |
| `ShadowPassOutput` | Shadow view/snapshot and transform metadata | `ShadowMap` |
| removed 4.x post-process helper | Resolve, tonemap, transfer conversion, packing | `execute_resolve_tonemap_pass` |
| `draw_queues_profiled` | Prepare/bin/rasterize draw lists | Backend `execute_phases_profiled` |
| `shared_depth_values` | Copy-on-write depth snapshot | `snapshot_depth_attachment` |

The repository also has both `core::pipeline` and top-level `pipeline`, which becomes confusing once real pipeline objects exist. After the API stabilizes, prefer:

```text
core::shader       # Shader trait, varyings, fragment I/O
core::rasterizer   # clipping, preparation, band rasterization
render::state      # pipeline and attachment state
render::command    # encoders and command buffers
render::software   # CPU backend and queue execution
render::target     # render targets and read-only views
passes             # shadow, background, main, tonemap builders
scene              # scene/assets, unchanged in responsibility
```

Do not perform this module move in the same commit as command execution changes.

## Phase 11.0: Freeze Contracts and Baselines

Priority: required

Difficulty: low

Goal: make correctness, API compatibility, profiling semantics, and performance budgets explicit before structural changes.

Completed slices:

- [x] `03b4edc` (`characterize render state behavior`) records the complete `RenderState::default()` contract and the truth table for all eight `DepthCompare` variants. Existing integration tests cover depth-test disable, depth writes, alpha blend/discard, culling, mirrored winding, and wireframe. This slice raised the full release suite to 131 tests.
- [x] `20d4cc9` (`parse benchmark columns by name`) replaces positional benchmark-column access with a tested header-name parser while preserving the v1 producer and merged CSV order. It rejects missing/duplicate columns, malformed row widths, and unsupported explicit schema versions.
- [x] `5dd57ca` (`record 5.0 render api boundary`) records the accepted public roots, custom-shader policy, internal backend/command types, binary/tooling boundary, `nalgebra` policy, and migration checks in `docs/render-api-5.md`.
- [x] `1861bff` (`characterize transparent ordering`) records the view-space depth/insertion/clip/band ordering contract and proves identical packed color and depth output with one versus four Rayon workers. The full release suite now contains 132 tests.
- [x] `9f42b5e` (`characterize mixed render phases`) adds a whole-main-pass regression covering opaque output, masked discard and depth writes, transparent blending, and transparent no-depth-write behavior in one frame. The full release suite now contains 133 tests.
- [x] `b1a6bfe` (`introduce benchmark schema v2`) replaces the ambiguous preparation timings with explicit pass setup, recording, attachment/background, backend preparation, rasterization, and inclusive submission measurements. The Rust producer, v1/v2 parser, matrix merger/summary, CLI contract, and benchmark documentation now agree; the full release suite contains 134 Rust tests plus 5 Node parser tests.
- [x] `f306e62` (`enforce benchmark regression budget`) adds a schema-v2 comparison command that requires matching environment/workload metadata, scenario sets, output hashes, and at least five samples, then fails above a 5% full-frame mean regression. Tracked exceptions require an affected scenario, a higher explicit threshold, and a rationale. The validation suite contains 134 Rust tests plus 9 Node benchmark tests.
- [x] `81f4a8a` (`record pre-11.1 benchmark baseline`) commits the schema-v2 matrix captured from `f306e62`: 12 scenarios (six workloads at 1 and 12 workers), 10 measured frames after 3 warmups, exact metadata, and stable hashes across frames and worker counts. Raw samples remain directly consumable by the performance gate.

### Recommended 5.0 public boundary

Ratify or replace this recommendation before Phase 11.1:

- [x] Continue supporting user-defined shaders. The programmable shader trait is already public, integration tests implement it externally, and programmability is a stated project feature.
- [x] Expose one stable `render` façade containing shader traits/I/O, vertex-facing types, pipeline and pass descriptors, typed bindings/resources, device/queue/encoder/command-buffer types, and structured errors.
- [x] Make `core` an implementation module (`pub(crate)`) where practical. Re-export only intentionally supported shader/geometry types through `render`; retain `scene` and `io` as the other stable public roots.
- [x] Keep `Rasterizer`, `RasterPrimitive`, `SoftwareRasterBackend`, band bins, and preparation scratch private to the backend.
- [x] Keep `RenderPhase` and `DrawPacket` private command-buffer representation. Public callers record through `RenderPassEncoder` rather than constructing backend work lists directly.
- [x] Keep command buffers and pipelines generic/typed enough for user-defined shader implementations. Built-in PBR and shadow pipelines are consumers of the same API, not privileged command enum variants.
- [x] Keep public fields private unless direct construction is an intentional compatibility commitment; prefer validated descriptors/builders and accessor methods.

If custom shaders are rejected as a 5.0 goal, revise Phases 11.4 and 11.5 before implementation: a closed built-in pipeline enum is then simpler and should not be disguised as a general graphics API.

### Work

- [x] Decide that the breaking path targets 5.0.0. If a 4.x experiment is desired instead, place the new API under an explicitly unstable module and do not remove current public symbols.
- [x] Record the current public rendering surface used by integration tests and examples, then approve the replacement boundary above.
- [x] Document the exact transparent ordering contract: ascending view-space `z` for the current camera convention, then insertion ID, then preservation through clipped primitives and every band bin.
- [x] Add or retain focused tests for default state, every depth compare function, depth-test-disabled behavior, depth-write behavior, alpha blend, discard, culling, mirrored winding, and wireframe.
- [x] Add a pass-level regression test covering opaque + masked + transparent rendering in one frame.
- [x] Keep the existing transparent depth/tie tests and add a one-versus-multiple-worker output comparison covering clipping and band execution.
- [x] Capture a fresh release benchmark matrix and hashes from the commit immediately preceding Phase 11.1.
- [x] Introduce benchmark schema v2 before adding recording/submission fields. Preserve committed v1 results as read-only historical data.
- [x] Give schema v2 an explicit `schema_version` and, for each render pass, report at least:
  - pass setup: deriving cameras, bindings, sort keys, and background resources before encoder work;
  - recording: appending and validating commands, including phase finalization/sorting;
  - attachment/background processing: clear or fused background initialization performed by the backend;
  - backend preparation: vertex-cache work, vertex shading, clipping, primitive creation, and band-bin construction;
  - rasterization: ordered band execution;
  - submission total: inclusive synchronous `submit` duration, not an additional exclusive duration to sum with its nested stages.
- [x] Continue reporting scene loading, post-processing (resolve, tonemap, transfer conversion, and packing), and complete-frame duration.
- [x] Replace fixed benchmark column indexes with a header-name lookup while consuming schema v1, with focused Node regression tests.
- [x] Update the Rust producer, parser's supported columns/version, summary output, and benchmark documentation together when schema v2 lands.
- [x] Adopt a default regression threshold of no more than 5% in full-frame mean for the representative matrix, with repeated same-machine samples. Any accepted exception must be recorded with the affected scenarios and rationale.

### Exit Criteria

- Correctness tests characterize behavior that later type changes must preserve.
- Benchmark results can be compared without reinterpreting old columns.
- The release/versioning strategy and supported 5.0 public surface are explicit.
- Benchmark scripts consume named schema-v2 columns rather than positional indexes.

### Decision Gate

Stop after 11.0 for approval of the public API boundary, custom-shader policy, benchmark schema, and baseline. Do not begin public renames in 11.1 or ownership types in 11.2 while those decisions remain open.

## Phase 11.1: Correct Vocabulary and State Modeling

Priority: required

Difficulty: low to medium

Goal: remove misleading names and define pipeline state that maps to behavior already implemented by the rasterizer.

Completed slices:

- [x] `3af9b01` (`rename render context to scene`) replaces the public `RenderContext` type with `RenderScene` across scene loading, pass orchestration, benchmarks, integration tests, and the 5.0 API record. No compatibility alias or behavior change was introduced; the validation suite remains at 134 Rust tests plus 9 Node benchmark tests.
- [x] `aecba54` (`rename render queue and command types`) replaces the public `RenderQueue` and `RenderCommand` types with the backend-oriented names `RenderPhase` and `DrawPacket` across execution signatures, passes, tests, and the 5.0 API record. Local `*_queue` names and `draw_queue*` methods remain intentionally scoped to the next mechanical slice.
- [x] `1d9b7cd` (`rename render execution to phases`) completes the vocabulary migration by renaming immediate draw methods, parameters, shadow/opaque/masked/transparent locals, capacity counts, and related test names from queue to phase. Submission ordering and rendering behavior are unchanged.
- [x] `3c34809` (`rename core pipeline module to shader`) moves the unchanged shader trait and programmable-stage I/O module from `core::pipeline` to `core::shader`, then updates internal imports, integration tests, and the 5.0 API record. Git records the module file as a 100% rename.
- [x] `e6953bf` (`define graphics pipeline state`) introduces the target `GraphicsPipelineState`, primitive/depth/color sub-state, and narrowly scoped topology, front-face, polygon, compare, and alpha-blend vocabulary. Exact defaults and every legacy depth comparison mapping are tested; a temporary `RenderState -> GraphicsPipelineState` adapter records the behavior-preserving migration without changing rasterizer execution yet.
- [x] `f862795` (`use primitive pipeline state`) replaces the legacy cull, winding-inversion, and wireframe fields with `PrimitiveState` throughout rasterization, passes, app/benchmark setup, and tests. `CullMode` now belongs to pipeline state; scene objects cache an explicit `FrontFace` variant instead of a boolean. No compatibility fields or alternate execution path remain. All output hashes match the pre-11.1 baseline and the 12-scenario same-machine performance gate passes.
- [x] `feb9a1e` (`use depth stencil state`) replaces `DepthCompare` plus the separate depth-test/compare/write fields with `CompareFunction` and `Option<DepthStencilState>` throughout execution and tests. `CompareFunction::Always` preserves independent writes when comparison is disabled, while `None` performs neither comparison nor depth storage. No legacy depth state remains; hashes and the 12-scenario performance gate are unchanged/passing.
- [x] `ab5e397` (`use color target state`) replaces `BlendMode` and the temporary `RenderState` with `ColorTargetState` and `GraphicsPipelineState` throughout recording and raster execution. Opaque uses `blend: None`, alpha uses `BlendState::Alpha`, and shadow uses `color_target: None`. Fragment execution remains independent of color storage, with both named masked-shadow tests plus a direct depth-only regression guard. No legacy state adapter or alternate path remains; hashes and the 12-scenario performance gate pass.

### Scene and submission names

- [x] Rename `RenderContext` to `RenderScene`.
- [x] Rename `RenderQueue` to `RenderPhase` and `RenderCommand` to `DrawPacket`.
- [x] Rename local shadow/opaque/masked/transparent queues to phases.
- [x] Keep `GraphicsQueue` reserved for submission of finished command buffers.
- [x] Rename `core::pipeline` to `core::shader` only as a separate mechanical commit.

### Pipeline state

Target shape:

```rust
pub struct GraphicsPipelineState {
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub color_target: Option<ColorTargetState>,
}

pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub polygon_mode: PolygonMode,
}

pub struct DepthStencilState {
    pub depth_compare: CompareFunction,
    pub depth_write_enabled: bool,
}

pub struct ColorTargetState {
    pub blend: Option<BlendState>,
}
```

- [x] Introduce only `PrimitiveTopology::TriangleList` initially.
- [x] Replace `wireframe: bool` with `PolygonMode::{Fill, Line}`.
- [x] Replace `depth_test: false` with `CompareFunction::Always` while preserving independent depth writes.
- [x] Represent opaque output as `blend: None` and current source-alpha blending as a concrete `BlendState` or narrowly scoped `BlendMode::Alpha` conversion.
- [x] Represent the shadow pipeline as `color_target: None`, so depth-only work skips color writes. `FrameBuffer` stores RGB but not attachment alpha; add a `ColorWriteMask` later only if it describes stored channels and at least one tested path uses a partial mask.
- [x] Keep fragment execution independent from color attachment writes. `color_target: None` suppresses only the final color store; the current shadow shader must still sample masked-material alpha and return `FragmentOutput::Discard` before depth is written.
- [x] Retain `masked_shadow_fragments_respect_material_alpha` and `masked_shadow_fragments_sample_base_color_texture_alpha` as explicit guards against incorrectly optimizing away shadow fragment execution.
- [x] Preserve all current defaults exactly.
- [x] Resolve mirrored objects explicitly:
  - selected: record the corresponding `FrontFace` pipeline-state variant for negative-determinant transforms; cache the two variants when immutable pipeline objects land.
- [x] Do not call per-object mutable state an immutable pipeline.

### Exit Criteria

- Existing raster tests and output hashes are unchanged.
- Shadow rendering does not require a meaningful color attachment write.
- Masked shadow fragments still execute alpha testing before depth writes.
- State passed to `core::rasterizer` is self-contained and contains no scene material access.
- No `RenderQueue` name remains for draw-list storage.

### Decision Gate

Stop here if the desired result is only clearer terminology and state modeling. The remaining phases intentionally change ownership and submission APIs.

## Phase 11.2: Split Backend, Targets, and Reusable Resources

Priority: required before command recording

Difficulty: medium

Goal: remove the ownership knot in `Renderer` while retaining immediate execution temporarily.

Completed slices:

- [x] `be2cab5` (`separate render targets`): introduced separately owned main and shadow `RenderTarget` values, made immediate execution accept an explicit mutable target, and removed framebuffer ownership plus dimension-based construction from `Renderer` without retaining a compatibility path.
- [x] `b4b4d64` (`extract frame resources`): moved background texture caching and reusable shadow-depth snapshot storage into an explicitly owned `FrameResources`, passed it through GUI, CLI, benchmarks, and render passes, and removed the corresponding state and methods from `Renderer`.
- [x] `853a1c2` (`rename renderer to software backend`): replaced the old `Renderer` type with `SoftwareRasterBackend` without an alias, made its `Rasterizer` private, updated production and test terminology, and removed the last direct test mutation of rasterizer internals.
- [x] `7c69171` (`name backend execution explicitly`): replaced `draw_phase(s)` and their profiled variants with explicit `execute_phase(s)` entry points, renamed the returned timings to `BackendExecutionTimings`, and removed the old methods without forwarding wrappers.
- [x] `8402902` (`share one software backend`): made GUI, CLI, and benchmarks execute shadow then main through one backend, added a different-sized-target sequencing regression, and removed all `shadow_backend` instances. The same-machine comparison against the preceding two-backend matrix passed every scenario; the largest mean regression was only 0.24%, so there is no evidence for retaining a second backend.

### Target ownership

```rust
pub struct SoftwareRasterBackend {
    rasterizer: Rasterizer,
    // reusable preparation/bin scratch as justified by profiles
}

pub struct RenderTarget {
    framebuffer: FrameBuffer,
}

pub struct FrameResources {
    background_cache: BackgroundCache,
    shadow_snapshot: Arc<Vec<f32>>, // transitional/fallback
}
```

- [x] Move `Rasterizer` and reusable band-bin/preparation scratch into `SoftwareRasterBackend`. The existing `Rasterizer` is now privately owned by the backend; no additional reusable scratch was introduced without profiling evidence.
- [x] Move the main and shadow framebuffers into separately owned `RenderTarget` values.
- [x] Move background texture caching and the reusable shadow snapshot out of target/backend types.
- [x] Change immediate execution to `backend.execute_phases(&mut target, phases, pipelines_or_shaders)` before adding encoders.
- [x] Start with one backend executing shadow then main. Benchmark target-size switching because the pre-split application had one `Rasterizer` per `Renderer`; retain two backend instances only if the measured scratch behavior is materially better.
- [x] Keep main-target, supersampling, and shadow-target rebuilds independent during hot reload.
- [x] Avoid a general resource registry in this phase.

### Exit Criteria

- One backend can execute against both main and shadow targets sequentially.
- Rebuilding a target does not reload models or recreate unrelated caches.
- Background caching no longer belongs to a framebuffer owner.
- GUI, CLI, rendering tests, and benchmarks use the split types.
- The old `Renderer` type and forwarding path are removed after migration.

## Phase 11.3: Define Safe Pass and Attachment Semantics

Priority: required

Difficulty: medium

Goal: make target use explicit without pretending that color and depth are separately allocated.

Completed slices:

- [x] `c71c66c` (`describe render pass target operations`): added typed `LoadOp`, `Operations`, and `RenderPassDescriptor` values around one exclusive `RenderTarget` borrow; migrated shadow initialization to an explicit depth-only descriptor; preserved color while clearing only depth; and kept constant color+depth clears eligible for one traversal. Dedicated color/depth clears retain the framebuffer's row-parallel task granularity. No shadow clear compatibility path remains, and masked shadow fragments still execute normally. All output hashes remained stable. A same-source-path 30-frame, 1/12-worker reverse comparison passed the 5% full-frame mean gate in all 12 scenarios; repeated matrices also exposed non-code 10-15% variance in short 12-worker scenarios, so future comparisons must use adjacent same-path builds rather than an older saved run alone.
- [x] `327a07c` (`separate background generation`): removed `ClearOptions` and `clear_with_options` rather than retaining a compatibility route; represented solid backgrounds as constant color attachment clears and gradients/images as explicit `BackgroundPass` inputs; and kept `depth clear -> BackgroundPass` fused into one row-parallel framebuffer traversal. Existing attachment/background timing remains one inclusive `attachment_processing` bucket. All matrix output hashes remained stable, the full release suite and image-background CLI path passed, and the repeated same-machine 30-frame matrix (including gradient backgrounds) passed the 5% mean gate in all 12 scenarios. This left the dedicated image-background performance comparison completed by `c0abd97` below.
- [x] `c0abd97` (`benchmark image backgrounds`): added a tracked deterministic 8x8 PNG and an off-screen-geometry image-background scenario to the standard matrix, expanding it to 14 scenarios at 1 and 12 workers. A same-path 30-frame comparison between the old `c71c66c` single-traversal implementation and the `BackgroundPass` implementation preserved the image hash (`ff2d93a1e4445e9f`) and passed the full 5% mean gate. The isolated image case changed by -11.55% at one worker and +1.02% at 12 workers, providing direct evidence that logical pass separation did not add another framebuffer traversal.
- [x] `bf89faf` (`validate render pass descriptors`): added one internal `RenderPassError` path and validates framebuffer dimensions, supersampling layout, sample count, and the presence of at least one declared color/depth/background operation before any attachment write. Immediate initialization now returns `Result`; built-in shadow/main descriptors handle failure as an internal invariant, while Phase 11.5 can reuse the same validation rather than retaining an unchecked path. One combined regression test covers both invalid forms and proves rejection happens before writes, bringing the full release suite to 143 tests. All 14 output hashes remained stable. Because saved and repeated matrices showed large non-code drift, the performance audit used freshly built adjacent old/new implementations plus a reverse-order run; the reverse comparison passed the 5% full-frame mean gate in every scenario, with no credible validation-related regression.
- [x] `254427e` (`name resolve and tonemap pass`): introduces explicit `MainHdrTarget` and `PresentBuffer` types plus a validated `ResolveTonemapPassDescriptor`. GUI, CLI, benchmark, and rendering tests use the same named CPU pass; the former `post_process_to_buffer` entry point is removed. Resolve, exposure, optional ACES, transfer conversion, and packing remain fused in one Rayon traversal. Benchmark schema v2 retains `post_processing_ms`; all 14 output hashes remained unchanged, 147 Rust tests and 9 Node benchmark tests passed, and the same-path five-frame performance gate passed every scenario with a largest mean regression of +2.63%.

### Initial descriptor

```rust
pub struct Operations<T> {
    pub load: LoadOp<T>,
    // Add StoreOp only when its behavior is defined.
}

pub enum LoadOp<T> {
    Load,
    Clear(T),
}

pub struct RenderPassDescriptor<'a> {
    pub label: Option<&'a str>,
    pub target: &'a mut RenderTarget,
    pub color_ops: Option<Operations<Vector3<f32>>>,
    pub depth_ops: Option<Operations<f32>>,
}
```

- [x] Hold one exclusive target borrow in the initial descriptor.
- [x] Treat `color_ops: None` as a depth-only pass and `depth_ops: None` as a color-only pass.
- [x] Validate target layout and required attachment/background declarations before immediate pass initialization. Reuse the same validation from Phase 11.5 encoder finalization rather than adding a second ruleset.
- [x] Do not expose separate mutable color/depth views over interleaved `Sample` storage.
- [x] Defer `StoreOp::Discard` until it has defined observable/debug behavior. A no-op field is not required for a convincing API.

### Background

- [x] Restrict attachment clear operations to constant values.
- [x] Encode a solid background directly as the color attachment clear value when no distinct background operation is needed.
- [x] Move gradients and images into a logical `BackgroundPass` with explicit input resources.
- [x] Preserve backend fusion freedom: when a background fully initializes color and depth also needs clearing, compile the logical `depth clear -> BackgroundPass` sequence into one parallel framebuffer traversal.
- [x] Keep logical pass order and debug labels visible even when operations are physically fused. Attribute fused time to one documented attachment/background timing bucket rather than double counting it.
- [x] Retain the efficient direct parallel target fill; a fullscreen triangle is not required.
- [x] Compare fused output with the current single-traversal clear/background implementation and benchmark gradient and image cases. Do not accept an unconditional extra full-framebuffer traversal merely to mirror GPU pass syntax.
- [x] Preserve frame order: clear, background, opaque, alpha masked, transparent.

### Resolve and tonemap

- [x] Package SSAA resolve, exposure, optional ACES, linear-to-sRGB conversion, and `u32` packing as a named CPU pass.
- [x] Use explicit `MainHdrTarget` and `PresentBuffer` inputs/outputs in the high-level API.
- [x] Keep the operation explicitly named as a CPU resolve-tonemap pass rather than a compute abstraction.
- [x] Keep resolve, exposure, tonemapping, transfer conversion, and packing fused in one parallel output traversal.

### Exit Criteria

- Shadow is represented as a depth-only pass.
- Main load/clear behavior is completely described by its pass descriptor.
- Background generation is no longer hidden in a clear operation.
- Gradient/image backgrounds can still initialize color and clear depth in one backend traversal when fusion preconditions hold.
- Existing output hashes remain unchanged, except for an explicitly reviewed correction.

## Phase 11.4: Separate Shader Algorithms, Pipelines, and Draw Bindings

Priority: required for a meaningful command API

Difficulty: medium to high

Goal: eliminate the per-object fully configured `PbrShader` vector and shader-index indirection.

Completed slices:

- [x] `3659114` (`pass draw context to shader stages`): changed the generic `Shader<C>` contract so both vertex and fragment stages receive the same copyable draw context; migrated PBR, shadow, backend, and external test shaders without a compatibility overload; renamed prepared-triangle storage from fragment context to draw context; and extended the pass-local transformed-vertex cache key with material-reference identity so shared geometry is reused for identical contexts but never reused across distinct contexts. The regression suite now contains 148 Rust tests, including a shader whose vertex position and varying depend on material context. Formatting, release Clippy with warnings denied, release tests/checks, and all 9 benchmark-script tests passed. A same-source-path reverse-order matrix with 3 warmups and 10 measured frames at 1/12 workers preserved all 14 output hashes and passed the 5% full-frame mean gate; the largest regression was +3.67% (`city-threads-1`). This is an interim identity scheme only: the remaining Phase 11.4 work must replace material pointer identity and `shader_index` with typed bindings and pass-local IDs.
- [x] `c9728a4` (`split shadow draw bindings`): generalized backend phases over copyable typed draw contexts, introduced pass-local ObjectBindingId cache identity, split shadow frame/object/material bindings, and replaced the per-object Vec<ShadowShader> with one stateless ShadowShader. Masked-material alpha remains explicit in ShadowMaterialBindings; regressions cover typed-context cache reuse/isolation and distinct shadow object transforms. Formatting, release Clippy with warnings denied, 149 Rust tests, release checks, and all 9 benchmark-script tests pass. An adjacent old-then-new 10-frame matrix at 1/12 workers preserved all 14 hashes and passed the 5% full-frame mean gate; the largest mean regression was +4.96% (city-threads-12). PBR still uses the interim material-pointer identity and per-object shader vector.
- [x] `3a7de3c` (`split pbr draw bindings`): split PBR frame/object/material data into typed bindings, made PbrShader stateless, replaced the per-object shader vector with one reusable algorithm value, and gave pre-transformed transparent geometry a distinct identity binding. Direct shader tests now construct typed contexts, and a main-path regression covers distinct object transforms and materials. The release suite contains 150 Rust tests; formatting, release Clippy with warnings denied, release checks, and all 9 benchmark-script tests pass. An adjacent 10-frame matrix at 1/12 workers preserved all 14 hashes; a reverse-order comparison passed the 5% full-frame mean gate, with the largest regression at +2.87% (default-car-threads-12). The remaining shader_index is now only pipeline/program selection and is deferred to the immutable pipeline slice.
- [x] `87506ff` (`use immutable graphics pipelines`): introduced immutable typed `GraphicsPipeline<S>` values, recorded pipeline references directly in `DrawPacket`, and removed `shader_index` plus the parallel shader-array execution arguments. `VertexProgramId` now separates vertex-affecting shader variants from raster-only pipeline variants in the transformed-vertex cache; regressions prove reuse across fill/line variants and separation across vertex programs. Built-in shadow and PBR passes select reusable immutable state variants without mutating recorded pipelines. The release suite contains 153 Rust tests; formatting, release Clippy with warnings denied, release checks, and all 9 benchmark-script tests pass. A same-machine 10-frame matrix at 1/12 workers preserved all 14 hashes and passed the 5% full-frame mean gate; the largest mean regression was +2.49% (`default-car-threads-12`).
- [x] `a7a6a14` (`complete vertex cache identity coverage`): adds focused regressions proving that pass-local caches are rebuilt for different camera/frame bindings, distinct tangent-frame/object bindings do not alias, and pre-transformed transparent world-vertex slices remain a separate source domain from mesh-local vertices even when IDs and vertex programs match. This closes the required Phase 11.4 cache-identity coverage; cross-frame generation IDs remain deferred because caching is not persistent. Formatting, release Clippy with warnings denied, 156 Rust tests, release checks, and all 9 benchmark-script tests pass.

### Data by update frequency

```text
PbrPipeline / ShadowPipeline
  `- shader algorithm + immutable GraphicsPipelineState

FrameBindings
  |- view/projection and camera position
  |- lights and ambient light
  |- shadow settings
  `- optional ShadowDepthView

ObjectBindings
  |- model matrix
  |- tangent-frame transform
  `- winding variant/override if not encoded in the pipeline key

MaterialBindings
  |- factors and alpha mode
  `- texture bindings
```

- [x] Change the shader/core boundary so the vertex stage receives the same cheap copyable typed draw context as the fragment stage.
- [x] Define typed PBR and shadow draw contexts containing references to frame, object, and material bindings.
- [x] Make PBR and shadow shader algorithm objects stateless or immutable.
- [x] Replace `shader_index` with a typed pipeline reference or pipeline key plus typed bindings.
- [x] Preserve transformed-vertex caching with the frame-local identity scheme below; do not hash floating-point matrices or binding contents.
- [x] Preserve the lower-overhead ordered transparent path until benchmarks justify changing it.
- [x] Keep alpha-mask material access explicit in the shadow pass.
- [x] Prefer typed binding structs. Numeric bind-group slots may be layered on later, but should not replace useful Rust type checking.

### Transformed-vertex cache identity

The current key, `(shader_index, vertex_source)`, indirectly distinguishes per-object transforms because each object owns a configured shader entry. Removing that vector removes the identity guarantee. In the initial replacement:

- [x] Scope the transformed-vertex cache to one recorded pass/submission and keep frame bindings immutable within that pass. View/projection identity is then implicit in the cache lifetime.
- [x] Assign each recorded object binding a pass-local `ObjectBindingId`. Reusing an ID with different model or tangent transforms is invalid encoder state.
- [x] Key cached output by `(vertex_program_id, object_binding_id, vertex_source)` where `vertex_program_id` distinguishes only variants that can change vertex output. Cull, blend, and other fragment/raster-only variants should not unnecessarily split the cache.
- [x] Give the pre-transformed transparent path an explicit identity-transform binding/domain and retain its distinct world-vertex source identity.
- [x] Keep pointer-based mesh/slice source identity only while the cache is frame-local and all borrowed resources outlive submission.
- [x] Defer generation-checked persistent cache IDs because caching remains pass-local; if caching later persists across frames or resource reloads, add generations plus a frame/binding epoch and include mutable frame bindings in the key.
- [x] Add tests showing that shared mesh data is transformed once for identical vertex inputs, but is not incorrectly reused across different object transforms, cameras, tangent transforms, vertex programs, or the transparent identity path.

### Pipeline caching

- [x] Begin with reusable immutable pipeline values.
- [x] Do not add a persistent pipeline cache yet; the small built-in variant sets are constructed once per pass and reused by recorded draws.
- [x] No persistent `PipelineKey` was added; fixed variant arrays compare complete `GraphicsPipelineState` values instead of maintaining a partial hash key.
- [x] GUI cull/wireframe changes select newly built pipeline variants for the frame rather than mutating pipelines already referenced by recorded work.

### Exit Criteria

- Pipeline objects are reusable across scene objects.
- Frame, object, and material update frequencies are explicit.
- `DrawPacket` no longer indexes a parallel shader vector.
- `core::rasterizer` remains generic over shader/context and independent of scene types.
- Static dispatch and transformed-vertex reuse remain intact without hashing `f32` binding data.

## Phase 11.5: Add Typed Command Recording and Synchronous Submission

Priority: required for the modern API goal

Difficulty: medium to high

Goal: record one typed render pass, finish it, and submit it synchronously through the software queue.

Completed slices:

- [x] `6ded6ea` (`record shadow render commands`): introduced typed `RenderDevice`, `CommandEncoder`, `RenderPassEncoder`, one-pass `CommandBuffer`, synchronous `GraphicsQueue`, `SubmissionReport`, and structured command/render errors. The shadow production path now records attachment operations and typed draws, finishes an immutable command buffer, and submits it synchronously; main rendering intentionally remains on the immediate backend path for the next slice. Focused tests cover missing pass/pipeline/bindings, unfinished and duplicate passes, invalid labeled descriptors, record-without-execute behavior, synchronous submission, and draw order. The release suite contains 164 Rust tests plus 9 benchmark-script tests. A same-machine 10-frame 1/12-worker matrix preserved all 14 hashes and passed the 5% full-frame mean gate, with the largest mean regression at +3.74% (`default-car-threads-12`).
- [x] `bed2207` (`record main render commands`): extended one typed render pass with ordered labeled draw phases and per-phase submission reports, then migrated main attachment/background, opaque, masked, and transparent work to `CommandEncoder` and synchronous `GraphicsQueue` submission. Opaque then masked recording order remains unchanged inside the first phase; transparent work has its own reserved phase and is sorted before the command buffer becomes immutable. Attachment/background processing still runs exactly once, and production pass builders no longer call backend initialization or immediate phase execution directly. The existing synchronous command test now also proves that transparent sorting cannot reorder an earlier phase, so the suite remains at 164 Rust tests plus 9 benchmark-script tests. Formatting, release Clippy with warnings denied, release tests/checks, and all benchmark-script tests pass. A fresh local 1/16-worker baseline, repeated candidate, and reverse adjacent `HEAD` run preserved all 14 hashes; the final old-to-new 10-frame comparison passed the 5% full-frame mean gate in every scenario, with the largest mean regression at +1.17% (`default-car-threads-16`).

### First command-buffer model

The required first version records one concrete pass family per command buffer:

```rust
let mut encoder = device.create_command_encoder("shadow");
{
    let mut pass = encoder.begin_render_pass(shadow_descriptor)?;
    pass.set_pipeline(&pipelines.shadow);
    pass.set_frame_bindings(&shadow_frame);
    pass.draw_mesh(mesh, &object, material)?;
    pass.end()?;
}
let shadow_report = queue.submit(encoder.finish()?)?;
```

After `submit` returns, the shadow target is no longer mutably borrowed. The main pass can then borrow shadow output read-only and the main target mutably.

- [x] Add `RenderDevice::create_command_encoder(label)`.
- [x] Add a typed `RenderPassEncoder` that records draw packets without executing.
- [x] Require a selected pipeline and all required typed bindings before a draw is accepted.
- [x] Prevent nested render passes through borrowing and validate unfinished passes in `finish()`.
- [x] Prefer explicit `end(self) -> Result<..., CommandError>`. `Drop` does not panic or hide an error.
- [x] Make `finish()` consume the encoder and return an immutable command buffer.
- [x] Add `GraphicsQueue::submit(&mut self, command_buffer) -> Result<SubmissionReport, RenderError>`.
- [x] Document that `submit` completes all work before returning; do not return a fake fence or submission future.
- [x] Preserve pass and draw order exactly.
- [x] Return structured errors containing resource/pass/pipeline labels where possible.

### Sorting boundary

- [x] Keep phase construction and transparent sort policy in the high-level pass builder, not in `core::rasterizer`.
- [x] Finalize transparent sorting before the phase becomes immutable recorded work.
- [x] Keep insertion IDs as deterministic tie breakers.
- [x] No opaque sorting was introduced. Treat any future experiment as a separate benchmarked optimization and never apply it to transparent work unless depth order remains correct.

### Heterogeneous pass policy

- [x] Do not use `dyn Shader` merely to put shadow and PBR passes in one vector.
- [x] Do not introduce a closed `enum Command::Shadow/Pbr` in the required path unless a non-generic public command buffer is proven more valuable than extensibility.
- [x] Keep separate shadow and main submissions in the first version.
- [x] Treat a frame-wide heterogeneous command stream as Phase 11.6 scope because it needs persistent handles, an explicit pass enum, or measured type erasure.

### Profiling

- [x] Add recording and inclusive synchronous submission totals without changing the definition of existing rasterization timings.
- [x] Report aggregate and labeled per-phase preparation/rasterization durations in `SubmissionReport`.
- [x] Preserve the schema-v2 column set while completing its intended recording and inclusive submission measurements; any future column or timing-definition change must update the schema version and scripts together.
- [x] Measure command allocation, validation, phase finalization, and sorting in recording time rather than raster work.

### Exit Criteria

- Application code records shadow and main work before each respective submission.
- `GraphicsQueue` is the only public execution entry point for recorded rendering work.
- Invalid command state returns deterministic structured errors rather than indexing panics.
- Command recording does not require dynamic shader dispatch.
- Output hashes, worker-count determinism, hot reload, and the performance budget pass.

### Decision Gate

This phase completes the required modernization. Stop here unless a concrete use case needs persistent resources, frame-wide recording, output polymorphism, or a frame graph.

## Required Cleanup Gate After Phase 11.5

Priority: required before declaring the 5.0 modernization complete

Difficulty: low to medium

Goal: remove transitional implementation surfaces, align tests with the approved public boundary, and strengthen weak coverage before optional features begin. This is a correctness and maintainability gate, not a target to minimize the number of tests.

Completed slices:

- [x] `55ce0d9` (`own backend in graphics queue`): made `GraphicsQueue` own its `SoftwareRasterBackend`, added `RenderDevice::create_queue`, migrated application, GUI/CLI, benchmark, shadow, main, and high-level integration-pass callers to retain and submit through the queue, and removed the borrowed-backend queue constructor instead of preserving a compatibility path. Backend-specific integration tests remain temporarily direct until their separate private-test migration. Formatting, release Clippy with warnings denied, 164 release Rust tests, 9 benchmark-script tests, and release checks pass. All 14 benchmark hashes remained stable. Repeated same-machine measurements exposed large run-order and thermal noise; the only remaining aggregate-matrix outlier (`image-background-threads-1`) passed an adjacent reverse-order 500-frame check at +0.11% (18.210 ms old, 18.230 ms new) without a threshold exception.
- [x] `53b09c3` (`route rendering test helpers through queue`): removed dual queue/backend ownership from the shared rendering test harness, routed generic mesh, PBR, shadow, and command-recording setup draws through typed command buffers and `GraphicsQueue`, and isolated the remaining cache/phase implementation tests behind an explicitly named `BackendTestHarness` for their next private-unit-test migration. No compatibility execution helper or new behavior test was added; the suite remains at 164 release Rust tests plus 9 benchmark-script tests, with formatting, release Clippy, and release checks passing. Before this test-only slice, a fresh same-machine 50-frame 1/16-worker recheck of `600382d` versus `55ce0d9` preserved all 14 hashes and passed the 5% mean gate in every scenario; the largest regression was +3.27% (`image-background-threads-16`).
- [x] `7b3fa41` (`move backend invariants to unit tests`): moved seven vertex-cache identity/reuse tests plus the target-switching and empty-phase preparation contracts out of the public integration crate and into a private `pipeline::renderer` test module. Their integration-only shaders, contexts, atomic counters, and direct backend setup were removed without deleting or weakening any distinct test; the release suite still contains 164 Rust tests plus 9 benchmark-script tests. Formatting, release Clippy with warnings denied, release checks, the full release suite, and benchmark-script tests pass. No performance matrix was run because this slice changes only test location and fixtures, not compiled production paths.
- [x] `4a56f4d` (`route integration tests through graphics queue`): migrated the remaining PBR binding, shadow binding, transparent ordering, and worker-count determinism integration tests from direct `RenderPhase`/backend execution to typed render-pass recording and synchronous `GraphicsQueue` submission. The insertion-ID tie-break assertion moved to the private renderer test module because it inspects internal command representation. The public integration crate no longer imports `SoftwareRasterBackend`, `RenderPhase`, `DrawPacket`, or `RenderGeometry`; all 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, and release checks. No performance matrix was run because compiled production paths are unchanged.
- [x] `c59e323` (`hide software backend internals`): made `Rasterizer` crate-private and made `SoftwareRasterBackend`, `BackendExecutionTimings`, `RenderPhase`, and `DrawPacket` private to the renderer module, including their construction, fields, and command-list helpers. Removed the unused public `execute_phase`/`execute_phases` immediate wrappers; `GraphicsQueue` retains the sole submission path and calls one private profiled backend primitive, while the plural preparation routine remains only a private implementation helper for that primitive and backend tests. All 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, and release checks. No performance matrix was run because execution logic is unchanged.
- [x] `b3a3868` (`add render api facade`): added the canonical `render` root for shader contracts, fixed-function pipeline state, typed command recording, synchronous queue submission, render targets, pass helpers, and built-in PBR/shadow bindings. Made the former `core` and `pipeline` roots crate-private and migrated the external rendering integration crate to the façade without compatibility aliases, so user-defined shaders now compile and submit through the intended public path. `FrameBuffer` and `Sample` remain temporarily re-exported only because current target readback signatures still return them; a dedicated readback boundary must replace that exposure before the final public-surface audit. All 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, release checks, and a warning-free rustdoc build. No performance matrix was run because this slice changes visibility, re-exports, imports, and documentation only.
- [x] `5465122` (`hide framebuffer behind readback`): added a zero-copy `RenderTargetReadback` view with explicit logical dimensions, sample-grid dimensions, resolved HDR color, per-sample color/depth, and copied depth-attachment access. Migrated application settings, resolve-tonemap code, and all public rendering integration tests to this boundary; removed `FrameBuffer` and `Sample` from `render`, made both types and their remaining operations crate-private, and moved the direct framebuffer supersampling contract into its private unit-test module without changing the 164-test total. All 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, release checks, and a warning-free rustdoc build. No performance matrix was run because the readback view delegates directly to the same storage and execution paths are unchanged.
- [x] `1861fe7` (`restrict internal module visibility`): changed all 14 module declarations beneath the crate-private `core` and `pipeline` roots from unrestricted `pub` to `pub(crate)` while preserving the intentional `render` re-exports. The library `unreachable_pub` audit dropped from 30 warnings to 16; every remaining warning is now confined to internal color/math functions or the private render-phase sorting method for the next cleanup slice. All 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, release checks, and a warning-free rustdoc build. No performance matrix was run because this slice changes visibility qualifiers only.
- [x] `6f4b648` (`restrict internal helper visibility`): resolved the remaining 16 `unreachable_pub` findings by making internal color conversion, interpolation, transform helpers, and `TransformFactory` crate-visible only, while making the backend `RenderPhase::sort_transparent` method private and preserving the public `RenderPassEncoder::sort_transparent` API. `cargo rustc --release --lib -- -D unreachable-pub` now passes without findings. All 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, release checks, and a warning-free rustdoc build. No performance matrix was run because this slice changes visibility qualifiers only.
- [x] `99178c0` (`encapsulate pbr object transform`): audited public command/resource construction and confirmed that command buffers, encoders, queues, render targets, present storage, frame resources, and immutable graphics pipelines already keep owned state private, while public render-pass/state structs are intentional authoring descriptors. Closed the first binding-specific hole by making `PbrObjectBindings::model_matrix` private so it cannot diverge from the tangent-frame transform derived at construction, and added a read-only accessor covered through the public integration crate. `PbrFrameBindings` still needs a separate decision for validating its coupled shadow-map fields before the command/resource-field exit check can close. All 164 release Rust tests and 9 benchmark-script tests pass with formatting, warnings-denied Clippy, release checks, a warning-free rustdoc build, and a clean warnings-denied `unreachable_pub` audit. No performance matrix was run because execution logic is unchanged.
- [x] `535473c` (`validate pbr shadow bindings`): replaced the eight independently mutable shadow fields in `PbrFrameBindings` with an optional `PbrShadowBindings` value whose private fields can only be produced through a public descriptor and structured validation error. Construction rejects zero, overflowed, or mismatched map dimensions, non-finite light-space matrices, negative/non-finite biases, and negative PCF kernels; the fragment hot path now trusts that invariant instead of rechecking map dimensions per shaded fragment. The built-in main pass safely treats inconsistent optional `ShadowPassOutput` metadata as no shadow, matching the previous failure behavior, while direct public construction reports the exact error. One consolidated contract test raises the release suite to 165 Rust tests rather than duplicating each case across layers. Formatting, warnings-denied Clippy, release checks/tests, 9 benchmark-script tests, rustdoc, `unreachable_pub`, and a release car-scene render all pass. All 14 output hashes remained stable. An initial 10-frame matrix contained one contradictory `city-threads-16` outlier; the required reverse-order adjacent rerun passed the 5% mean gate in all scenarios, with that case at +0.21% and the largest remaining mean regression at +2.04%. This closes the command/resource-field construction check recorded in `docs/render-api-5.md`.
- [x] `6d04329` (`remove obsolete phase execution path`): completed the superseded-name and compatibility-path scan across production code, tests, README, and API documentation. Legacy execution/state identifiers no longer occur in compiled code; remaining mentions are deliberate historical migration records. Removed the private plural backend execution helper left by the immediate renderer, so `GraphicsQueue` submission and backend tests now share the sole single-phase execution primitive without first allocating and flattening a `Vec<&DrawPacket>`. The former cross-phase test now preserves its meaningful empty-mesh preparation contract inside one phase instead of retaining an unreachable execution model; the suite remains at 165 Rust tests. Formatting, warnings-denied Clippy, release checks/tests, 9 benchmark-script tests, rustdoc, `unreachable_pub`, and a release car-scene render all pass. All 14 hashes remained stable. The first ordered matrix exhibited broad unrelated 16-worker slowdown; the reverse-order adjacent 10-frame rerun passed every 5% mean gate with no candidate mean regression, confirming environmental variance rather than a code regression.
- [x] `a53273e` (`encapsulate shadow pass output`): audited shadow resources, `FrameResources`, backend/submission timings, and benchmark consumers. Every timing field and both reusable resource owners remain live; the removable invalid state was the public `ShadowPassOutput` field tuple. Replaced independent optional depth/light metadata plus disabled sentinel size/matrix values with one private optional shadow-map record, public read-only accessors, an explicit `disabled()` constructor, and an `Arc`-cloning `depth_snapshot()` for callers that retain depth beyond the output borrow. Enabled outputs can no longer represent mismatched depth, size, matrix, and light index, while disabled outputs carry no meaningless placeholders. Existing public rendering tests were migrated without adding or deleting a distinct behavior test, so the suite remains at 165 Rust tests. Formatting, warnings-denied Clippy, release checks/tests, 9 benchmark-script tests, rustdoc, `unreachable_pub`, and a release car-scene render all pass. All 14 hashes remained stable. The initial ordered matrix had three contradictory outliers, including two scenarios with shadows disabled; the reverse-order adjacent 10-frame rerun passed every 5% mean gate with a largest regression of +1.23%, confirming environmental variance.
- [x] `3d06fe2` (`remove duplicate pass validation`): made `CommandEncoder::begin_render_pass` the single validation boundary before a borrowed target enters an encoded command. The private command representation and retained mutable borrow prevent the validated descriptor or target shape from changing before submission, so the software backend now only processes already-validated attachments and no longer repeats validation or maps an impossible submission error. Removed the resulting dead encoded-pass label and unused `BackgroundPass` label without retaining compatibility state; production and nearby test code shrank by 54 net lines while the suite remains at 165 Rust tests. Formatting, warnings-denied Clippy, release checks/tests, 9 benchmark-script tests, rustdoc, `unreachable_pub`, and a release car-scene render all pass, and all 14 output hashes remain stable. The two ordered 10-frame matrices produced different isolated 16-worker outliers; two symmetric 100-frame `city-threads-16` checks passed at +0.71% and +2.39%, identifying scheduling variance rather than a stable regression. Together with the preceding ownership, profiling-field, obsolete-path, and feature-surface audits, this closes the production dead-state and duplicated-validation cleanup item.
- [x] `998a279` (`document exclusive band ownership`): completed the final framebuffer concurrency and mutation audit. Production code contains no `unsafe`, `UnsafeCell`, atomics, mutexes, read-write locks, manual `Send`, or manual `Sync`; the only atomics are vertex-call counters in private backend tests. `FrameBuffer` keeps its sample vector private, row clear/fill operations receive disjoint mutable slices from Rayon, rasterization receives exclusive eight-row bands (including the shorter final band), clamps prepared coordinates to the target, and converts global rows to band-local offsets before indexing. Resolve/tonemap similarly partitions a separate present buffer while reading the render target immutably. Texture `OnceLock` values only publish immutable mip data and do not participate in framebuffer mutation. Added an ownership-boundary comment rather than another test: the existing height-70 cross-band test checks the final row of a non-multiple-of-eight target, while the transparent determinism test compares complete color and depth output between one and four workers. Both targeted contracts, all 165 release Rust tests, 9 benchmark-script tests, formatting, warnings-denied Clippy, release checks, rustdoc, `unreachable_pub`, and a release car-scene render pass. No performance matrix was run because the only repository change is a comment.
- [x] `9f03909` (`move rasterizer invariants to backend tests`): moved the near-plane clipping, cross-band/final-short-band, and top-left shared-edge contracts from the public rendering integration crate into the private renderer backend test module, alongside the vertex-cache and insertion-order invariants migrated earlier. Removed the integration-only coverage shader, reused one explicit private mesh-execution helper and one full-screen-quad fixture, and kept every distinct assertion; the release suite remains at 165 Rust tests, redistributed from 93 private plus 42 rendering tests to 96 private plus 39 rendering tests. The integration crate imports only the supported `render`, `scene`, `io`, and `error` surfaces and contains no backend implementation type or private module path. A coverage audit confirms independent tests remain for depth, culling, mirrored winding, blending, fragment discard, transparent total order, worker-count determinism, masked shadows, hot reload, and output hashes. All three targeted migrations, formatting, warnings-denied Clippy, release checks/tests, 9 benchmark-script tests, rustdoc, and `unreachable_pub` pass. No render or performance matrix was run because compiled production paths are unchanged. Together with `7b3fa41` and `4a56f4d`, this closes the test-boundary migration and required-behavior preservation items.
- [x] `e66ae7e` (`strengthen mip selection tests`): replaced the resized four-color and uniform mip fixtures with one explicit 4x4 data texture whose generated values at the sampled coordinate are independently predictable at the base, 2x2, and 1x1 levels (`0`, `64/255`, and `112/255`). The density test now proves exact selection of all three generated levels; the non-mip nearest and linear cases assert distinct base-level results instead of comparing two calls; and all four mip minification modes assert either the exact 2x2 value or the exact halfway blend (`32/255`) rather than only finiteness and alpha. Clamp-to-edge is explicit so wrap behavior cannot contaminate the mip-selection contract. The existing trilinear color-space test already asserted its decoded blend and remains unchanged. No test was added or removed, so the release suite remains at 165 Rust tests plus 9 benchmark-script tests. Formatting, warnings-denied Clippy, release checks/tests, rustdoc, and `unreachable_pub` pass. No render or performance matrix was run because compiled production paths are unchanged.
- [x] `e87f323` (`consolidate gltf fixture checks`): folded five standalone glTF fixture-precondition test shells into the importer tests that consume the same fixtures. Texture/source index structure now precedes shared-image binding assertions; nested-node names/hierarchy precede loaded-mesh assertions; indexed/non-indexed triangle modes precede converted index-list assertions; malformed attribute/index parseability precedes each contextual importer error; and unsupported primitive/image reachability precedes the corresponding structured error checks. Every structural and importer assertion remains, with fixture-specific panic messages preserved or strengthened, while the glTF integration target drops from 21 to 16 tests and the complete release suite from 165 to 160 tests. The file shrinks by 10 net lines without merging unrelated behaviors. Formatting, warnings-denied Clippy, release checks/tests, all 9 benchmark-script tests, rustdoc, and `unreachable_pub` pass. No render or performance matrix was run because compiled production paths are unchanged.
- [x] `0647896` (`clarify frame resource reuse tests`): retained both pointer-identity tests after tracing their production call sites and ownership semantics. Main rendering requests the configured background binding every frame, so identical `Arc` identity is the observable proof that an unchanged path/mip policy does not reload the image or allocate a replacement binding, while a policy change still invalidates the cache. Shadow submission copies depth every frame; after consumers release the previous snapshot, stable allocation identity proves `Arc::make_mut` regains and reuses the owned vector across target rebuilds instead of allocating each frame. These are intentional `FrameResources` performance responsibilities exercised by the standard image-background and shadow benchmark scenarios; Phase 11.6 has not selected a replacement resource model. Renamed both tests and documented what pointer identity observes, including the released-snapshot precondition, without adding another allocation-detail test. The suite remains at 160 release Rust tests plus 9 benchmark-script tests. Formatting, warnings-denied Clippy, release checks/tests, rustdoc, and `unreachable_pub` pass. No render or performance matrix was run because compiled production paths are unchanged.

### Production code and API cleanup

- [x] Search the repository for superseded type names, module paths, methods, fields, aliases, adapters, and comments introduced by the 4.0 execution model. Remove obsolete paths rather than deprecating or forwarding them inside 5.0.
- [x] Ensure application code, examples, GUI, CLI, benchmarks, and public integration tests submit through `GraphicsQueue`; only backend-internal tests may call `SoftwareRasterBackend` execution directly.
- [x] Make `Rasterizer`, `SoftwareRasterBackend`, `RenderPhase`, `DrawPacket`, preparation scratch, and band-bin types private implementation details behind the `render` façade.
- [x] Remove transitional immediate-pass helpers that are no longer used by command submission. Keep the backend execution primitive used by `GraphicsQueue`; do not preserve a second public entry point.
- [x] Remove dead state, duplicated validation, redundant resource ownership, stale profiling fields, and unused feature scaffolding. Do not add placeholder abstractions merely to make the cleanup look architecturally complete.
- [x] Run a final `unsafe`, `UnsafeCell`, atomics, locks, and manual `Sync` audit around framebuffer mutation and preserve exclusive-band writes.

### Test cleanup and migration

- [x] Migrate user-observable rendering tests to the public `render` façade and typed command path. Move top-left rules, clipping, band boundaries, vertex reuse, and other backend invariants into private backend unit tests instead of keeping internals public for integration tests.
- [x] Preserve behavior coverage for depth, culling, mirrored winding, blending, discard, transparent total order, worker-count determinism, masked shadows, hot reload, and output hashes. Do not delete a distinct behavior test solely to reduce the test count.
- [x] Replace weak mip-selection smoke tests with fixtures whose mip levels have distinct expected values, then assert the selected level and blend result rather than only finiteness or unchanged uniform color.
- [x] Consolidate fixture-structure precondition tests into their importer tests only when doing so retains equally clear failure diagnostics.
- [x] Revisit pointer-identity tests for cached backgrounds and reusable shadow snapshots. Keep them only while allocation reuse remains an explicit measured contract; replace or remove them if Phase 11.6 selects a different resource model.
- [ ] Remove duplicated helpers and oversized setup only when a shared builder makes the tested contract clearer. Avoid a generic test framework that hides pass order, state, bindings, or expected pixels.

### Documentation and validation

- [ ] Update `README.md`, `docs/render-api-5.md`, examples, module documentation, and benchmark documentation so historical 4.0 inventory is clearly distinguished from the implemented 5.0 API.
- [ ] Run formatting, Clippy with warnings denied, release checks, the complete Rust and benchmark-script test suites, and the full release benchmark matrix.
- [ ] Confirm deterministic hashes and the 5% full-frame mean budget against the immediately preceding required-phase baseline; investigate any cleanup-only output or timing change rather than accepting it as architectural churn.

### Exit Criteria

- The approved 5.0 façade is the only public rendering construction and submission path.
- No obsolete execution aliases, compatibility wrappers, dual ownership paths, or unused transitional types remain.
- Tests cover the same distinct behaviors through the correct public or private boundary, and known weak mip assertions have been strengthened.
- Documentation, examples, GUI, CLI, tests, and benchmarks describe and exercise the implementation that actually ships.
- The full correctness, determinism, hot-reload, output-hash, and performance gates pass.

## Phase 11.6: Optional Direct Shadow View and Persistent Resources

Priority: optional, evidence-driven

Difficulty: medium to high

Goal: remove avoidable resource copies or borrowed-command lifetime limits where measurements or features justify it.

### Direct shadow sampling

- [ ] After shadow submission, expose a read-only `DepthTextureView<'a>` over the completed shadow target.
- [ ] Let the main command buffer borrow that view immutably while borrowing the distinct main target mutably.
- [ ] Make the view sample interleaved `Sample` storage without exposing mutable aliases.
- [ ] Compare direct strided/interleaved access with the current contiguous `Arc<Vec<f32>>` snapshot. The avoided copy may be offset by worse PCF cache behavior.
- [ ] Retain the snapshot path if it is faster, simpler, or required by consumers that outlive the producing target borrow.
- [ ] Test large shadow maps, PCF on/off, and one/all workers.

### Persistent handles

Add handles only if command buffers must outlive frame-local borrows, be cached/replayed, or contain multiple dependent pass types.

```text
BufferHandle<T>
TextureHandle<T>
PipelineHandle<P>
BindGroupHandle<B>
```

- [ ] Use typed generation-checked handles backed by a centralized arena/registry.
- [ ] Return structured stale-handle, usage, dimension, and missing-resource errors.
- [ ] Define hot-reload invalidation and replacement semantics before storing handles in commands.
- [ ] Add resource usage flags only when validation consumes them.
- [ ] Preserve `Mesh` as the scene-level vertex/index/material grouping.
- [ ] Preserve texture image sharing already provided by `Arc`; do not duplicate it merely for terminology.
- [ ] Design frame-wide command buffers only after the handle lifetime and heterogeneous dispatch policy are settled.

### Exit Criteria

- The selected resource model solves a demonstrated lifetime, replay, or performance problem.
- No raw IDs, dangling borrows, hidden global registry, or fake GPU allocation concepts are introduced.
- Hot reload has explicit behavior for replaced resources and recorded commands.

## Phase 11.7: Shared Frame Execution and Optional Output Abstraction

Priority: recommended cleanup after Phase 11.5; generic surfaces remain optional

Difficulty: medium

Goal: remove duplicated GUI/CLI/benchmark frame orchestration before generalizing presentation.

- [ ] Introduce one high-level `FrameRenderer::render_frame` or equivalent that records/submits shadow and main passes and runs resolve/tonemap.
- [ ] Return a present buffer plus structured frame/submission timings.
- [ ] Route GUI, CLI, and benchmark through this function.
- [ ] Preserve GUI hot reload and its distinct target/resource/window rebuild policies.
- [ ] Keep output side effects outside render passes: GUI presents, CLI saves PNG, benchmark hashes.

Only add a trait if multiple output implementations benefit from the same lifecycle:

```rust
pub trait Surface {
    type Error;

    fn dimensions(&self) -> (usize, usize);
    fn present(&mut self, frame: &[u32]) -> Result<(), Self::Error>;
}
```

Possible implementations are `WindowSurface`, `ImageSurface`, and `BenchmarkSurface`. Do not add `acquire_frame` or `SwapChain` terminology unless ownership or buffering behavior actually requires it.

### Exit Criteria

- GUI, CLI, and benchmarks share frame recording/execution.
- Presentation, saving, and hashing remain outside rendering commands.
- A `Surface` trait exists only if it reduces real duplication or enables a concrete consumer.

## Phase 11.8: Optional Minimal Frame Graph

Priority: defer until the pass/resource model has nontrivial dependencies

Difficulty: high

Trigger this phase only if the renderer gains optional post-process branches, multiple shadow maps, multiple outputs, transient targets, or capture/replay needs. A fixed shadow-main-tonemap chain does not require a graph.

Possible minimal graph:

```text
ShadowPass --writes--> ShadowDepth
                         |
                         v
MainPass ----reads-------+----writes--> MainHdr
                                            |
                                            v
ResolveTonemapPass ----------------------reads----writes--> PresentBuffer
```

- [ ] Record typed resource reads/writes and pass labels.
- [ ] Reject transient read-before-write and incompatible same-pass access.
- [ ] Preserve declared order initially.
- [ ] Compile nodes into the established command-buffer/submission model.
- [ ] Export a readable graph or DOT file for education/debugging.
- [ ] Add topological sorting, pass culling, and transient reuse only when real graphs benefit.
- [ ] Do not add automatic parallel pass execution without a proven safe resource model and benchmark evidence.

## Error and Validation Model

The command API creates new invalid states that should not become panics.

Recommended categories:

```text
CommandError
  |- PassAlreadyActive
  |- PassNotEnded
  |- MissingPipeline
  |- MissingBindings
  |- IncompatiblePipeline
  |- InvalidTarget
  `- InvalidResourceUsage        # only after usage flags exist

RenderError
  |- Command(CommandError)
  |- Target(...)
  |- Resource(...)              # only after a registry exists
  `- Backend(...)
```

- Validation errors should include labels and stable context.
- Cheap structural validation is always enabled.
- Expensive redundant validation may be feature-gated only after profiling.
- Internally impossible states may use assertions; user-constructible command state must return `Result`.

## Migration Rules

- Implement each phase as a vertical slice through GUI, CLI, benchmark, and tests.
- Introduce the replacement, migrate all call sites, verify, then remove the superseded execution path in the same phase.
- Temporary adapters are allowed inside one branch/phase but are not an exit state.
- Do not combine a public rename, module move, ownership rewrite, and performance optimization in one commit.
- Do not optimize opaque ordering, framebuffer layout, or shader math during command API migration.
- Keep scene configuration and rendered output stable unless a separately documented behavior fix is approved.
- Update README architecture text and examples when public API names or module paths change.

## Recommended Implementation Order

```text
11.0 contracts, tests, benchmark schema, 5.0 decision
  -> review
11.1 vocabulary and graphics state
  -> review
11.2 backend/target/resource ownership split
  -> review
11.3 pass descriptors, background, resolve/tonemap
  -> review
11.4 immutable pipelines and typed draw bindings
  -> review
11.5 typed command recording and synchronous queue submission
  -> required cleanup and public-surface audit
  -> required modernization complete
11.6 direct shadow view / persistent handles only when justified
11.7 shared frame execution; Surface only when useful
11.8 frame graph only after a concrete trigger
```

The essential change from the previous order is that ownership, attachment safety, and shader data boundaries are settled before command recording. The encoder then describes a stable execution model instead of becoming an adapter around the current coupled `Renderer`.

## Suggested Commit Boundaries

Use focused commits; split further when a reviewable boundary appears.

1. `characterize render api behavior`
2. `version benchmark timing schema`
3. `rename draw submission concepts`
4. `define graphics pipeline state`
5. `separate raster backend and targets`
6. `move reusable frame resources`
7. `describe render pass target operations`
8. `separate background generation`
9. `name resolve and tonemap pass` (`254427e`, completed)
10. `split pbr frame object material bindings`
11. `make shader algorithms reusable`
12. `record typed render pass commands`
13. `submit commands through graphics queue`
14. `clean transitional render code`
15. `share frame execution paths`
16. `sample shadow target directly` (optional)
17. `add persistent resource handles` (optional)
18. `add output surfaces` (optional)
19. `build minimal frame graph` (optional)

## Standard Validation for Every Subphase

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets --all-features -- -D warnings
cargo test --release
cargo check --release
```

For changes affecting order, state, resources, framebuffer access, shader bindings, pass boundaries, resolve, or output:

- run the release benchmark matrix with one and the configured all-worker count;
- compare hashes when floating-point operation order is unchanged;
- inspect release renders when a deliberate operation-order change makes a hash change legitimate;
- compare same-machine, same-toolchain full-frame mean and p95 against a freshly captured immediately preceding baseline; never use a repository-stored result from another environment as the performance denominator;
- report recording, preparation, rasterization, post-processing, submission, and full-frame timings using the versioned definitions;
- test GUI hot reload for live settings, supersampling rebuild, shadow-map rebuild, asset reload, and rejected window resize;
- confirm transparent ordering and insertion-ID ties remain stable;
- confirm shadow alpha masks still discard correctly;
- confirm no framebuffer locks, atomics, `UnsafeCell`, manual `Sync`, or new `unsafe` code are introduced.

Always render performance-sensitive scenarios with `--release`.

## Required Completion Criteria

The required modernization, Phases 11.0 through 11.5, is complete when:

- the required post-11.5 cleanup gate has passed;
- the supported 5.0 rendering surface is exposed through one intentional `render` façade, including statically dispatched user-defined shaders;
- backend implementation types such as `Rasterizer`, `RasterPrimitive`, `RenderPhase`, and `DrawPacket` are not public construction APIs;
- application code records typed render-pass work into command buffers;
- a project-level `GraphicsQueue` submits those buffers synchronously to the software backend;
- shadow and main passes use separate sequential submissions without fake synchronization;
- render phases and draw packets use semantically accurate names;
- immutable shader/pipeline state is separate from frame, object, and material bindings;
- render-target ownership and attachment operations are explicit and safe for interleaved storage;
- background and resolve/tonemap behavior are no longer hidden inside ambiguous clear/post-process helpers;
- command validation returns structured errors;
- GUI, CLI, tests, and benchmarks use the new execution path;
- safe exclusive-band rasterization and transparent total order are preserved;
- benchmark schema v2 separates pass setup, recording, attachment/background work, backend preparation, rasterization, inclusive submission totals, post-processing, and complete-frame time using name-based script parsing;
- deterministic output and the agreed release performance budget pass;
- obsolete public execution types and compatibility aliases are absent in the 5.0 API.

Persistent handles, direct shadow views, output surfaces, and a frame graph are not required for completion. They remain follow-up work only when they solve a measured or demonstrated problem.
