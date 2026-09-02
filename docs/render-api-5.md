# Render API 5.0 Boundary

## Status

Accepted as the Phase 11.0 API-boundary decision for the planned 5.0 refactor.

This is a design contract for implementation, not documentation for the current 4.0 API. Concrete method signatures may be refined during Phases 11.1–11.5, but changes must preserve the visibility and extensibility decisions below or explicitly amend this record first.

## Decision

Version 5.0 will provide a small public rendering façade, continue to support statically dispatched user-defined shaders, and stop exposing the software rasterizer's command representation and execution internals.

The intended public roots are:

```rust
pub mod render;
pub mod scene;
pub mod io;
```

Public error types will be re-exported from the module whose operations produce them. A separate root `error` module is not part of the required stable surface.

The final 5.0 library will not expose `core`, the current high-level `pipeline` module, `app`, `benchmark`, or `ui` as stable public modules.

## Why This Boundary Is Required

The 4.0 crate exports almost its entire source tree through `lib.rs`. Integration tests currently import all of these layers:

- programmable types from `core::shader`;
- `Vertex` and `FrameBuffer` from `core`;
- raster state and `Rasterizer` from `core::rasterizer`;
- immediate draw lists and `Renderer` from `pipeline::renderer`;
- built-in shaders and pass functions from `pipeline`;
- cameras, lights, materials, meshes, models, textures, and `RenderScene` from `scene`;
- configuration and glTF loading from `io`.

This is useful test coverage but not a deliberate stability boundary. In particular, `Rasterizer` is public even though its preparation and execution methods are crate-private, while `Renderer`, `RenderPhase`, and shader-index coupling expose the implementation that Phase 11 replaces.

The new boundary distinguishes supported programmability from backend representation:

- users may define shaders, resources, pipelines, and render-pass commands;
- users may not directly construct prepared primitives, band bins, or backend draw packets.

## Supported User-Defined Shaders

User-defined shaders remain a 5.0 feature.

The stable `render` façade will expose the types required to implement them, including the equivalents of:

- `Shader`;
- `Interpolatable` or its replacement varying contract;
- `Vertex`;
- `FragmentInput` and `FragmentOutput`;
- typed draw/binding context requirements;
- graphics pipeline descriptors and state.

The shader interface must satisfy these rules:

- shader algorithms use static generic dispatch in the required implementation;
- a user shader can provide its own varying and typed draw context;
- both vertex and fragment stages can access the context needed by that stage;
- `Send + Sync` requirements remain explicit because preparation and shading may execute through Rayon;
- the public trait does not mention `Material`, `PbrShader`, `PreparedTriangle`, framebuffer bands, or backend caches;
- the command API does not require users to implement `dyn Shader` or select a closed built-in shader enum.

Built-in PBR and shadow pipelines use the same public shader/pipeline model. They may be offered under a namespace such as `render::builtin`, but they do not receive privileged command variants unavailable to custom shaders.

## Public `render` Surface

The stable rendering façade is expected to contain these capability groups.

### Shader programming

- shader and varying traits;
- vertex and fragment I/O;
- the supported texture-coordinate-set limit or a query replacing the constant;
- public math types used by the shader contract.

### State and descriptors

- `GraphicsPipelineState` and its primitive, depth/stencil, and color-target descriptors;
- `PrimitiveTopology`, `FrontFace`, `CullMode`, `PolygonMode`, and `CompareFunction`;
- the implemented blend state;
- render-pass attachment operations;
- validated target, pipeline, and binding descriptors.

### Recording and execution

- `RenderDevice`;
- `GraphicsQueue`;
- typed `CommandEncoder`, `RenderPassEncoder`, and `CommandBuffer`;
- immutable pipeline values;
- typed frame, object, and material/custom bindings;
- synchronous `SubmissionReport` and structured command/render errors.

### Resources and output

- safe render-target creation and read-only output/readback operations;
- public mesh, texture, sampler, and binding types through canonical façade paths or the public `scene` module;
- `MainHdrTarget`, `PresentBuffer`, and the validated fused CPU resolve-tonemap pass.

Resource-owning and command types keep their fields private. Construction goes through validated constructors, descriptors, or encoders. Data containers intended for authoring, such as configuration values and mesh vertex/index data, may retain public fields when mutation is part of their supported use.

## Public `scene` Surface

`scene` remains public because constructing and manipulating renderable content is a supported library use case, not merely application plumbing.

Its intended stable concepts include:

- camera and projection types;
- lights;
- materials and alpha modes;
- meshes and models;
- scene objects and transforms;
- texture images, sampler state, texture bindings, and UV selection;
- `RenderScene`, replacing `RenderContext`.

Scene types may refer to canonical types re-exported from `render`, but backend types must not leak into their public fields or method signatures.

## Public `io` Surface

`io` remains public for supported configuration, asset loading, and image output workflows.

Its intended stable concepts include:

- typed TOML configuration and validation;
- relative-path resolution;
- supported glTF loading;
- packed-frame image output;
- contextual public errors for those operations.

Configuration structs remain ordinary serializable data with public fields where direct editing is intentional. Renderer caches, hot-reload classification, minifb integration, and CLI dispatch do not belong to `io`.

The broad `io` name may be split later only as a separate reviewed module-layout change. Phase 11 does not require an `asset` rename.

## Internal Implementation Types

The following concepts are not public construction APIs in 5.0:

- `Rasterizer`;
- `RasterPrimitive` / current `PreparedTriangle`;
- `SoftwareRasterBackend`;
- framebuffer samples and mutable band views;
- band bins and preparation scratch;
- `RenderPhase`;
- `DrawPacket`;
- shader-index tables and vertex-source pointer keys;
- `ClearOptions`;
- the current coupled `Renderer`;
- immediate `draw_phase` and `draw_phases` execution methods.

`RenderPhase` and `DrawPacket` remain useful internal names. Their fields are encoded through the public `RenderPassEncoder`; a finished `CommandBuffer` exposes labels and safe inspection only where a concrete debugging use case requires it.

`FrameBuffer` is an implementation detail. `RenderTarget::readback` and `MainHdrTarget::readback` return a read-only `RenderTargetReadback` view for dimensions, resolved HDR colors, individual sample colors/depths, and explicit depth copying; public callers cannot access or mutate the interleaved `Sample { color, depth }` storage.

## Binary and Tooling Modules

The current `app`, `benchmark`, and `ui` modules are public largely because the package's binary target imports the library crate. They are not stable 5.0 rendering API.

Before 5.0 release, their implementation should move behind one of these non-library boundaries:

- binary-local modules under the rasterizer executable;
- a separate workspace/tool crate;
- a narrowly scoped public application entry point only if an actual external consumer is identified.

Using `#[doc(hidden)] pub` alone is not the desired final state because it remains callable public Rust API. Benchmark report formats may remain documented tooling contracts without making the benchmark runner part of the rendering library façade.

## Math-Type Policy

The 5.0 public shader and scene APIs will continue using `nalgebra` vectors, points, and matrices rather than adding project-specific wrapper math types during this refactor.

Consequences:

- the supported `nalgebra` major/minor compatibility range affects the public source API;
- dependency upgrades that change exposed types require normal compatibility review;
- internal raster types should expose only the math values needed by supported shader/scene contracts.

Replacing the math vocabulary is outside Phase 11 and would require its own migration plan.

## Command Extensibility Policy

The first command buffer is typed and may contain one concrete render-pass family. Shadow and main work are submitted separately and synchronously.

A `GraphicsQueue` is created by `RenderDevice` and owns the software execution backend used for submission. Application code, pass builders, and other public callers record work and submit it through the queue; they do not construct, borrow, or pass a `SoftwareRasterBackend` alongside it.

A typed render pass may seal multiple ordered draw phases before it ends. Attachment and background operations still run once for the pass; `GraphicsQueue` executes the sealed phases in order and returns both aggregate timings and labeled per-phase reports. Sorting applies only to the currently recorded phase, so the built-in main pass keeps opaque/masked draws in recorded order and finalizes back-to-front transparent order before the command buffer becomes immutable.

This is a public capability limitation, not a built-in-pipeline limitation:

- custom pipeline types can use the same typed encoder and queue path;
- command-buffer fields and backend phases remain private;
- a frame-wide heterogeneous stream is optional future work;
- heterogeneous recording must not be achieved by exposing backend enums or forcing `dyn Shader` without a separate decision and measurements.

## Transparent Ordering Contract

Transparent alpha blending has one observable total-order contract across the public command model and private software backend:

- the high-level pass builder computes a view-space depth key for each transparent primitive;
- visible geometry uses negative view-space Z, so ascending Z is back-to-front;
- equal depth keys retain submission order through a monotonically increasing insertion ID;
- command expansion retains mesh triangle/index order;
- clipping retains the generated triangle-fan order for each source primitive;
- parallel preparation and collection retain the complete encoded primitive order;
- each framebuffer band appends primitive indexes in that order and visits them sequentially;
- Rayon workers partition disjoint pixel bands only and never define blend order.

This order must remain identical across worker counts. Pipeline/material sorting is not permitted for transparent phases unless it provably preserves the same depth and insertion order. The contract applies to built-in and user-defined pipelines that use order-dependent blending.

## Compatibility and Migration

- The change targets 5.0.0; it is not compatible with the 4.x module layout.
- Old `core::*`, `pipeline::*`, `Renderer`, and `RenderQueue` paths do not receive permanent aliases.
- Integration tests migrate to the same public `render` façade expected of external users.
- Backend-only tests move into unit-test modules when they need private access.
- Public examples and README snippets use only canonical façade paths.
- Public types referenced in signatures must have one documented canonical path even if a temporary internal re-export exists during migration.
- Once the module migration is complete, enable or run an `unreachable_pub` audit so accidentally public items inside private modules do not recreate an implicit API surface.

## Phase Exit Checks Derived From This Decision

Before Phase 11.1 begins:

- [x] custom user shaders are an approved 5.0 capability;
- [x] `render`, `scene`, and `io` are the intended stable public roots;
- [x] `core` and the current `pipeline` tree are implementation details in the target API;
- [x] `Rasterizer`, `RenderPhase`, and `DrawPacket` are internal;
- [x] built-in PBR/shadow pipelines use the same typed command model as custom shaders;
- [x] binary/tooling modules are outside the stable rendering-library surface;
- [x] `nalgebra` remains the public math vocabulary for this refactor.

During Phases 11.1–11.5:

- [x] add compile-time integration coverage for a minimal external custom shader through `render`;
- [x] migrate existing rendering integration tests away from private module paths;
- [ ] make command/resource fields private and validate construction;
- [x] remove obsolete public execution paths after their replacements pass validation;
- [ ] audit the final rustdoc/public item list against this record.
