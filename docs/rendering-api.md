# Rendering API

## Crate Boundaries

The canonical rendering surface is `rasterizer_rust::render`. It supports statically dispatched
user shaders, immutable graphics pipelines, typed command recording, synchronous queue submission,
safe render targets, built-in PBR/shadow passes, and read-only output access.

The library exposes three capability roots:

```rust
pub mod render;
pub mod scene;
pub mod io;
```

`core` and `pipeline` are crate-private. The rasterizer, software backend, recorded draw packets,
framebuffer storage, preparation scratch, and band bins cannot be named or constructed by external
users. The bundled application, benchmark runner, window input, and combined CLI errors are
binary-local modules built against the same public library surface.

## User-Defined Shaders

User-defined Rust shaders are supported through the same public model as built-in shaders.

The `render` façade exports:

- `Shader<C>`, `Interpolatable`, `Vertex`, `FragmentInput`, and `FragmentOutput`;
- `SUPPORTED_TEXCOORD_SETS` for the two supported UV-density channels;
- `GraphicsPipeline<S>`, `GraphicsPipelineState`, and `VertexProgramId`;
- primitive, depth, color-target, culling, winding, polygon, comparison, and alpha-blend state.

A shader supplies one varying type and one copyable typed draw context `C`. Both stages receive the
same context. `Shader<C>` and its varying require the thread-safety bounds needed by Rayon, while
the public traits do not mention materials, built-in shaders, prepared triangles, framebuffer
bands, or backend caches.

Dispatch remains static. Callers do not implement `dyn Shader` and do not select a closed
Shadow/PBR command enum. The public API continues to use `nalgebra` vectors, points, and matrices;
those dependency types appear directly in public signatures.

A complete, compiled custom-shader program lives in
[`examples/custom_shader.rs`](../examples/custom_shader.rs) and runs with:

```bash
cargo run --release --example custom_shader
```

## Public `render` Surface

### State and pipelines

`GraphicsPipeline<S>` owns an immutable shader algorithm, complete `GraphicsPipelineState`, and a
process-unique `VertexProgramId`. Raster-only state variants may share one token and transformed
vertices; variants that can change vertex output must use different tokens.

Each render pass assigns an internal identity whenever draw bindings are set. The vertex cache key
combines the vertex-program token, this binding identity, and the vertex-source domain instead of
hashing floating-point matrices. `VertexProgramId::new()` creates an unforgeable process-unique
token that may be copied across pipeline variants only when their vertex output is identical.
Caches remain submission-local.

### Typed recording and synchronous submission

`RenderDevice` creates command encoders and a `GraphicsQueue`. The queue owns the private software
backend. A command buffer contains one concrete shader/context family and one render pass; a pass
may contain multiple ordered phases.

The implemented recording shape is:

```rust
let device = RenderDevice::new();
let mut queue = device.create_queue();
let mut encoder = device.create_command_encoder("triangle");

{
    let mut pass = encoder.begin_render_pass(
        RenderPassDescriptor {
            label: Some("triangle"),
            target: &mut target,
            color_ops: Some(Operations {
                load: LoadOp::Clear(Vector3::zeros()),
            }),
            depth_ops: Some(Operations {
                load: LoadOp::Clear(f32::INFINITY),
            }),
        },
        None,
    )?;
    pass.set_pipeline(&pipeline);
    pass.set_draw_bindings(context);
    pass.draw_mesh(&mesh, 0.0)?;
    pass.end()?;
}

let command_buffer = encoder.finish()?;
let report = queue.submit(command_buffer);
```

`begin_render_pass` validates the descriptor before recording. Drawing without a pipeline or typed
bindings, finishing without a pass, recording a second pass, or abandoning an active pass produces
a structured `CommandError`. `finish` consumes the encoder. `GraphicsQueue::submit` consumes the
command buffer and completes attachment processing, backend preparation, and rasterization before
returning a `SubmissionReport`; there is no fence, future, or asynchronous completion state.

`finish_phase(label)` seals the current ordered draw phase and begins another within the same pass.
`sort_transparent()` sorts only the current phase before it is sealed. Shadow and main work use
separate typed command buffers and synchronous submissions because they have different shader types
and the main pass depends on completed shadow output.

### Attachments, background, and output

`RenderPassDescriptor` borrows one `RenderTarget` mutably and declares optional color/depth
`Operations`. `LoadOp::Load` preserves an attachment and `LoadOp::Clear` initializes it. A
`BackgroundPass` represents generated gradient or image color separately from attachment load
semantics. The software backend may fuse background fill with depth clear into one disjoint-band
traversal.

A depth-only pass uses `GraphicsPipelineState { color_target: None, .. }` to suppress color writes.
It does not skip fragment shading: masked shadow materials still run the fragment shader and may
discard from sampled alpha.

`RenderTarget` and `MainHdrTarget` own private interleaved HDR color/depth storage.
`RenderTargetReadback` exposes resolved color, individual sample color/depth, dimensions, and an
explicit copied depth vector without exposing mutable framebuffer samples. `PresentBuffer` receives
packed output from the fused resolve/tonemap pass.

Target construction and dimension-only validation return `RenderTargetError`; present-buffer
construction returns `PresentBufferError`. Configuration loading and standalone validation expose
`ConfigError` and `ConfigValidationError` from `io::config`, so callers can inspect failure kinds
without parsing display strings.

`FrameResources` retains reusable background and shadow-snapshot storage for built-in frame passes.
`ShadowPassOutput` represents either a fully consistent shadow map or disabled output; its fields
cannot be independently mutated. Validated `PbrShadowBindings` reject mismatched dimensions,
non-finite transforms, and invalid bias or PCF settings before fragment execution.

### Built-in passes and shaders

`render::builtin::pbr` exports `PbrShader` and its typed frame, object, material, draw, and shadow
bindings. `render::builtin::shadow` exports the equivalent depth-only shader and bindings. Both use
ordinary `GraphicsPipeline` values and typed commands.

`render_shadow_pass` and `render_main_pass` are convenience builders for the repository's scene
and configuration types. Their profiled variants report pass setup, recording, attachment
processing, backend preparation, rasterization, and inclusive submission time.
`execute_resolve_tonemap_pass` performs SSAA resolve, exposure, optional ACES, linear-to-sRGB
conversion, and packed presentation output in one CPU pass.

## `scene` and `io` Surfaces

`scene` remains the authoring surface for cameras, lights, materials, meshes, models, scene objects,
transforms, texture images, samplers, texture bindings, UV selection, and `RenderScene`. Backend
types do not appear in scene fields or constructors.

`io` remains the workflow surface for validated TOML configuration, relative-path resolution,
glTF loading, and PNG output. Configuration and mesh containers retain public authoring fields where
direct editing is intentional. Renderer caches, command recording, minifb interaction, and CLI
dispatch are not part of `io`.

## Ownership and Internal Execution

Command/resource owners such as `GraphicsPipeline`, `CommandEncoder`, `CommandBuffer`,
`GraphicsQueue`, `RenderTarget`, `PresentBuffer`, `FrameResources`, and
`ShadowPassOutput` keep execution-sensitive fields private. Public descriptors and scene/config
containers expose fields specifically intended for authoring.

The private backend prepares primitives and bins them into exclusive horizontal framebuffer bands.
Production framebuffer mutation contains no `unsafe`, `UnsafeCell`, atomics, locks, or manual
`Sync`. Rayon workers mutate disjoint bands; they never determine primitive blend order.

## Transparent Ordering Contract

Transparent alpha blending has one observable total order:

- the high-level pass builder computes a view-space depth key;
- visible geometry uses negative view-space Z, so ascending Z is back-to-front;
- equal depth keys retain submission order through monotonically increasing insertion IDs;
- mesh expansion retains triangle/index order;
- clipping retains generated triangle-fan order;
- parallel preparation and collection retain encoded primitive order;
- each framebuffer band visits its binned primitives sequentially in that order.

The result is identical across worker counts. Pipeline/material sorting must not violate the
depth-plus-insertion order. This contract applies equally to built-in and user-defined pipelines
using order-dependent blending.
