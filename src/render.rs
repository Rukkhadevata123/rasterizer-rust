//! Canonical rendering API for programmable shaders, typed commands, and synchronous submission.
//!
//! A [`RenderDevice`] creates immutable [`GraphicsPipeline`] values, typed [`CommandEncoder`]s,
//! and a backend-owning [`GraphicsQueue`]. Each command buffer records one concrete shader/context
//! family and one render pass, which may contain multiple ordered phases. [`GraphicsQueue::submit`]
//! completes attachment processing, vertex preparation, and rasterization before it returns.
//!
//! Implement [`Shader`] and [`Interpolatable`] to use custom statically dispatched shaders. The
//! [`builtin`] module exposes PBR and shadow shaders through the same pipeline and command model.
//! Completed targets are inspected through [`RenderTargetReadback`]; framebuffer storage and the
//! software backend remain private.

pub use crate::core::framebuffer::RenderTargetError;
pub use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
pub use crate::core::pipeline_state::{
    BlendState, ColorTargetState, CompareFunction, CullMode, DepthStencilState, FrontFace,
    GraphicsPipeline, GraphicsPipelineState, PolygonMode, PrimitiveState, PrimitiveTopology,
    VertexProgramId,
};
pub use crate::core::shader::{FragmentInput, FragmentOutput, Interpolatable, Shader};
pub use crate::pipeline::passes::{
    MainPassTimings, ResolveTonemapError, ResolveTonemapPassDescriptor, ShadowPassOutput,
    ShadowPassTimings, TonemapOperator, execute_resolve_tonemap_pass, render_main_pass,
    render_main_pass_profiled, render_shadow_pass, render_shadow_pass_profiled,
};
pub use crate::pipeline::renderer::{
    BackgroundPass, BackgroundSource, CommandBuffer, CommandEncoder, CommandError, FrameResources,
    GraphicsQueue, LoadOp, MainHdrTarget, Operations, PhaseSubmissionReport, PresentBuffer,
    PresentBufferError, RenderDevice, RenderError, RenderPassDescriptor, RenderPassEncoder,
    RenderTarget, RenderTargetReadback, SubmissionReport,
};

/// Built-in shader implementations and their typed binding groups.
pub mod builtin {
    /// Physically based material shader and bindings.
    pub mod pbr {
        pub use crate::pipeline::shaders::pbr::{
            PbrDrawContext, PbrFrameBindings, PbrMaterialBindings, PbrObjectBindings, PbrShader,
            PbrShadowBindings, PbrShadowBindingsDescriptor, PbrShadowBindingsError, PbrVarying,
        };
    }

    /// Depth-only shadow shader and bindings.
    pub mod shadow {
        pub use crate::pipeline::shaders::shadow::{
            ShadowDrawContext, ShadowFrameBindings, ShadowMaterialBindings, ShadowObjectBindings,
            ShadowShader, ShadowVarying,
        };
    }
}
