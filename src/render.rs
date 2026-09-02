//! Stable rendering API for programmable shaders, command recording, and synchronous submission.

pub use crate::core::geometry::{SUPPORTED_TEXCOORD_SETS, Vertex};
pub use crate::core::pipeline_state::{
    BlendState, ColorTargetState, CompareFunction, CullMode, DepthStencilState, FrontFace,
    GraphicsPipeline, GraphicsPipelineState, PolygonMode, PrimitiveState, PrimitiveTopology,
    VertexProgramId,
};
pub use crate::core::shader::{FragmentInput, FragmentOutput, Interpolatable, Shader};
pub use crate::error::ResolveTonemapError;
pub use crate::pipeline::passes::{
    MainPassTimings, ResolveTonemapPassDescriptor, ShadowPassOutput, ShadowPassTimings,
    TonemapOperator, execute_resolve_tonemap_pass, render_main_pass, render_main_pass_profiled,
    render_shadow_pass, render_shadow_pass_profiled,
};
pub use crate::pipeline::renderer::{
    BackgroundPass, BackgroundSource, CommandBuffer, CommandEncoder, CommandError, FrameResources,
    GraphicsQueue, LoadOp, MainHdrTarget, ObjectBindingId, Operations, PhaseSubmissionReport,
    PresentBuffer, RenderDevice, RenderError, RenderGeometry, RenderPassDescriptor,
    RenderPassEncoder, RenderTarget, RenderTargetReadback, SubmissionReport,
};

/// Built-in shader implementations and their typed binding groups.
pub mod builtin {
    /// Physically based material shader and bindings.
    pub mod pbr {
        pub use crate::pipeline::shaders::pbr::{
            PbrDrawContext, PbrFrameBindings, PbrMaterialBindings, PbrObjectBindings, PbrShader,
            PbrVarying,
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
