use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum CullMode {
    Back,
    Front,
    None,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, Default)]
pub enum PrimitiveTopology {
    #[default]
    TriangleList,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, Default)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, Default)]
pub enum PolygonMode {
    #[default]
    Fill,
    Line,
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum CompareFunction {
    Never,
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
    Always,
}

impl CompareFunction {
    pub(crate) fn test(self, incoming: f32, stored: f32) -> bool {
        match self {
            Self::Never => false,
            Self::Less => incoming < stored,
            Self::LessEqual => incoming <= stored,
            Self::Equal => incoming == stored,
            Self::NotEqual => incoming != stored,
            Self::GreaterEqual => incoming >= stored,
            Self::Greater => incoming > stored,
            Self::Always => true,
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum BlendState {
    Alpha,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub polygon_mode: PolygonMode,
}

impl Default for PrimitiveState {
    fn default() -> Self {
        Self {
            topology: PrimitiveTopology::TriangleList,
            front_face: FrontFace::CounterClockwise,
            cull_mode: CullMode::Back,
            polygon_mode: PolygonMode::Fill,
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub struct DepthStencilState {
    pub depth_compare: CompareFunction,
    pub depth_write_enabled: bool,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_compare: CompareFunction::Less,
            depth_write_enabled: true,
        }
    }
}

#[derive(PartialEq, Eq, Copy, Clone, Debug, Default)]
pub struct ColorTargetState {
    pub blend: Option<BlendState>,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub struct GraphicsPipelineState {
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub color_target: Option<ColorTargetState>,
}

impl Default for GraphicsPipelineState {
    fn default() -> Self {
        Self {
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState::default()),
            color_target: Some(ColorTargetState::default()),
        }
    }
}

/// Unique identity shared by pipeline variants with identical vertex output.
#[derive(PartialEq, Eq, Copy, Clone, Debug, Hash)]
pub struct VertexProgramId(usize);

static NEXT_VERTEX_PROGRAM_ID: AtomicUsize = AtomicUsize::new(0);

impl VertexProgramId {
    /// Allocates a new identity for one vertex-output contract.
    pub fn new() -> Self {
        let id = NEXT_VERTEX_PROGRAM_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("vertex program ID space exhausted");
        Self(id)
    }
}

impl Default for VertexProgramId {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable shader algorithm and fixed-function state used by recorded draws.
pub struct GraphicsPipeline<S> {
    shader: S,
    state: GraphicsPipelineState,
    vertex_program_id: VertexProgramId,
}

impl<S> GraphicsPipeline<S> {
    pub fn new(
        shader: S,
        state: GraphicsPipelineState,
        vertex_program_id: VertexProgramId,
    ) -> Self {
        Self {
            shader,
            state,
            vertex_program_id,
        }
    }

    pub fn shader(&self) -> &S {
        &self.shader
    }

    pub fn state(&self) -> GraphicsPipelineState {
        self.state
    }

    pub fn vertex_program_id(&self) -> VertexProgramId {
        self.vertex_program_id
    }
}
#[cfg(test)]
mod pipeline_tests {
    use super::*;

    #[test]
    fn graphics_pipeline_keeps_immutable_state_and_vertex_program_identity() {
        let state = GraphicsPipelineState {
            primitive: PrimitiveState {
                polygon_mode: PolygonMode::Line,
                ..Default::default()
            },
            ..Default::default()
        };
        let vertex_program_id = VertexProgramId::new();
        let pipeline = GraphicsPipeline::new("shader", state, vertex_program_id);

        assert_eq!(pipeline.shader(), &"shader");
        assert_eq!(pipeline.state(), state);
        assert_eq!(pipeline.vertex_program_id(), vertex_program_id);
        assert_ne!(pipeline.vertex_program_id(), VertexProgramId::new());
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pipeline_state_preserves_the_current_render_contract() {
        let state = GraphicsPipelineState::default();

        assert_eq!(
            state.primitive,
            PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::CounterClockwise,
                cull_mode: CullMode::Back,
                polygon_mode: PolygonMode::Fill,
            }
        );
        assert_eq!(state.depth_stencil, Some(DepthStencilState::default()));
        assert_eq!(state.color_target, Some(ColorTargetState::default()));
    }

    #[test]
    fn alpha_pipeline_state_preserves_independent_primitive_and_depth_state() {
        let state = GraphicsPipelineState {
            primitive: PrimitiveState {
                front_face: FrontFace::Clockwise,
                cull_mode: CullMode::Front,
                polygon_mode: PolygonMode::Line,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                depth_compare: CompareFunction::Always,
                depth_write_enabled: true,
            }),
            color_target: Some(ColorTargetState {
                blend: Some(BlendState::Alpha),
            }),
        };

        assert_eq!(state.primitive.topology, PrimitiveTopology::TriangleList);
        assert_eq!(state.primitive.front_face, FrontFace::Clockwise);
        assert_eq!(state.primitive.cull_mode, CullMode::Front);
        assert_eq!(state.primitive.polygon_mode, PolygonMode::Line);
        assert_eq!(
            state.depth_stencil,
            Some(DepthStencilState {
                depth_compare: CompareFunction::Always,
                depth_write_enabled: true,
            })
        );
        assert_eq!(
            state.color_target,
            Some(ColorTargetState {
                blend: Some(BlendState::Alpha),
            })
        );
    }

    #[test]
    fn compare_functions_match_their_ordering_contract() {
        let stored = 0.5;
        let incoming = [0.25, 0.5, 0.75];
        let cases = [
            (CompareFunction::Never, [false, false, false]),
            (CompareFunction::Less, [true, false, false]),
            (CompareFunction::LessEqual, [true, true, false]),
            (CompareFunction::Equal, [false, true, false]),
            (CompareFunction::NotEqual, [true, false, true]),
            (CompareFunction::GreaterEqual, [false, true, true]),
            (CompareFunction::Greater, [false, false, true]),
            (CompareFunction::Always, [true, true, true]),
        ];

        for (compare, expected) in cases {
            assert_eq!(
                incoming.map(|value| compare.test(value, stored)),
                expected,
                "unexpected {compare:?} comparison results"
            );
        }
    }
}
