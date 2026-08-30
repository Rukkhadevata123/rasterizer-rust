use super::rasterizer::{BlendMode, CullMode, DepthCompare, RenderState};

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

impl From<DepthCompare> for CompareFunction {
    fn from(value: DepthCompare) -> Self {
        match value {
            DepthCompare::Never => Self::Never,
            DepthCompare::Less => Self::Less,
            DepthCompare::LessEqual => Self::LessEqual,
            DepthCompare::Equal => Self::Equal,
            DepthCompare::NotEqual => Self::NotEqual,
            DepthCompare::GreaterEqual => Self::GreaterEqual,
            DepthCompare::Greater => Self::Greater,
            DepthCompare::Always => Self::Always,
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

impl From<RenderState> for GraphicsPipelineState {
    fn from(value: RenderState) -> Self {
        Self {
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                front_face: if value.front_face_inverted {
                    FrontFace::Clockwise
                } else {
                    FrontFace::CounterClockwise
                },
                cull_mode: value.cull_mode,
                polygon_mode: if value.wireframe {
                    PolygonMode::Line
                } else {
                    PolygonMode::Fill
                },
            },
            depth_stencil: Some(DepthStencilState {
                depth_compare: if value.depth_test {
                    value.depth_compare.into()
                } else {
                    CompareFunction::Always
                },
                depth_write_enabled: value.depth_write,
            }),
            color_target: Some(ColorTargetState {
                blend: match value.blend_mode {
                    BlendMode::Opaque => None,
                    BlendMode::Alpha => Some(BlendState::Alpha),
                },
            }),
        }
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
    fn legacy_state_maps_to_the_new_pipeline_vocabulary() {
        let state = GraphicsPipelineState::from(RenderState {
            cull_mode: CullMode::Front,
            front_face_inverted: true,
            depth_test: false,
            depth_compare: DepthCompare::Greater,
            depth_write: true,
            blend_mode: BlendMode::Alpha,
            wireframe: true,
        });

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
    fn every_legacy_depth_compare_has_an_exact_pipeline_equivalent() {
        let cases = [
            (DepthCompare::Never, CompareFunction::Never),
            (DepthCompare::Less, CompareFunction::Less),
            (DepthCompare::LessEqual, CompareFunction::LessEqual),
            (DepthCompare::Equal, CompareFunction::Equal),
            (DepthCompare::NotEqual, CompareFunction::NotEqual),
            (DepthCompare::GreaterEqual, CompareFunction::GreaterEqual),
            (DepthCompare::Greater, CompareFunction::Greater),
            (DepthCompare::Always, CompareFunction::Always),
        ];

        for (legacy, pipeline) in cases {
            assert_eq!(CompareFunction::from(legacy), pipeline);
        }
    }
}
