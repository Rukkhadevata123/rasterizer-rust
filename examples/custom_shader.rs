use nalgebra::{Point3, Vector2, Vector3, Vector4};
use rasterizer_rust::render::{
    CullMode, FragmentInput, FragmentOutput, GraphicsPipeline, GraphicsPipelineState,
    Interpolatable, LoadOp, ObjectBindingId, Operations, PrimitiveState, RenderDevice,
    RenderPassDescriptor, RenderTarget, Shader, Vertex, VertexProgramId,
};
use rasterizer_rust::scene::mesh::Mesh;
use std::error::Error;
use std::ops::{Add, Mul};

#[derive(Clone, Copy)]
struct ColorVarying(Vector4<f32>);

impl Add for ColorVarying {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul<f32> for ColorVarying {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Interpolatable for ColorVarying {}

#[derive(Clone, Copy)]
struct VertexColorShader;

impl Shader<()> for VertexColorShader {
    type Varying = ColorVarying;

    fn vertex(&self, vertex: &Vertex, (): ()) -> (Vector4<f32>, Self::Varying) {
        (
            vertex.position.to_homogeneous(),
            ColorVarying(vertex.tangent),
        )
    }

    fn fragment(&self, input: FragmentInput<Self::Varying>, (): ()) -> FragmentOutput {
        FragmentOutput::Color(input.varying.0)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let device = RenderDevice::new();
    let mut queue = device.create_queue();
    let mut target = RenderTarget::new(256, 256, 1).map_err(std::io::Error::other)?;
    let pipeline = GraphicsPipeline::new(
        VertexColorShader,
        GraphicsPipelineState {
            primitive: PrimitiveState {
                cull_mode: CullMode::None,
                ..Default::default()
            },
            ..Default::default()
        },
        VertexProgramId::from_pass_index(0),
    );
    let mut vertices = vec![
        Vertex::new(Point3::new(-0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.8, -0.8, 0.0), Vector3::z(), Vector2::zeros()),
        Vertex::new(Point3::new(0.0, 0.8, 0.0), Vector3::z(), Vector2::zeros()),
    ];
    vertices[0].tangent = Vector4::new(1.0, 0.0, 0.0, 1.0);
    vertices[1].tangent = Vector4::new(0.0, 1.0, 0.0, 1.0);
    vertices[2].tangent = Vector4::new(0.0, 0.0, 1.0, 1.0);
    let mesh = Mesh::new(vertices, vec![0, 1, 2], 0);

    let mut encoder = device.create_command_encoder("custom-shader");
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
        pass.set_draw_bindings((), ObjectBindingId::from_pass_index(0));
        pass.draw_mesh(&mesh, 0.0)?;
        pass.end()?;
    }

    let report = queue.submit(encoder.finish()?)?;
    let center = target
        .readback()
        .color(128, 128)
        .expect("the center coordinate is inside the target");
    println!(
        "center linear RGB: {center:?}; synchronous submission: {:?}",
        report.submission_total
    );
    Ok(())
}
