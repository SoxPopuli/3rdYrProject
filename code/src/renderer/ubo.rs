use nalgebra::Matrix4;

type Mat4f = Matrix4<f32>;

#[derive(Debug, Default, Clone, Copy)]
pub struct UniformBufferObject 
{
    pub model: Mat4f,
    pub view: Mat4f,
    pub proj: Mat4f,
}
