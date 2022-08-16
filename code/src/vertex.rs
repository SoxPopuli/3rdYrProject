use memoffset::offset_of;
use nalgebra::{
    Vector2,
    Vector3
};

use ash::vk::{
    VertexInputBindingDescription,
    VertexInputRate,
    VertexInputAttributeDescription,
    Format,
};

pub struct Vertex 
{
    pub pos: Vector3<f32>,
    pub color: Vector3<f32>,
    pub tex_coord: Vector2<f32>,
}

impl Vertex
{
    pub fn new(pos: [f32; 3], color: [f32; 3], tex: [f32; 2]) -> Self 
    {
        let pv = Vector3::from_row_slice(&pos);
        let cv = Vector3::from_row_slice(&color);
        let tex = Vector2::from_row_slice(&tex);

        Self{
            pos: pv,
            color: cv,
            tex_coord: tex,
        }
    }

    pub fn binding_description() -> VertexInputBindingDescription 
    {
        VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Vertex>() as u32)
            .input_rate(VertexInputRate::VERTEX)
            .build()
    }


    pub fn attribute_descriptions() -> [VertexInputAttributeDescription; 3]
    {
        let mut attrs: [VertexInputAttributeDescription; 3] = Default::default();

        attrs[0] = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32)
            .build();

        attrs[1] = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32)
            .build();

        attrs[2] = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, tex_coord) as u32)
            .build();

        attrs
    }

}
