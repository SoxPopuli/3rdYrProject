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
    pub pos: Vector2<f32>,
    pub color: Vector3<f32>,
}

impl Vertex
{
    pub fn new(pos: [f32; 2], color: [f32; 3]) -> Self 
    {
        let pv = Vector2::from_row_slice(&pos);
        let cv = Vector3::from_row_slice(&color);

        Self{
            pos: pv,
            color: cv,
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


    pub fn attribute_descriptions() -> [VertexInputAttributeDescription; 2]
    {
        let mut attrs: [VertexInputAttributeDescription; 2] = Default::default();

        attrs[0] = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32)
            .build();

        attrs[1] = VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32)
            .build();

        attrs
    }

}
