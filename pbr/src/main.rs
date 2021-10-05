use vulkano::{
    Version,
    device::{
        Device,
        Features,
        DeviceExtensions,
        physical::{ 
            PhysicalDevice,
            PhysicalDeviceType
        }
    },
    buffer::{ CpuAccessibleBuffer, BufferUsage },
    pipeline::shader::ShaderModule,
    instance::Instance,
    swapchain::Swapchain,
    image::ImageUsage,
};
use vulkano_win::VkSurfaceBuild;
use winit::{event_loop::EventLoop, window::WindowBuilder};

fn main() 
{
    let required_extensions = vulkano_win::required_extensions();
    
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None).unwrap();
    
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_exts = DeviceExtensions{
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| { //get first device that supports extensions
            p.supported_extensions().is_superset_of(&device_exts)
        })
        .filter_map(|p| { //get the queue family that supports graphics and surface queues
            p.queue_families().find(|&q| {
                q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
            })
            .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| { //choose the device based on type
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            }
        }).unwrap();
        

    println!("Device: {} ({:?})", 
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );
    
    let ( device, mut queues  ) = Device::new(
        physical_device,
        &Features::none(), 
        &physical_device.required_extensions().union(&device_exts),
        [(queue_family, 0.5)]
    ).unwrap();
    
    let queue = queues.next().unwrap();
    
    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        
        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };
    
    #[derive(Debug, Default, Clone)]
    struct Vertex {
        position: [f32; 2]
    }
    
    vulkano::impl_vertex!(Vertex, position);
    
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex{ position: [-0.5, -0.25] },
            Vertex{ position: [0.0, 0.5] },
            Vertex{ position: [0.25, -0.1] },
        ].iter()
    ).unwrap();
    
    const VERTEX_SHADER: &[u32] = vk_shader_macros::include_glsl!("shaders/triangle.vert");
    const FRAGMENT_SHADER: &[u32] = vk_shader_macros::include_glsl!("shaders/triangle.frag");
    
    let vs = unsafe{ ShaderModule::new(device.clone(), std::mem::transmute(VERTEX_SHADER)) }.unwrap();
    let fs = unsafe{ ShaderModule::new(device.clone(), std::mem::transmute(FRAGMENT_SHADER)) }.unwrap();
    
    
}
