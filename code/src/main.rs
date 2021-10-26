use ash::{
    Instance,
    Entry,
    Device,
    extensions::khr,
    extensions::ext,
    vk::{
        self,
        DeviceCreateInfo,
        DeviceQueueCreateInfo,
        Handle,
        PhysicalDeviceType,
        QueueFlags,
        SurfaceKHR,
        SurfaceFormatKHR,
        PresentModeKHR,
        SurfaceCapabilitiesKHR,
        Extent2D,
        Offset2D,
        Rect2D,
        ShaderModule,
    }
};

use std::{
    collections::{
        HashSet,
        hash_map::RandomState
    }, 
    ffi::{ CStr, c_void }, 
    iter::FromIterator, 
    marker::PhantomPinned, 
    pin::Pin, 
    str::Utf8Error,
    cell::Cell,
};

#[derive(Debug, Default)]
struct Extensions 
{
    slices: Vec<[i8; 256]>,
    ptrs: Vec<*const i8>,
    _pin: PhantomPinned
}
impl Extensions
{
    fn get(entry: &Entry, wanted: HashSet<String>) -> Pin<Box<Self>> 
    {
        let exts = entry.enumerate_instance_extension_properties().unwrap();

        let mut slices = Vec::with_capacity(exts.len());

        for e in exts {
            let s = unsafe{ CStr::from_ptr(e.extension_name.as_ptr()) }
                .to_str()
                .unwrap();

            if wanted.contains(s) {
                slices.push(e.extension_name.clone());
            }
        }

        let ptrs = slices.iter().map(|e| { e.as_ptr() }).collect();

        Box::pin(Extensions {
            slices, 
            ptrs,
            _pin: PhantomPinned
        })
    }

    fn len(&self) -> u32 { self.slices.len() as u32 }
}

#[derive(Debug, Default)]
struct Layers 
{
    slices: Vec<[i8; 256]>,
    ptrs: Vec<*const i8>,
    _pin: PhantomPinned
}
impl Layers 
{
    fn get(entry: &Entry, wanted: HashSet<String>) -> Pin<Box<Self>> 
    {
        let layers = entry.enumerate_instance_layer_properties().unwrap();

        let mut slices = Vec::with_capacity( layers.len() );

        for l in layers {
            let s = unsafe{ CStr::from_ptr( l.layer_name.as_ptr() ) }
                .to_str()
                .unwrap();

            if wanted.contains(s) {
                slices.push( l.layer_name.clone() );
            }
        }

        let ptrs = slices.iter().map(|l| { l.as_ptr() }).collect();
        
        Box::pin(Layers {
            slices,
            ptrs,
            _pin: PhantomPinned
        })
    }

    fn len(&self) -> u32 { self.slices.len() as u32 }
}

struct Surface 
{
    obj: khr::Surface,
    handle: vk::SurfaceKHR,
}

struct QueueIndices 
{
    graphics: u32,
    present: u32,
}

struct DeviceQueues 
{
    graphics: vk::Queue,
    present: vk::Queue,
}

struct Application 
{
    sdl_context: sdl2::Sdl,
    window: sdl2::video::Window,

    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    queue_indices: QueueIndices,

    surface_loader: khr::Surface,
    surface: vk::SurfaceKHR,

    device_queues: DeviceQueues,

    surface_capabilites: vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    swap_extent: Extent2D,

    swapchain_loader: khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,

    render_pass: vk::RenderPass,
    graphics_pipeline: vk::Pipeline,
    graphics_pipeline_layout: vk::PipelineLayout,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,

    current_frame: usize,

    #[cfg(debug_assertions)]
    debug_loader: ext::DebugUtils,
    #[cfg(debug_assertions)]
    debug: vk::DebugUtilsMessengerEXT,
} 
impl Application
{
    const fn max_frames_in_flight() -> usize { 2 }
}
impl Drop for Application 
{
    fn drop(&mut self) 
    {
        unsafe {
            for i in 0..Application::max_frames_in_flight() {
                self.device.destroy_semaphore(self.image_available_semaphores[i], None);
                self.device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
                //self.device.destroy_fence(self.images_in_flight[i], None);
            }

            self.device.free_command_buffers(self.command_pool, self.command_buffers.as_slice());
            self.device.destroy_command_pool(self.command_pool, None);

            for fb in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }

            self.device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_render_pass(self.render_pass, None);
            
            for view in &self.swapchain_image_views {
                self.device.destroy_image_view(*view, None);
            }

            //images destroyed by destroy_swapchain
            //for img in &self.swapchain_images {
            //    self.device.destroy_image(*img, None);
            //}

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);

            self.device.destroy_shader_module(self.vertex_module, None);
            self.device.destroy_shader_module(self.fragment_module, None);

            cfg_if::cfg_if! {
                if #[cfg(debug_assertions)] {
                    self.debug_loader.destroy_debug_utils_messenger(self.debug, None);
                }
            }

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn string_from_slice(slice: &[i8]) -> Result<&str, Utf8Error> 
{
    unsafe{ CStr::from_ptr(slice.as_ptr()) }
        .to_str()
}

const PROGRAM_NAME: &'static str = env!("CARGO_PKG_NAME");

fn choose_physical_device(instance: &Instance) -> vk::PhysicalDevice
{
    let physical_devices = unsafe{ instance.enumerate_physical_devices() }.unwrap();

    let physical_device = *physical_devices.iter().max_by_key(|pd| {
        let mut score = 0u64;
        let props = unsafe{ instance.get_physical_device_properties(**pd) };

        //prioritize discrete gpus
        score += match props.device_type {
            PhysicalDeviceType::DISCRETE_GPU => 1000,
            PhysicalDeviceType::INTEGRATED_GPU => 100,
            PhysicalDeviceType::VIRTUAL_GPU => 10,
            PhysicalDeviceType::CPU => 5,
            _ => 1,
        };

        //prioritize highest vram
        score += props.limits.max_memory_allocation_count as u64;

        score
    }).unwrap();

    physical_device
}

fn get_queue_indices(instance: &Instance, physical_device: vk::PhysicalDevice, surface_loader: &khr::Surface, surface: vk::SurfaceKHR) -> (u32, u32) 
{
    let queue_props = unsafe{ instance.get_physical_device_queue_family_properties(physical_device) };
    let mut graphics_index = None;
    let mut present_index = None;
    let mut index = 0u32;
    for q in queue_props {
        let has_surface_support = || {
            unsafe{ surface_loader.get_physical_device_surface_support(physical_device, index, surface) }.unwrap()
        };

        if q.queue_flags.contains( QueueFlags::GRAPHICS ) { 
            graphics_index = Some(index);

            //explicitly prefer presenting on graphics queue
            if has_surface_support() {
                present_index = Some(index);
                break;
            }
        }
    
        if has_surface_support() {
            present_index = Some(index);
        }

        index += 1;
    }
    (graphics_index.expect("no graphics index found"),
     present_index.expect("no present index found"))
}

fn create_debug_messenger(entry: &Entry, instance: &Instance) -> (ext::DebugUtils, vk::DebugUtilsMessengerEXT)
{
    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
    use vk::DebugUtilsMessageTypeFlagsEXT as MessageType;

    #[no_mangle]
    unsafe extern "system" fn debug_callback(severity: vk::DebugUtilsMessageSeverityFlagsEXT, msg_type: vk::DebugUtilsMessageTypeFlagsEXT, cb: *const vk::DebugUtilsMessengerCallbackDataEXT, data: *mut c_void) -> u32
    {
        let msg = CStr::from_ptr((*cb).p_message).to_str().unwrap();
        println!("[VALIDATION] {}", msg);

        vk::FALSE
    }

    let message_severity =
        //Severity::VERBOSE |
        Severity::WARNING |
        Severity::ERROR;

    let message_type = 
        MessageType::GENERAL |
        MessageType::VALIDATION |
        MessageType::PERFORMANCE;

    let info = vk::DebugUtilsMessengerCreateInfoEXT {
        message_severity,
        message_type,
        pfn_user_callback: Some(debug_callback),

        ..Default::default()
    };

    let debug_utils_loader = ext::DebugUtils::new(entry, instance);
    let debug = unsafe{ debug_utils_loader.create_debug_utils_messenger(&info, None) }.unwrap();

    (debug_utils_loader, debug)
}

fn main() 
{
    let mut app = unsafe{ init() };

    let mut event_pump = app.sdl_context.event_pump().unwrap();

    'running: loop {
        for ev in event_pump.poll_iter() {
            if !handle_event(&ev, &app) { break 'running; }
        }

        draw_frame(&mut app);
        std::thread::sleep( std::time::Duration::from_secs_f32(1.0 / 60.0) );
    }

    unsafe{ app.device.device_wait_idle() }.unwrap();
}

fn handle_event(ev: &sdl2::event::Event, app: &Application) -> bool
{
    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;

    match ev {
        Event::KeyDown { keycode, .. } => { 
            match keycode {
                Some(Keycode::Escape) => { return false },
                _ => {  }
            };
        },
        _ => {  }
    };

    true
}

fn draw_frame(app: &mut Application)
{
    unsafe{ app.device.wait_for_fences(&[app.in_flight_fences[app.current_frame]], true, u64::MAX) }.unwrap();

    let (image_index, _) = unsafe{ app.swapchain_loader.acquire_next_image(
        app.swapchain,
        u64::MAX,
        app.image_available_semaphores[app.current_frame],
        vk::Fence::null()
    ) }.unwrap();

    if app.images_in_flight[image_index as usize] != vk::Fence::null() {
        unsafe{ app.device.wait_for_fences(&[app.images_in_flight[image_index as usize]], true, u64::MAX) }.unwrap();
    }
    app.images_in_flight[image_index as usize] = app.in_flight_fences[ app.current_frame ];

    let wait_semaphores = [app.image_available_semaphores[app.current_frame]];
    let signal_semaphores = [app.render_finished_semaphores[app.current_frame]];

    let submit_info = vk::SubmitInfo::builder()
        .wait_semaphores(&wait_semaphores)
        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
        .command_buffers(&[app.command_buffers[image_index as usize]])
        .signal_semaphores(&signal_semaphores)
        .build();

    unsafe{ app.device.reset_fences(&[app.in_flight_fences[app.current_frame]]) }.unwrap();
    unsafe{ app.device.queue_submit(app.device_queues.graphics, &[submit_info], app.in_flight_fences[app.current_frame]) }.unwrap();

    let swapchains = [app.swapchain];
    let present_info = vk::PresentInfoKHR::builder()
        .wait_semaphores(&signal_semaphores)
        .swapchains(&swapchains)
        .image_indices(&[image_index])
        .build();
        
    unsafe{ app.swapchain_loader.queue_present(app.device_queues.present, &present_info) }.unwrap();
    //unsafe{ app.device.queue_wait_idle(app.device_queues.present) }.unwrap();

    app.current_frame = (app.current_frame + 1) % Application::max_frames_in_flight()
}

unsafe fn init() -> Application
{
    let entry = Entry::new().unwrap();
    let app_info = vk::ApplicationInfo {
        api_version: vk::make_api_version(0, 1, 0, 0),
        ..Default::default()
    };

    //window creation
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window(PROGRAM_NAME, 800, 800)
        .position_centered()
        .vulkan()
        .build()
        .unwrap();

    let mut required_exts: Vec<_> = window.vulkan_instance_extensions().unwrap()
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut required_layers = Vec::<String>::new();

    if cfg!(debug_assertions) {
        required_exts.push("VK_EXT_debug_utils".into());
        required_layers.push("VK_LAYER_KHRONOS_validation".into());
    }

    let exts = Extensions::get(&entry, HashSet::from_iter(required_exts));
    let layers = Layers::get(&entry, HashSet::from_iter(required_layers));

    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,

        pp_enabled_extension_names: exts.ptrs.as_ptr(),
        enabled_extension_count: exts.len(),

        pp_enabled_layer_names: layers.ptrs.as_ptr(),
        enabled_layer_count: layers.len(),

        ..Default::default()
    };

    println!("Enabled Extensions:");
    for e in &exts.slices {
        println!("\t{}", string_from_slice(e).unwrap());
    }

    println!("Enabled Layers:");
    for l in &layers.slices {
        println!("\t{}", string_from_slice(l).unwrap());
    }

    let instance = entry.create_instance(&create_info, None).unwrap();
    let surface_loader = khr::Surface::new(&entry, &instance);
    let surface = SurfaceKHR::from_raw(
        window.vulkan_create_surface(instance.handle().as_raw() as usize).unwrap()
    );

    cfg_if::cfg_if! {
        if #[cfg(debug_assertions)] {
            let debug_msg = create_debug_messenger(&entry, &instance);
        }
    }

    let physical_device = choose_physical_device(&instance);
    
    println!("Device:");
    println!("\t{}", string_from_slice(
        &instance.get_physical_device_properties(physical_device).device_name
    ).unwrap());

    let (graphics_index, present_index) = get_queue_indices(&instance, physical_device, &surface_loader, surface);
    let queue_indices: HashSet<u32, RandomState> = HashSet::from_iter([
        graphics_index,
        present_index
    ]);

    let mut device_queue_infos = Vec::with_capacity(queue_indices.len());
    for index in queue_indices {
        let info = DeviceQueueCreateInfo {
            queue_count: 1,
            queue_family_index: index,
            p_queue_priorities: [1.0f32].as_ptr(),

            ..Default::default()
        };

        device_queue_infos.push(info);
    }

    let device_features = vk::PhysicalDeviceFeatures::default();

    //check for swapchain support
    let device_exts = get_device_extensions(&instance, physical_device);

    println!("Device Extensions:");
    for e in &device_exts.slices {
        println!("\t{}", string_from_slice(e).unwrap());
    }

    let device_create_info = DeviceCreateInfo {
        pp_enabled_layer_names: layers.ptrs.as_ptr(),
        enabled_layer_count: layers.len(),

        pp_enabled_extension_names: device_exts.ptrs.as_ptr(),
        enabled_extension_count: device_exts.len(),

        p_queue_create_infos: device_queue_infos.as_ptr(),
        queue_create_info_count: device_queue_infos.len() as u32,

        p_enabled_features: &device_features,

        ..Default::default()
    };

    let device = instance.create_device(physical_device, &device_create_info, None).unwrap();

    let graphics_queue = device.get_device_queue(graphics_index, 0);
    let present_queue = device.get_device_queue(present_index, 0);
    
    let capabilites = surface_loader.get_physical_device_surface_capabilities(physical_device, surface).unwrap();
    let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface).unwrap();
    let present_modes = surface_loader.get_physical_device_surface_present_modes(physical_device, surface).unwrap();

    let surface_format = choose_format(formats);
    let present_mode = choose_present_mode(present_modes);

    let swap_extent = choose_swap_extent(capabilites, window.vulkan_drawable_size());

    let mut image_count = capabilites.min_image_count + 1;
    if capabilites.max_image_count > 0 && image_count > capabilites.max_image_count {
        image_count = capabilites.max_image_count;
    }

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count: image_count,
        image_format: surface_format.format,
        image_color_space: surface_format.color_space,
        image_extent: swap_extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        pre_transform: capabilites.current_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        clipped: vk::TRUE,

        present_mode,
        ..Default::default()
    };

    let queue_indices_array = [graphics_index, present_index];
    if graphics_index != present_index {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
        swapchain_create_info.queue_family_index_count = 2;
        swapchain_create_info.p_queue_family_indices = queue_indices_array.as_ptr();
    } else {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
        swapchain_create_info.queue_family_index_count = 0;
        swapchain_create_info.p_queue_family_indices = std::ptr::null();
    }

    let swapchain_loader = khr::Swapchain::new(&instance, &device);
    let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap();

    let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

    let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());
    for img in &swapchain_images {
        let components = vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        };

        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,

            ..Default::default()
        };

        let view_info = vk::ImageViewCreateInfo {
            format: surface_format.format,
            image: *img,
            view_type: vk::ImageViewType::TYPE_2D,
            components,
            subresource_range,

            ..Default::default()
        };

        let view = device.create_image_view(&view_info, None).unwrap();
        swapchain_image_views.push(view);
    }

    const VERTEX_SHADER: &[u32] = vk_shader_macros::include_glsl!("shaders/triangle.vert");
    const FRAGMENT_SHADER: &[u32] = vk_shader_macros::include_glsl!("shaders/triangle.frag");

    let vertex_module = create_shader_module(&device, VERTEX_SHADER);
    let fragment_module = create_shader_module(&device, FRAGMENT_SHADER);


    let entry_point = "main\0".as_bytes().as_ptr().cast();
    let vert_shader_stage = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::VERTEX,
        module: vertex_module,
        p_name: entry_point,
        ..Default::default()
    };
    let frag_shader_stage = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::FRAGMENT,
        module: fragment_module,
        p_name: entry_point,
        ..Default::default()
    };

    let shader_stages = [ vert_shader_stage, frag_shader_stage ];

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
        vertex_binding_description_count: 0,
        vertex_attribute_description_count: 0,

        ..Default::default()
    };

    let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: vk::FALSE,
        
        ..Default::default()
    };

    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swap_extent.width as f32,
        height: swap_extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,

        ..Default::default()
    };

    let scissor = vk::Rect2D {
        offset: vk::Offset2D{ x: 0, y: 0 },
        extent: swap_extent,
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo {
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor,

        ..Default::default()
    };

    let rasterizer = vk::PipelineRasterizationStateCreateInfo {
        depth_clamp_enable: vk::FALSE,
        rasterizer_discard_enable: vk::FALSE,
        polygon_mode: vk::PolygonMode::FILL,
        line_width: 1.0,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::CLOCKWISE,
        depth_bias_enable: vk::FALSE,

        ..Default::default()
    };

    let multisampling = vk::PipelineMultisampleStateCreateInfo {
        sample_shading_enable: vk::FALSE,
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };

    let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::all(),
        blend_enable: vk::FALSE,

        ..Default::default()
    };

    let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&[color_blend_attachment])
        .build();

    let dynamic_states = [ ];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&dynamic_states)
        .build();

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .build();

    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None).unwrap();

    let color_attachment = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&[color_attachment_ref])
        .build();

    let subpass_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build();

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&[color_attachment])
        .subpasses(&[subpass])
        .dependencies(&[subpass_dependency])
        .build();

    let render_pass = device.create_render_pass(&render_pass_info, None).unwrap();

    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build();

    let pipeline_cache = vk::PipelineCache::null();
    
    let graphics_pipeline = device.create_graphics_pipelines(pipeline_cache, &[pipeline_info], None).unwrap()[0];

    let mut framebuffers = Vec::with_capacity(swapchain_image_views.len());
    for img in &swapchain_image_views {
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&[*img])
            .width(swap_extent.width)
            .height(swap_extent.height)
            .layers(1)
            .build();

        let framebuffer = device.create_framebuffer(&framebuffer_info, None).unwrap();
        framebuffers.push(framebuffer);
    }

    let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(graphics_index)
        .build();
    let command_pool = device.create_command_pool(&pool_info, None).unwrap();

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(framebuffers.len() as u32)
        .build();

    let command_buffers = device.allocate_command_buffers(&alloc_info).unwrap();
    for i in 0..command_buffers.len() {
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .build();

        device.begin_command_buffer(command_buffers[i], &begin_info).unwrap();
        
        let clear_color = vk::ClearValue{
            color: vk::ClearColorValue{ float32: [0.0, 0.0, 0.0, 1.0] }
        };

        let render_pass_begin = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffers[i])
            .render_area(Rect2D{
                offset: Offset2D{ x: 0, y: 0 },
                extent: swap_extent
            })
            .clear_values(&[ clear_color ])
            .build();

        device.cmd_begin_render_pass(command_buffers[i], &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(command_buffers[i], vk::PipelineBindPoint::GRAPHICS, graphics_pipeline);
        device.cmd_draw(command_buffers[i], 3, 1, 0, 0);
        device.cmd_end_render_pass(command_buffers[i]);
        
        device.end_command_buffer(command_buffers[i]).unwrap();
    }

    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED)
        .build();
    let mut image_available_semaphores = Vec::with_capacity( Application::max_frames_in_flight() );
    let mut render_finished_semaphores = Vec::with_capacity( Application::max_frames_in_flight() );
    let mut in_flight_fences = Vec::with_capacity( Application::max_frames_in_flight() );
    let mut images_in_flight = Vec::new();

    images_in_flight.resize( swapchain_images.len(), vk::Fence::null() );

    for _ in 0..Application::max_frames_in_flight() {
        image_available_semaphores.push( device.create_semaphore(&semaphore_info, None).unwrap() );
        render_finished_semaphores.push( device.create_semaphore(&semaphore_info, None).unwrap() );
        in_flight_fences.push( device.create_fence(&fence_info, None).unwrap() );
    }

    Application {
        sdl_context,
        window,

        entry,
        instance,
        physical_device,
        device,
        queue_indices: QueueIndices{ graphics: graphics_index, present: present_index },
        device_queues: DeviceQueues{ graphics: graphics_queue, present: present_queue },
        surface_loader,
        surface,
        surface_capabilites: capabilites,
        surface_format,
        present_mode,
        swap_extent,
        swapchain_loader,
        swapchain,
        swapchain_images,
        swapchain_image_views,
        swapchain_framebuffers: framebuffers,
        vertex_module,
        fragment_module,
        graphics_pipeline,
        graphics_pipeline_layout: pipeline_layout,
        render_pass,
        command_pool,
        command_buffers,
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight,

        current_frame: 0,

        #[cfg(debug_assertions)]
        debug_loader: debug_msg.0,
        #[cfg(debug_assertions)]
        debug: debug_msg.1,
    }
}

fn create_shader_module(device: &Device, data: &[u32]) -> ShaderModule
{
    let create_info = vk::ShaderModuleCreateInfo {
        code_size: data.len()*4,
        p_code: data.as_ptr(),

        ..Default::default()
    };

    unsafe{ device.create_shader_module(&create_info, None) }.unwrap()
}

fn choose_swap_extent(capabilites: SurfaceCapabilitiesKHR, size: (u32, u32)) -> Extent2D
{
    if capabilites.current_extent.width != u32::MAX {
        capabilites.current_extent
    } else {
        let width = size.0.clamp(
            capabilites.min_image_extent.width,
            capabilites.max_image_extent.width
        );

        let height = size.1.clamp(
            capabilites.min_image_extent.height,
            capabilites.max_image_extent.height
        );

        Extent2D { width, height }
    }
    
}

fn choose_format(formats: Vec<SurfaceFormatKHR>) -> SurfaceFormatKHR
{
    let mut fmt = None;

    for f in formats {
        if f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
            fmt = Some(f);
        }
    }

    fmt.expect("failed to find requested format")
}

fn choose_present_mode(modes: Vec<PresentModeKHR>) -> PresentModeKHR
{
    let mode_list = [
        PresentModeKHR::FIFO,
        PresentModeKHR::IMMEDIATE,
    ];

    for m in modes {
        for ml in mode_list {
            if m == ml { return m; }
        }
    }

    panic!("failed to find requested present mode")
}

fn get_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Pin<Box<Extensions>>
{
    let props = unsafe{ instance.enumerate_device_extension_properties(physical_device) }.unwrap();

    let mut extensions =  Extensions::default();
    let required_exts: HashSet<&str, RandomState> = HashSet::from_iter([
        "VK_KHR_swapchain"
    ]);
    extensions.slices.reserve( required_exts.len() );

    for p in props {
        let s = string_from_slice(&p.extension_name).unwrap();

        if required_exts.contains(s) {
            extensions.slices.push(p.extension_name);
        }
    }

    extensions.ptrs = extensions.slices.iter().map(|e|  e.as_ptr()).collect();
    Box::pin( extensions )
}
