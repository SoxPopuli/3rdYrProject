pub mod window;
use window::Window;

mod ubo;
use ubo::UniformBufferObject;

use crate::vertex::Vertex;

use nalgebra_glm as glm;
use nalgebra::{
    Matrix4,
    Vector2,
    Vector3,
};

use lazy_static::lazy_static;

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
    io::{
        BufReader,
        Cursor,
    },
    collections::{
        HashSet,
        hash_map::RandomState
    }, 
    time::Instant,
    ffi::{ CStr, c_void }, 
    iter::{FromIterator}, 
    sync::Once,
    marker::PhantomPinned, 
    pin::Pin, 
    str::Utf8Error,
    cell::Cell,
    rc::Rc,
};

static INITIAL: Once = Once::new();

pub struct BufferMemory
{
    buf: vk::Buffer,
    mem: vk::DeviceMemory
}
impl BufferMemory {
    fn destroy(&mut self, device: &Device) {
        unsafe{
            device.destroy_buffer(self.buf, None);
            device.free_memory(self.mem, None);

            self.buf = vk::Buffer::null();
            self.mem = vk::DeviceMemory::null();
        }
    }

    fn copy_into<T>(&self, device: &Device, data: &[T], size: u64) {
        unsafe{
            let ptr = device.map_memory(self.mem, 0, size, vk::MemoryMapFlags::empty()).expect("failed to map buffer memory");
            ptr.copy_from(data.as_ptr().cast(), size as usize);
            device.unmap_memory(self.mem);
        }
    }
}

pub struct Renderer 
{
    window: Window,

    framebuffer_resized: bool,

    pub entry: Entry,
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub physical_device: vk::PhysicalDevice,
    pub queue_indices: QueueIndices,

    pub surface_loader: khr::Surface,
    pub surface: vk::SurfaceKHR,

    pub device_queues: DeviceQueues,

    pub surface_capabilites: vk::SurfaceCapabilitiesKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub swap_extent: Extent2D,

    pub swapchain_loader: khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>,

    pub vertex_module: vk::ShaderModule,
    pub fragment_module: vk::ShaderModule,

    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,

    pub render_pass: vk::RenderPass,
    pub graphics_pipeline: vk::Pipeline,
    pub graphics_pipeline_layout: vk::PipelineLayout,

    pub vertex_buffer: BufferMemory,
    pub index_buffer: BufferMemory,

    pub uniform_buffers: Vec<BufferMemory>,

    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,

    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,

    pub image_texture: vk::Image,
    pub image_texture_memory: vk::DeviceMemory,
    pub image_texture_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,

    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,

    pub current_frame: usize,

    #[cfg(feature = "validation")]
    debug_loader: ext::DebugUtils,
    #[cfg(feature = "validation")]
    debug: vk::DebugUtilsMessengerEXT,

    start_time: Instant,
}

impl Renderer
{
    const fn max_frames_in_flight() -> usize { 2 }

    fn get_vertices() -> [Vertex; 8] {
         let vertices: [Vertex; 8] = [
            Vertex::new([-0.5, -0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([0.5, -0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex::new([0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]),
            Vertex::new([-0.5, 0.5, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0]),

            Vertex::new([-0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([0.5, -0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex::new([0.5, 0.5, -0.5], [0.0, 0.0, 1.0], [1.0, 1.0]),
            Vertex::new([-0.5, 0.5, -0.5], [1.0, 1.0, 1.0], [0.0, 1.0]),
        ];

         vertices
    }

    fn get_indices() -> [u16; 36] {
        [
            //bottom
            0, 1, 2, 
            2, 3, 0,

            //top
            4, 5, 6,
            6, 7, 4,

            //back
            4, 5, 1,
            1, 0, 4,

            //left
            7, 4, 0,
            0, 3, 7,

            //right
            6, 5, 1,
            1, 2, 6,

            //front
            7, 6, 2,
            2, 3, 7
        ]
    }



    fn create_buffer(&self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, props: vk::MemoryPropertyFlags, buffer_memory: &vk::DeviceMemory) -> BufferMemory
    {
        create_buffer(&self.instance, self.physical_device, &self.device, size, usage, props)
    }

    fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, size: u64)
    {
        copy_buffer(&self.device, self.command_pool, self.device_queues.graphics, 
            src, dst, size);
    }

    fn create_vertex_buffer(&self, vertices: &[Vertex]) -> BufferMemory
    {
        let buffer;
        unsafe{
            buffer = create_vertex_buffer(&self.instance, self.physical_device, &self.device, self.command_pool, self.device_queues.graphics, vertices);
        }

        buffer
    }

    pub fn window(&self) -> &Window { &self.window }
    pub fn window_mut(&mut self) -> &mut Window { &mut self.window }

    pub unsafe fn new(width: u32, height: u32) -> Self
    {
        let entry = Entry::new().unwrap();
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 0, 0),
            ..Default::default()
        };

        let sdl_window = Window::new(width, height);

        let mut required_exts: Vec<_> = sdl_window.window.vulkan_instance_extensions().unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut required_layers = Vec::<String>::new();

        if cfg!(feature = "validation") {
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
            sdl_window.window.vulkan_create_surface(instance.handle().as_raw() as usize).unwrap()
        );

        cfg_if::cfg_if! {
            if #[cfg(feature = "validation")] {
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

        let device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

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

        let swap_extent = choose_swap_extent(capabilites, sdl_window.window.vulkan_drawable_size());

        let swapchain_loader = khr::Swapchain::new(&instance, &device);
        let swapchain = create_swapchain(&instance, 
            &device, 
            &swapchain_loader,
            graphics_index, 
            present_index,
            surface,
            present_mode, 
            capabilites,
            surface_format,
            swap_extent,
            None
        );

        let swapchain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
        let swapchain_image_views = create_image_views(&device, surface_format, &swapchain_images);

        const VERTEX_SHADER: &[u32] = vk_shader_macros::include_glsl!("shaders/triangle.vert");
        const FRAGMENT_SHADER: &[u32] = vk_shader_macros::include_glsl!("shaders/triangle.frag");

        let descriptor_set_layout = create_descriptor_set_layout(&device);

        let vertex_module = create_shader_module(&device, VERTEX_SHADER);
        let fragment_module = create_shader_module(&device, FRAGMENT_SHADER);

        let render_pass = create_render_pass(&device, surface_format);

        let (pipeline_layout, graphics_pipeline) = create_graphics_pipeline(
            &device,
            vertex_module,
            fragment_module,
            swap_extent,
            render_pass,
            descriptor_set_layout,
            vk::PipelineCache::null()
        );


        let pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_index)
            .build();
        let command_pool = device.create_command_pool(&pool_info, None).unwrap();

        let depth_resources = create_depth_resources(&instance, physical_device, &device, graphics_queue, command_pool, width, height);

        let framebuffers = create_framebuffers(&device, &swapchain_image_views, swap_extent, depth_resources.2, render_pass);


        let vertices = Renderer::get_vertices();
        let vertex_buffer = create_vertex_buffer(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            &vertices);

        let indices = Renderer::get_indices();
        let index_buffer = create_index_buffer(
            &instance,
            physical_device,
            &device,
            command_pool,
            graphics_queue,
            &indices
        );

        let uniform_buffers = create_uniform_buffers(&instance, physical_device, &device, swapchain_images.len());

        let (image_texture, image_texture_memory) = create_texture_image(
            &instance, 
            physical_device, 
            &device,
            graphics_queue,
            command_pool
        );

        let image_texture_view = create_image_view(&device, image_texture, vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR);
        let texture_sampler = create_texture_sampler(&device);

        let descriptor_pool = create_descriptor_pool(&device, swapchain_images.len() as u32);
        let descriptor_sets = create_descriptor_sets(
            &device,
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
            image_texture_view,
            texture_sampler,
            swapchain_images.len() as u32
        );

        let command_buffers = create_command_buffers(&device, 
            command_pool, 
            graphics_index, 
            render_pass, 
            &framebuffers, 
            swap_extent, 
            graphics_pipeline, 
            pipeline_layout,
            &vertices, 
            &indices,
            &descriptor_sets,
            vertex_buffer.buf,
            index_buffer.buf);

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();
        let mut image_available_semaphores = Vec::with_capacity( Self::max_frames_in_flight() );
        let mut render_finished_semaphores = Vec::with_capacity( Self::max_frames_in_flight() );
        let mut in_flight_fences = Vec::with_capacity( Self::max_frames_in_flight() );
        let mut images_in_flight = Vec::new();

        images_in_flight.resize( swapchain_images.len(), vk::Fence::null() );

        for _ in 0..Self::max_frames_in_flight() {
            image_available_semaphores.push( device.create_semaphore(&semaphore_info, None).unwrap() );
            render_finished_semaphores.push( device.create_semaphore(&semaphore_info, None).unwrap() );
            in_flight_fences.push( device.create_fence(&fence_info, None).unwrap() );
        }

        let instance = Rc::new(instance);
        let device = Rc::new(device);

        Self {
            window: sdl_window,
            framebuffer_resized: false,

            entry,
            instance,
            device,
            physical_device,
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
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            vertex_module,
            fragment_module,
            graphics_pipeline,
            graphics_pipeline_layout: pipeline_layout,
            vertex_buffer,
            index_buffer,
            uniform_buffers,
            image_texture,
            image_texture_memory,
            image_texture_view,
            texture_sampler,
            render_pass,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,

            depth_image: depth_resources.0,
            depth_image_memory: depth_resources.1,
            depth_image_view: depth_resources.2,

            current_frame: 0,
            start_time: Instant::now(),

            #[cfg(feature = "validation")]
            debug_loader: debug_msg.0,
            #[cfg(feature = "validation")]
            debug: debug_msg.1,
        }
    }

    pub fn draw_frame(&mut self)
    {
        let in_flight_fences = [self.in_flight_fences[self.current_frame]];
        unsafe{ self.device.wait_for_fences(&in_flight_fences, true, u64::MAX) }.unwrap();

        let image_result = unsafe{ self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available_semaphores[self.current_frame],
            vk::Fence::null()
        ) };

        let image_index = match image_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swap_chain();
                return;
            },
            _ => panic!("failed to acquire next image from swapchain")
        };

        update_uniform_buffers(
            self.start_time,
            &self.device,
            self.swap_extent.width,
            self.swap_extent.height,
            &mut self.uniform_buffers[image_index as usize]
        );

        if self.images_in_flight[image_index as usize] != vk::Fence::null() {
            unsafe{ self.device.wait_for_fences(&[self.images_in_flight[image_index as usize]], true, u64::MAX) }.unwrap();
        }
        self.images_in_flight[image_index as usize] = self.in_flight_fences[ self.current_frame ];

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

        let dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[image_index as usize]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&dst_stage_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores)
            .build();

        unsafe{ self.device.reset_fences(&[self.in_flight_fences[self.current_frame]]) }.unwrap();
        unsafe{ self.device.queue_submit(self.device_queues.graphics, &[submit_info], self.in_flight_fences[self.current_frame]) }.unwrap();

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .build();
            
        let present_result = unsafe{ self.swapchain_loader.queue_present(self.device_queues.present, &present_info) };

        let mut swapchain_recreated = false;

        match present_result {
            Ok(_) => {},
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) |
            Err(vk::Result::SUBOPTIMAL_KHR) => { 
                self.recreate_swap_chain();
                swapchain_recreated = true;
            },
            _ => { panic!("failed to present swapchain image") }
        };
        //unsafe{ self.device.queue_wait_idle(self.device_queues.present) }.unwrap();
        if self.framebuffer_resized && !swapchain_recreated {
            self.framebuffer_resized = false;

            self.recreate_swap_chain();
        }

        self.current_frame = (self.current_frame + 1) % Self::max_frames_in_flight()
    }

    pub fn recreate_swap_chain(&mut self) 
    {
        unsafe{

            let (mut width, mut height) = self.window.window.vulkan_drawable_size();
            while width == 0 || height == 0 {
                let new_size = self.window.window.vulkan_drawable_size();
                width = new_size.0;
                height = new_size.1;

                sdl2::sys::SDL_WaitEvent( std::ptr::null_mut() );
            }

            self.device.device_wait_idle().expect("device failed to wait");

            let size = self.window.window.vulkan_drawable_size();

            self.surface_capabilites = self.surface_loader.get_physical_device_surface_capabilities(self.physical_device, self.surface).unwrap();
            self.swap_extent = choose_swap_extent(self.surface_capabilites, size);

            let old_swapchain = self.swapchain;
            self.swapchain = create_swapchain(&self.instance, 
                &self.device, 
                &self.swapchain_loader,
                self.queue_indices.graphics, 
                self.queue_indices.present, 
                self.surface, 
                self.present_mode, 
                self.surface_capabilites, 
                self.surface_format, 
                self.swap_extent,
                Some(self.swapchain)
            );

            self.cleanup_swapchain(old_swapchain);

            self.swapchain_images = self.swapchain_loader.get_swapchain_images(self.swapchain).unwrap();
            self.swapchain_image_views = create_image_views(&self.device, self.surface_format, &self.swapchain_images);
            self.render_pass = create_render_pass(&self.device, self.surface_format);

            let (new_pipeline_layout, new_graphics_pipeline) = create_graphics_pipeline(
                &self.device, 
                self.vertex_module, 
                self.fragment_module, 
                self.swap_extent, 
                self.render_pass, 
                self.descriptor_set_layout,
                vk::PipelineCache::null()
            );
            self.graphics_pipeline_layout = new_pipeline_layout;
            self.graphics_pipeline = new_graphics_pipeline;

            let depth_resources = create_depth_resources(
                &self.instance,
                self.physical_device,
                &self.device,
                self.device_queues.graphics,
                self.command_pool,
                width,
                height
            );

            self.depth_image = depth_resources.0;
            self.depth_image_memory = depth_resources.1;
            self.depth_image_view = depth_resources.2;

            self.swapchain_framebuffers = create_framebuffers(&self.device, &self.swapchain_image_views, self.swap_extent, self.depth_image_view, self.render_pass);

            let vertices = Self::get_vertices();
            let indices = Self::get_indices();
            
            self.uniform_buffers = create_uniform_buffers(
                &self.instance,
                self.physical_device,
                &self.device,
                self.swapchain_images.len()
            );

            self.descriptor_pool = create_descriptor_pool(&self.device, self.swapchain_images.len() as u32);

            self.descriptor_sets = create_descriptor_sets(
                &self.device,
                self.descriptor_pool,
                self.descriptor_set_layout,
                &self.uniform_buffers,
                self.image_texture_view,
                self.texture_sampler,
                self.swapchain_images.len() as u32
            );

            self.command_buffers = create_command_buffers(
                &self.device, 
                self.command_pool,
                self.queue_indices.graphics, 
                self.render_pass, 
                &self.swapchain_framebuffers, 
                self.swap_extent, 
                self.graphics_pipeline,
                self.graphics_pipeline_layout,
                &vertices,
                &indices,
                &self.descriptor_sets,
                self.vertex_buffer.buf,
                self.index_buffer.buf
            );
        }
    }

    fn cleanup_swapchain(&mut self, old_swapchain: vk::SwapchainKHR) 
    {
        unsafe{
            self.device.free_command_buffers(self.command_pool, self.command_buffers.as_slice());

            for fb in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(*fb, None);
            }

            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);

            for b in self.uniform_buffers.iter_mut() {
                b.destroy(&self.device);
            }

            self.device.destroy_descriptor_pool(self.descriptor_pool, None);

            self.device.destroy_pipeline_layout(self.graphics_pipeline_layout, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_render_pass(self.render_pass, None);
            
            for view in &self.swapchain_image_views {
                self.device.destroy_image_view(*view, None);
            }

            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
    }

    pub fn on_resize(&mut self, new_width: i32, new_height: i32)
    {
        self.framebuffer_resized = false;
    }

    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32, String>
    {
        find_memory_type(&self.instance, self.physical_device, type_filter, properties)
    }


    pub fn set_vertex_module(&mut self, module: vk::ShaderModule) 
    {
        unsafe{
            self.device.destroy_shader_module(self.vertex_module, None);
        }
        self.vertex_module = module;
    }

    pub fn set_fragment_module(&mut self, module: vk::ShaderModule)
    {
        unsafe{
            self.device.destroy_shader_module(self.fragment_module, None);
        }
        self.fragment_module = module;
    }
}
impl Drop for Renderer 
{
    fn drop(&mut self) 
    {
        unsafe {
            for i in 0..Self::max_frames_in_flight() {
                self.device.destroy_semaphore(self.image_available_semaphores[i], None);
                self.device.destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
                //self.device.destroy_fence(self.images_in_flight[i], None);
            }

            self.cleanup_swapchain(self.swapchain);

            self.device.destroy_sampler(self.texture_sampler, None);
            self.device.destroy_image_view(self.image_texture_view, None);
            self.device.destroy_image(self.image_texture, None);
            self.device.free_memory(self.image_texture_memory, None);

            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.index_buffer.destroy(&self.device);
            self.vertex_buffer.destroy(&self.device);

            for i in 0..self.swapchain_images.len() {
                self.uniform_buffers[i].destroy(&self.device);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            //images destroyed by destroy_swapchain
            //for img in &self.swapchain_images {
            //    self.device.destroy_image(*img, None);
            //}

            self.surface_loader.destroy_surface(self.surface, None);

            self.device.destroy_shader_module(self.vertex_module, None);
            self.device.destroy_shader_module(self.fragment_module, None);

            cfg_if::cfg_if! {
                if #[cfg(feature = "validation")] {
                    self.debug_loader.destroy_debug_utils_messenger(self.debug, None);
                }
            }

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

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

pub struct QueueIndices 
{
    graphics: u32,
    present: u32,
}

pub struct DeviceQueues 
{
    graphics: vk::Queue,
    present: vk::Queue,
}


// free functions
fn create_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    size: vk::DeviceSize, 
    usage: vk::BufferUsageFlags, 
    props: vk::MemoryPropertyFlags,
    ) -> BufferMemory
{
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size as u64)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    let buffer;
    let memory_reqs;
    unsafe{ 
        buffer = device.create_buffer(&buffer_info, None).expect("failed to create buffer");
        memory_reqs = device.get_buffer_memory_requirements(buffer);
    }

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_reqs.size)
        .memory_type_index(find_memory_type(instance, physical_device, memory_reqs.memory_type_bits, props).unwrap())
        .build();

    let buffer_memory;
    unsafe{
        buffer_memory = device.allocate_memory(&alloc_info, None).expect("failed to allocate buffer memory");
        device.bind_buffer_memory(buffer, buffer_memory, 0);
    }

    BufferMemory { buf: buffer, mem: buffer_memory }
}


fn create_index_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    indices: &[u16]) -> BufferMemory
{
    let size = (std::mem::size_of_val(&indices[0]) * indices.len()) as u64;

    let mut staging_buffer = create_staging_buffer(instance, physical_device, device, size);
    staging_buffer.copy_into(device, indices, size);

    let index_buffer = create_buffer(instance, physical_device, device, size, 
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL);

    copy_buffer(device, command_pool, graphics_queue, staging_buffer.buf, index_buffer.buf, size);
    staging_buffer.destroy(device);

    index_buffer
}

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
    unsafe extern "system" fn debug_callback(severity: vk::DebugUtilsMessageSeverityFlagsEXT, 
        msg_type: vk::DebugUtilsMessageTypeFlagsEXT, 
        cb: *const vk::DebugUtilsMessengerCallbackDataEXT, 
        data: *mut c_void) -> u32
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

unsafe fn create_swapchain(
    instance: &Instance,
    device: &Device,
    swapchain_loader: &khr::Swapchain,
    graphics_index: u32,
    present_index: u32,
    surface: vk::SurfaceKHR, 
    present_mode: vk::PresentModeKHR, 
    capabilites: SurfaceCapabilitiesKHR, 
    surface_format: SurfaceFormatKHR, 
    swap_extent: vk::Extent2D,
    old_swapchain: Option<vk::SwapchainKHR>
) -> vk::SwapchainKHR
{
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
        old_swapchain: old_swapchain.unwrap_or(vk::SwapchainKHR::null()),
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

    let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None).unwrap();

    swapchain
}

unsafe fn create_image_views(device: &Device, surface_format: vk::SurfaceFormatKHR, swapchain_images: &Vec<vk::Image>) -> Vec<vk::ImageView>
{
    let mut swapchain_image_views = Vec::with_capacity(swapchain_images.len());
    for img in swapchain_images {
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

    swapchain_image_views
}

unsafe fn create_render_pass(device: &Device, surface_format: vk::SurfaceFormatKHR) -> vk::RenderPass
{
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

    let color_attachment_refs = [ 
        vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build()
    ];

    let depth_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();

    let depth_attachment_ref = 
        vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attachment_ref)
        .build();

    let subpass_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .build();

    let attachments = [color_attachment, depth_attachment];
    let subpasses = [subpass];
    let subpass_dependencies = [subpass_dependency];
    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&subpass_dependencies)
        .build();

    device.create_render_pass(&render_pass_info, None).unwrap()
}

fn find_memory_type(
    instance: &Instance, 
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Result<u32, String> {
    let mem_props = unsafe{ instance.get_physical_device_memory_properties(physical_device) };

    for i in 0..mem_props.memory_type_count {
        if (type_filter & (1 << i) != 0) &&
            ((mem_props.memory_types[i as usize].property_flags & properties) == properties) {

            return Ok(i);
        }
    }

    Err(format!("Failed to find memory type {}", type_filter))
}

unsafe fn allocate_vertex_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    buffer: vk::Buffer
) -> vk::DeviceMemory {
    let mem_req = device.get_buffer_memory_requirements(buffer);
    let mem_type = find_memory_type(instance, physical_device, mem_req.memory_type_bits, 
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT);

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type.unwrap())
        .build();

    device.allocate_memory(&alloc_info, None).expect("failed to allocate vertex buffer")
}

fn create_staging_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    size: u64) -> BufferMemory
{
    let buffer = create_buffer(instance, physical_device, device, size, 
        vk::BufferUsageFlags::TRANSFER_SRC, 
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE);

    buffer
}

unsafe fn create_vertex_buffer(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    vertices: &[Vertex]
) -> BufferMemory {
    let size = ( std::mem::size_of::<Vertex>() * vertices.len() ) as u64;
    let mut staging_buffer = create_staging_buffer(instance, physical_device, device, size);

    //get cpu accessable memory
    let data = device.map_memory(staging_buffer.mem, 0, size, vk::MemoryMapFlags::empty()).expect("failed to map vertex buffer");

    //copy memory into buffer
    data.copy_from(vertices.as_ptr().cast(), size as usize);
    device.unmap_memory(staging_buffer.mem);

    let buffer = create_buffer(instance, physical_device, device, size, 
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL);

    copy_buffer(device, command_pool, graphics_queue, staging_buffer.buf, buffer.buf, size);
    staging_buffer.destroy(device);
    
    buffer
}

fn copy_buffer(
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
    src: vk::Buffer, dst: vk::Buffer, size: u64)
{
    let buffer = begin_single_time_command(device, command_pool);

    unsafe{
        let copy_region = vk::BufferCopy::builder()
            .size(size)
            .build();

        let regions = [copy_region];
        device.cmd_copy_buffer(buffer, src, dst, &regions);
    }

    end_single_time_command(device, graphics_queue, buffer, command_pool);
}

unsafe fn create_graphics_pipeline(device: &Device, 
    vertex_module: ShaderModule, 
    fragment_module: ShaderModule,
    swap_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_cache: vk::PipelineCache) -> (vk::PipelineLayout, vk::Pipeline)
{
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

    let vertex_bindings = [ Vertex::binding_description() ];
    let vertex_attributes = Vertex::attribute_descriptions();

    let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attributes)
        .build();

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
        cull_mode: vk::CullModeFlags::NONE,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
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

    let color_blend_attachments = [color_blend_attachment];
    let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments)
        .build();

    let dynamic_states = [ ];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
        .dynamic_states(&dynamic_states)
        .build();

    let layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .build();

    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None).unwrap();

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_info)
        .input_assembly_state(&input_assembly_info)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .depth_stencil_state(&depth_stencil)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .dynamic_state(&dynamic_state)
        .build();

    let pipeline_infos = [pipeline_info];
    let pipeline = device.create_graphics_pipelines(pipeline_cache, &pipeline_infos, None).unwrap()[0];

    (pipeline_layout, pipeline)
}

unsafe fn create_framebuffers(device: &Device, 
    swapchain_image_views: &Vec<vk::ImageView>, 
    swap_extent: vk::Extent2D, 
    depth_image_view: vk::ImageView,
    render_pass: vk::RenderPass) 
    -> Vec<vk::Framebuffer>
{
    let mut width = swap_extent.width;
    let mut height = swap_extent.height;

    let mut framebuffers = Vec::with_capacity(swapchain_image_views.len());
    for img in swapchain_image_views {

        let framebuffer_attachments = [*img, depth_image_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&framebuffer_attachments)
            .width(width)
            .height(height)
            .layers(1)
            .build();

        let framebuffer = device.create_framebuffer(&framebuffer_info, None).unwrap();
        framebuffers.push(framebuffer);
    }

    framebuffers
}

unsafe fn create_command_buffers(
    device: &Device,
    command_pool: vk::CommandPool,
    graphics_index: u32,
    render_pass: vk::RenderPass,
    framebuffers: &Vec<vk::Framebuffer>,
    swap_extent: vk::Extent2D,
    graphics_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    vertices: &[Vertex],
    indices: &[u16],
    descriptor_sets: &Vec<vk::DescriptorSet>,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
) -> Vec<vk::CommandBuffer>
{

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
            color: vk::ClearColorValue{ float32: [0.0, 0.0, 0.0, 1.0] },
        };
        let depth_clear = vk::ClearValue{
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 }
        };
        let clear_values = [ clear_color, depth_clear ];

        let render_pass_begin = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffers[i])
            .render_area(Rect2D{
                offset: Offset2D{ x: 0, y: 0 },
                extent: swap_extent
            })
            .clear_values(&clear_values)
            .build();

        device.cmd_begin_render_pass(command_buffers[i], &render_pass_begin, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(command_buffers[i], vk::PipelineBindPoint::GRAPHICS, graphics_pipeline);

        let vertex_buffers = [vertex_buffer];
        let offsets = [0];
        device.cmd_bind_vertex_buffers(command_buffers[i], 0, &vertex_buffers, &offsets);

        device.cmd_bind_index_buffer(command_buffers[i], index_buffer, 0, vk::IndexType::UINT16);

        let sets = [descriptor_sets[i]];
        let offsets = [];
        device.cmd_bind_descriptor_sets(command_buffers[i], 
            vk::PipelineBindPoint::GRAPHICS,
            pipeline_layout,
            0, 
            &sets,
            &offsets
        );

        device.cmd_draw_indexed(command_buffers[i], indices.len() as u32, 1, 0, 0, 0);

        device.cmd_end_render_pass(command_buffers[i]);
        
        device.end_command_buffer(command_buffers[i]).unwrap();
    }

    command_buffers
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
    if modes.contains(&PresentModeKHR::FIFO) {
        PresentModeKHR::FIFO
    }
    else {
        PresentModeKHR::IMMEDIATE
    }

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

fn string_from_slice(slice: &[i8]) -> Result<&str, Utf8Error> 
{
    unsafe{ CStr::from_ptr(slice.as_ptr()) }
        .to_str()
}

fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout
{
    let layout_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();
    
    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();

    let bindings = [layout_binding, sampler_binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings)
        .build();

    unsafe{
        device.create_descriptor_set_layout(&layout_info, None)
    }.expect("failed to create descriptor set layout")
}

fn create_uniform_buffers(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    count: usize) -> Vec<BufferMemory>
{
    let buffer_size = std::mem::size_of::<UniformBufferObject>() as u64;

    let mut buffers = Vec::with_capacity(count);
    for _ in 0..count {
        buffers.push(create_buffer(instance, physical_device, device, 
                buffer_size, 
                vk::BufferUsageFlags::UNIFORM_BUFFER, 
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT));
    }

    buffers
}

fn deg_to_rad(deg: f32) -> f32
{
    deg * (std::f32::consts::PI / 180.0)
}

fn update_uniform_buffers(
    start_time: Instant,
    device: &Device,
    screen_width: u32,
    screen_height: u32,
    buffer: &mut BufferMemory) 
{
    let current_time = Instant::now();

    let time = (current_time - start_time).as_secs_f32();

    let mut ubo = UniformBufferObject::default();
    ubo.model = glm::rotate(
        &Matrix4::identity(),
        time * deg_to_rad(90.0),
        &Vector3::new(0.0, 0.0, 1.0)
    );

    ubo.view = glm::look_at(
        &Vector3::new(2.0, 2.0, 2.0),
        &Vector3::new(0.0, 0.0, 0.0),
        &Vector3::new(0.0, 0.0, 1.0)
    );

    ubo.proj = glm::perspective(
        deg_to_rad(70.0), 
        (screen_width as f32) / (screen_height as f32),
        0.1,
        10.0
    );

    //invert Y projection, to stop things being upside down
    ubo.proj[(1, 1)] *= -1.0;

    buffer.copy_into(&device, 
        &[ubo],
        std::mem::size_of_val(&ubo) as u64
    );
}

fn create_descriptor_pool(
    device: &Device,
    count: u32) -> vk::DescriptorPool
{
    let pool_size = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(count)
        .build();

    let sampler_pool = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(count)
        .build();

    let pool_sizes = [pool_size, sampler_pool];
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(count)
        .build();

    let descriptor_pool;
    unsafe{
        descriptor_pool = device.create_descriptor_pool(&pool_info, None).expect("failed to create descriptor pool");
    }

    descriptor_pool
}

fn create_descriptor_sets(
    device: &Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_buffers: &Vec<BufferMemory>,
    image_texture_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    count: u32) -> Vec<vk::DescriptorSet>
{
    let mut layouts = Vec::new();
    layouts.resize(count as usize, descriptor_set_layout);

    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts)
        .build();

    let sets;
    unsafe{
        sets = device.allocate_descriptor_sets(&alloc_info).expect("failed to allocate descriptor sets");
    }

    for i in 0..count as usize {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffers[i].buf)
            .offset(0)
            .range(std::mem::size_of::<UniformBufferObject>() as u64)
            .build();

        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(image_texture_view)
            .sampler(texture_sampler)
            .build();
        
        let buffer_infos = [buffer_info];
        let buffer_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos)
            .build();

        let image_infos = [image_info];
        let image_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)
            .build();

        let writes = [buffer_descriptor_write, image_descriptor_write];
        unsafe{
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    sets
}

const IMAGE_TEXTURE: &[u8] = include_bytes!("../../shaders/Red_brick_wall_texture.png");

fn create_texture_image(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    ) -> (vk::Image, vk::DeviceMemory)
{
    let buf_reader = BufReader::new(Cursor::new(IMAGE_TEXTURE));
    let image_reader = image::io::Reader::new(buf_reader)
        .with_guessed_format().unwrap();

    let img = image_reader.decode().unwrap();
    //let channels = img.color().channel_count();
    let width = img.width();
    let height = img.height();
    
    let bytes = img.as_rgba8().unwrap();

    let staging_buffer = create_staging_buffer(instance, physical_device, device, bytes.len() as u64);

    staging_buffer.copy_into(device, &bytes, bytes.len() as u64);

    let (image, image_memory) = create_image(instance, physical_device, device, 
        width, 
        height, 
        vk::Format::R8G8B8A8_SRGB, 
        vk::ImageTiling::OPTIMAL, 
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    );

    transition_to_image_layout(device,
        graphics_queue,
        command_pool,
        image,
        vk::Format::R8G8B8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL
    );
    copy_buffer_to_image(device, graphics_queue, command_pool, staging_buffer.buf, image, width, height);
    transition_to_image_layout(device, graphics_queue, command_pool, image, vk::Format::R8G8B8_SRGB, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);


    unsafe{
        device.destroy_buffer(staging_buffer.buf, None);
        device.free_memory(staging_buffer.mem, None);
    }

    (image, image_memory)
}

fn create_image(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    width: u32,
    height: u32,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
    ) -> ( vk::Image, vk::DeviceMemory )
{
    let image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D{
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1)
        .build();

    unsafe{
        let image = device.create_image(&image_create_info, None).unwrap();

        let image_reqs = device.get_image_memory_requirements(image);

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(image_reqs.size)
            .memory_type_index(find_memory_type(instance, physical_device, image_reqs.memory_type_bits, properties).unwrap())
            .build();

        let image_memory = device.allocate_memory(&alloc_info, None).unwrap();

        device.bind_image_memory(image, image_memory, 0).unwrap();

        (image, image_memory)
    }

}

fn begin_single_time_command(
    device: &Device,
    command_pool: vk::CommandPool,
    ) -> vk::CommandBuffer
{
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1)
        .build();

    let command_buffer;
    unsafe{
        command_buffer = device.allocate_command_buffers(&alloc_info).unwrap();
    }

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .build();

    unsafe{
        device.begin_command_buffer(command_buffer[0], &begin_info).unwrap();
    }

    return command_buffer[0];
}

fn end_single_time_command(device: &Device, 
    graphics_queue: vk::Queue, 
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool)
{
    unsafe{
        device.end_command_buffer(command_buffer).unwrap();
    }

    let buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&buffers)
        .build();

    let submits = [submit_info];
    unsafe{
        device.queue_submit(graphics_queue, &submits, vk::Fence::null()).unwrap();
        device.queue_wait_idle(graphics_queue).unwrap();
        device.free_command_buffers(command_pool, &buffers);
    }
}

fn transition_to_image_layout(
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout
) 
{
    let command_buffer = begin_single_time_command(device, command_pool);

    let subresource_range = vk::ImageSubresourceRange{
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1
    };

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .build();

    let source_stage;
    let dest_stage;

    match old_layout {
        vk::ImageLayout::UNDEFINED => {
            barrier.src_access_mask = vk::AccessFlags::empty();
            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        },
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            source_stage = vk::PipelineStageFlags::TRANSFER;
        },
        _ => { panic!("unsupported layout"); }
    };

    match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => {
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            dest_stage = vk::PipelineStageFlags::TRANSFER;
        },
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;
            dest_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        },
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => {
            barrier.dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ |
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            dest_stage = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;

            barrier.subresource_range.aspect_mask = vk::ImageAspectFlags::DEPTH;
        },
        _ => { panic!("unsupported layout"); }
    }

    unsafe{
        let barriers = [barrier];
        device.cmd_pipeline_barrier(command_buffer,
            source_stage,
            dest_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );

    }

    end_single_time_command(device, graphics_queue, command_buffer, command_pool)
}

fn copy_buffer_to_image(
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32, 
    height: u32)
{
    let command_buffer = begin_single_time_command(device, command_pool);

    let copy_region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers{
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1
        })
        .image_offset(vk::Offset3D{ x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D{
            width,
            height,
            depth: 1
        })
        .build();

    let regions = [copy_region];
    unsafe{
        device.cmd_copy_buffer_to_image(command_buffer, buffer, image, 
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
    }

    end_single_time_command(device, graphics_queue, command_buffer, command_pool)
}

fn create_image_view( device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspect_mask: vk::ImageAspectFlags) -> vk::ImageView
{
    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange{
            aspect_mask: aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1
        })
        .build();

    unsafe{
        device.create_image_view(&create_info, None).unwrap()
    }
}

fn create_texture_sampler(device: &Device) -> vk::Sampler
{
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0)
        .build();

    unsafe{
        device.create_sampler(&sampler_info, None).unwrap()
    }
}

fn create_depth_resources(instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    width: u32,
    height: u32,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView)
{
    let depth_format = vk::Format::D32_SFLOAT;

    let (image, image_memory) = create_image(instance, physical_device, device, 
        width,
        height,
        depth_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    );

    let image_view = create_image_view(device, image, depth_format, vk::ImageAspectFlags::DEPTH);

    transition_to_image_layout(device, 
        graphics_queue, 
        command_pool, 
        image, 
        depth_format, 
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );

    (image, image_memory, image_view)
}
