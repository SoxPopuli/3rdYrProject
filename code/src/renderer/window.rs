use sdl2::{
    Sdl,
    video,
};

pub struct Window 
{
    pub sdl_context: Sdl,
    pub window: video::Window,
}

impl Window
{
    pub fn new(width: u32, height: u32) -> Self 
    {
       let sdl_context = sdl2::init().unwrap();
       let video = sdl_context.video().unwrap();

       let window = video.window(crate::PROGRAM_NAME, width, height)
           .position_centered()
           .resizable()
           .vulkan()
           .build()
           .unwrap();

        Self {
            sdl_context,
            window,
        }
    }
}

