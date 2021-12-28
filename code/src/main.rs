mod renderer;

use crate::{
    renderer::Renderer,
};

use ash::vk::PresentModeKHR;

use std::time::{ Duration, Instant };
use std::thread::sleep;


struct Application 
{
    renderer: Renderer,
} 
impl Application
{
    fn new() -> Self
    {
        let renderer = unsafe{
            Renderer::new(800, 800)
        };

        Self {
            renderer,
        }
    }

    fn run(&mut self) 
    {
        let mut pump = self.renderer.window().sdl_context.event_pump().unwrap();
        
        'running: loop {
            let start_time = Instant::now();

            for ev in pump.poll_iter() {
                if !self.handle_event(&ev) { break 'running; }
            }

            self.renderer.draw_frame();
            let end_time = Instant::now();


            if self.renderer.present_mode == PresentModeKHR::IMMEDIATE {
                //if not using fifo present (vsync) 
                //clamp fps to 60

                let frame_time = end_time - start_time;
                sleep( Duration::from_secs_f32( 1.0 / 60.0 ) - frame_time );
            }

        }

        unsafe{
            self.renderer.device.device_wait_idle().unwrap();
        };

    }

    fn handle_event(&mut self, ev: &sdl2::event::Event) -> bool
    {
        use sdl2::event::Event;
        use sdl2::keyboard::Keycode;

        match ev {
            Event::KeyDown { keycode, .. } => { 
                match keycode {
                    Some(Keycode::Escape) => { return false; },
                    _ => {  }
                };
            },
            Event::Window { win_event, .. } => {
                self.window_event(&win_event);
            },
            Event::Quit { .. } => { return false; }
            _ => {  }
        };

        true
    }

    fn window_event(&mut self, ev: &sdl2::event::WindowEvent) 
    {
        use sdl2::event::WindowEvent;

        match ev {
            WindowEvent::Resized(width, height) => {
                self.renderer.on_resize(*width, *height);
            },
            _ => {}
        }
    }
}


pub const PROGRAM_NAME: &'static str = env!("CARGO_PKG_NAME");

fn main() 
{
    let mut app = Application::new();
    app.run();
}




