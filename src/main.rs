use bytemuck::{Pod, Zeroable};
use onion::engine::{make_move, try_player_move, Position, Square};
use std::borrow::Cow;
use wgpu::util::DeviceExt;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

fn mouse_to_square(
    position: PhysicalPosition<f64>,
    window_size: PhysicalSize<u32>,
) -> Option<Square> {
    if window_size.width == 0 || window_size.height == 0 {
        return None;
    }

    let x = (position.x / window_size.width as f64 * 8.0).floor() as i32;
    let y = 7 - (position.y / window_size.height as f64 * 8.0).floor() as i32;

    if !(0..8).contains(&x) || !(0..8).contains(&y) {
        return None;
    }

    Some((x + y * 8) as Square)
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct GpuBoard {
    pieces: [i32; 64],
}

unsafe impl Zeroable for GpuBoard {}
unsafe impl Pod for GpuBoard {}

impl GpuBoard {
    fn new(position: &Position) -> GpuBoard {
        let mut board = GpuBoard { pieces: [0; 64] };
        for square in 0..64 {
            board.pieces[square] = position.piece_at(square as Square) as i32;
        }
        board
    }

    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let size = window.inner_size();

    let instance = wgpu::Instance::default();
    let surface = unsafe { instance.create_surface(&window) }.unwrap();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let mut position = Position::new();
    let gpu_board = GpuBoard::new(&position);

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: gpu_board.as_byte_slice(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuBoard>() as u64),
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
    };

    surface.configure(&device, &config);
    window.request_redraw();

    let mut current_mouse_pos = None;
    let mut mouse_index = 0;
    let mut mouse_positions = [PhysicalPosition::default(); 2];
    let mut move_piece = false;

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { button, state, .. },
                ..
            } => {
                if button == MouseButton::Left {
                    if state == winit::event::ElementState::Released {
                        if let Some(mouse_pos) = current_mouse_pos {
                            mouse_positions[mouse_index] = mouse_pos;
                            move_piece = mouse_index == 1;
                            mouse_index = (mouse_index + 1) % 2;
                        }
                    }

                    if move_piece {
                        move_piece = false;
                        let window_size = window.inner_size();
                        if let (Some(from), Some(to)) = (
                            mouse_to_square(mouse_positions[0], window_size),
                            mouse_to_square(mouse_positions[1], window_size),
                        ) {
                            if try_player_move(&mut position, from, to) {
                                let _ = make_move(&mut position);
                                let gpu_board = GpuBoard::new(&position);
                                queue.write_buffer(&uniform_buf, 0, gpu_board.as_byte_slice());
                                window.request_redraw();
                            }
                        }
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                current_mouse_pos = Some(position);
            }
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.set_pipeline(&render_pipeline);
                    rpass.draw(0..6, 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::WindowBuilder::new()
        .with_title("Onion")
        .with_inner_size(winit::dpi::LogicalSize::new(800, 800))
        .build(&event_loop)
        .unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }

    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");

        use winit::platform::web::WindowExtWebSys;

        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
