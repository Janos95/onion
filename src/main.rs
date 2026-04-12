use bytemuck::{Pod, Zeroable};
use onion::engine::{try_player_move, Move, Position, Square, SEARCH_TIME_BUDGET_MS};
use std::borrow::Cow;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use wgpu::util::DeviceExt;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::{Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy},
    window::Window,
};

#[cfg(not(target_arch = "wasm32"))]
use onion::engine::Searcher;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{closure::Closure, JsCast, JsValue};

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
struct GpuState {
    pieces: [i32; 64],
    status: [f32; 4],
}

unsafe impl Zeroable for GpuState {}
unsafe impl Pod for GpuState {}

impl GpuState {
    fn new(position: &Position, engine_is_thinking: bool, animation_time_seconds: f32) -> GpuState {
        let mut board = GpuState {
            pieces: [0; 64],
            status: [
                if engine_is_thinking { 1.0 } else { 0.0 },
                animation_time_seconds,
                0.0,
                0.0,
            ],
        };
        for square in 0..64 {
            board.pieces[square] = position.piece_at(square as Square) as i32;
        }
        board
    }

    fn as_byte_slice(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn animation_time_seconds(start_time: Instant) -> f32 {
    start_time.elapsed().as_secs_f32()
}

#[cfg(target_arch = "wasm32")]
fn animation_time_seconds(start_time_ms: f64) -> f32 {
    ((js_sys::Date::now() - start_time_ms) / 1000.0) as f32
}

fn write_gpu_state(
    queue: &wgpu::Queue,
    uniform_buf: &wgpu::Buffer,
    position: &Position,
    engine_is_thinking: bool,
    animation_time_seconds: f32,
) {
    let gpu_state = GpuState::new(position, engine_is_thinking, animation_time_seconds);
    queue.write_buffer(uniform_buf, 0, gpu_state.as_byte_slice());
}

#[cfg(target_arch = "wasm32")]
fn sync_debug_state(position: &Position, engine_is_thinking: bool, request_id: u32) {
    let window = web_sys::window().expect("window should exist");
    let debug_state = js_sys::Object::new();
    let board = js_sys::Array::new();

    for square in 0..64 {
        board.push(&JsValue::from_f64(
            position.piece_at(square as Square) as i32 as f64,
        ));
    }

    js_sys::Reflect::set(&debug_state, &JsValue::from_str("board"), &board).unwrap();
    js_sys::Reflect::set(
        &debug_state,
        &JsValue::from_str("sideToMove"),
        &JsValue::from_f64(position.side_to_move() as u8 as f64),
    )
    .unwrap();
    js_sys::Reflect::set(
        &debug_state,
        &JsValue::from_str("castlingRights"),
        &JsValue::from_f64(position.castling_rights_bits() as f64),
    )
    .unwrap();
    js_sys::Reflect::set(
        &debug_state,
        &JsValue::from_str("enPassantSquare"),
        &JsValue::from_f64(
            position
                .en_passant_square()
                .map_or(-1.0, |square| square as f64),
        ),
    )
    .unwrap();
    js_sys::Reflect::set(
        &debug_state,
        &JsValue::from_str("engineThinking"),
        &JsValue::from_bool(engine_is_thinking),
    )
    .unwrap();
    js_sys::Reflect::set(
        &debug_state,
        &JsValue::from_str("requestId"),
        &JsValue::from_f64(request_id as f64),
    )
    .unwrap();
    js_sys::Reflect::set(
        window.as_ref(),
        &JsValue::from_str("__onionDebug"),
        &debug_state,
    )
    .unwrap();
}

#[derive(Debug, Copy, Clone)]
enum AppEvent {
    #[cfg(target_arch = "wasm32")]
    AnimationTick,
    EngineMoveReady {
        request_id: u32,
        best_move: Option<Move>,
    },
}

#[cfg(target_arch = "wasm32")]
struct AnimationLoop {
    active: Rc<Cell<bool>>,
    callback: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>,
}

#[cfg(target_arch = "wasm32")]
impl AnimationLoop {
    fn new(proxy: EventLoopProxy<AppEvent>) -> AnimationLoop {
        let active = Rc::new(Cell::new(false));
        let callback = Rc::new(RefCell::new(None::<Closure<dyn FnMut(f64)>>));
        let active_for_loop = Rc::clone(&active);
        let callback_for_loop = Rc::clone(&callback);

        *callback.borrow_mut() = Some(Closure::wrap(Box::new(move |_timestamp: f64| {
            let _ = proxy.send_event(AppEvent::AnimationTick);
            if active_for_loop.get() {
                if let Some(window) = web_sys::window() {
                    if let Some(callback) = callback_for_loop.borrow().as_ref() {
                        let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
                    }
                }
            }
        }) as Box<dyn FnMut(f64)>));

        AnimationLoop { active, callback }
    }

    fn start(&self) {
        if self.active.replace(true) {
            return;
        }

        if let Some(window) = web_sys::window() {
            if let Some(callback) = self.callback.borrow().as_ref() {
                let _ = window.request_animation_frame(callback.as_ref().unchecked_ref());
            }
        }
    }

    fn stop(&self) {
        self.active.set(false);
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct EngineDriver {
    proxy: EventLoopProxy<AppEvent>,
}

#[cfg(not(target_arch = "wasm32"))]
impl EngineDriver {
    fn new(proxy: EventLoopProxy<AppEvent>) -> EngineDriver {
        EngineDriver { proxy }
    }

    fn request_move(&self, request_id: u32, position: Position) {
        let proxy = self.proxy.clone();
        std::thread::spawn(move || {
            let mut searcher = Searcher::new();
            let best_move = searcher
                .best_move_with_time_budget(
                    &position,
                    Duration::from_millis(SEARCH_TIME_BUDGET_MS as u64),
                )
                .map(|(best_move, _)| best_move);
            let _ = proxy.send_event(AppEvent::EngineMoveReady {
                request_id,
                best_move,
            });
        });
    }
}

#[cfg(target_arch = "wasm32")]
struct EngineDriver {
    worker: web_sys::Worker,
    worker_url: String,
    _onmessage: Closure<dyn FnMut(web_sys::MessageEvent)>,
    _onerror: Closure<dyn FnMut(web_sys::Event)>,
    _onmessageerror: Closure<dyn FnMut(web_sys::Event)>,
}

#[cfg(target_arch = "wasm32")]
impl EngineDriver {
    fn new(proxy: EventLoopProxy<AppEvent>) -> EngineDriver {
        let module_url = worker_module_url().expect("could not resolve wasm module URL");
        let script = format!(
            r#"
import init, {{ search_best_move }} from {module_url:?};

let initialized;

async function ensureInit() {{
  if (!initialized) {{
    initialized = init();
  }}
  await initialized;
}}

self.onmessage = async (event) => {{
  const [requestId, sideToMove, castlingRights, enPassantSquare, timeBudgetMs, board] = event.data;
  try {{
    await ensureInit();
    const bestMove = search_best_move(
      Int32Array.from(board),
      sideToMove,
      castlingRights,
      enPassantSquare,
      timeBudgetMs,
    );
    self.postMessage([requestId, bestMove, null]);
  }} catch (error) {{
    console.error("engine worker error", error);
    self.postMessage([requestId, -1, String(error)]);
  }}
}};
"#
        );

        let script_parts = js_sys::Array::new();
        script_parts.push(&JsValue::from_str(&script));

        let mut blob_options = web_sys::BlobPropertyBag::new();
        blob_options.type_("text/javascript");
        let blob = web_sys::Blob::new_with_str_sequence_and_options(&script_parts, &blob_options)
            .expect("could not create worker blob");
        let worker_url =
            web_sys::Url::create_object_url_with_blob(&blob).expect("could not create worker URL");

        let mut worker_options = web_sys::WorkerOptions::new();
        worker_options.type_(web_sys::WorkerType::Module);
        let worker = web_sys::Worker::new_with_options(&worker_url, &worker_options)
            .expect("could not start engine worker");

        let proxy_for_messages = proxy.clone();
        let onmessage = Closure::wrap(Box::new(move |event: web_sys::MessageEvent| {
            let data = js_sys::Array::from(&event.data());
            let Some(request_id) = data.get(0).as_f64().map(|value| value as u32) else {
                return;
            };
            if let Some(error) = data.get(2).as_string() {
                web_sys::console::error_1(&JsValue::from_str(&format!(
                    "engine worker failed: {error}"
                )));
            }
            let best_move = data.get(1).as_f64().and_then(|value| {
                let value = value as i32;
                (value >= 0).then_some(value as Move)
            });
            let _ = proxy_for_messages.send_event(AppEvent::EngineMoveReady {
                request_id,
                best_move,
            });
        }) as Box<dyn FnMut(_)>);
        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        let onerror = Closure::wrap(Box::new(move |event: web_sys::Event| {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "engine worker onerror: {:?}",
                event.type_()
            )));
        }) as Box<dyn FnMut(_)>);
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        let onmessageerror = Closure::wrap(Box::new(move |event: web_sys::Event| {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "engine worker onmessageerror: {:?}",
                event.type_()
            )));
        }) as Box<dyn FnMut(_)>);
        worker.set_onmessageerror(Some(onmessageerror.as_ref().unchecked_ref()));

        EngineDriver {
            worker,
            worker_url,
            _onmessage: onmessage,
            _onerror: onerror,
            _onmessageerror: onmessageerror,
        }
    }

    fn request_move(&self, request_id: u32, position: Position) {
        let board = js_sys::Array::new();
        for square in 0..64 {
            board.push(&JsValue::from_f64(
                position.piece_at(square as Square) as i32 as f64,
            ));
        }

        let payload = js_sys::Array::new();
        payload.push(&JsValue::from_f64(request_id as f64));
        payload.push(&JsValue::from_f64(position.side_to_move() as u8 as f64));
        payload.push(&JsValue::from_f64(position.castling_rights_bits() as f64));
        payload.push(&JsValue::from_f64(
            position
                .en_passant_square()
                .map_or(-1.0, |square| square as f64),
        ));
        payload.push(&JsValue::from_f64(SEARCH_TIME_BUDGET_MS as f64));
        let board_js: JsValue = board.into();
        payload.push(&board_js);

        let payload_js: JsValue = payload.into();
        self.worker
            .post_message(&payload_js)
            .expect("could not send engine search request");
    }
}

#[cfg(target_arch = "wasm32")]
impl Drop for EngineDriver {
    fn drop(&mut self) {
        self.worker.set_onmessage(None);
        self.worker.terminate();
        let _ = web_sys::Url::revoke_object_url(&self.worker_url);
    }
}

#[cfg(target_arch = "wasm32")]
fn worker_module_url() -> Result<String, JsValue> {
    let window = web_sys::window().expect("window should exist");
    let base = window.location().href()?;
    Ok(web_sys::Url::new_with_base("engine_worker.js", &base)?.href())
}

async fn run(event_loop: EventLoop<AppEvent>, window: Window) {
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

    #[cfg(not(target_arch = "wasm32"))]
    let animation_start = Instant::now();
    #[cfg(target_arch = "wasm32")]
    let animation_start = js_sys::Date::now();

    let engine_driver = EngineDriver::new(event_loop.create_proxy());
    let mut position = Position::new();
    let gpu_state = GpuState::new(&position, false, 0.0);

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: gpu_state.as_byte_slice(),
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
                min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<GpuState>() as u64),
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

    #[cfg(target_arch = "wasm32")]
    let animation_loop = AnimationLoop::new(event_loop.create_proxy());

    let mut current_mouse_pos = None;
    let mut mouse_index = 0;
    let mut mouse_positions = [PhysicalPosition::default(); 2];
    let mut pending_move_selection = false;
    let mut engine_is_thinking = false;
    let mut thinking_started_time_seconds = 0.0f32;
    let mut active_request_id = 0u32;
    #[cfg(target_arch = "wasm32")]
    sync_debug_state(&position, engine_is_thinking, active_request_id);

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter, &shader, &pipeline_layout);
        #[cfg(target_arch = "wasm32")]
        let _ = &animation_loop;

        #[cfg(not(target_arch = "wasm32"))]
        {
            *control_flow = if engine_is_thinking {
                ControlFlow::Poll
            } else {
                ControlFlow::Wait
            };
        }
        #[cfg(target_arch = "wasm32")]
        {
            *control_flow = ControlFlow::Wait;
        }
        match event {
            #[cfg(target_arch = "wasm32")]
            Event::UserEvent(AppEvent::AnimationTick) => {
                if engine_is_thinking {
                    let thinking_elapsed =
                        animation_time_seconds(animation_start) - thinking_started_time_seconds;
                    write_gpu_state(
                        &queue,
                        &uniform_buf,
                        &position,
                        engine_is_thinking,
                        thinking_elapsed,
                    );
                    window.request_redraw();
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Event::MainEventsCleared => {
                if engine_is_thinking {
                    let thinking_elapsed =
                        animation_time_seconds(animation_start) - thinking_started_time_seconds;
                    write_gpu_state(
                        &queue,
                        &uniform_buf,
                        &position,
                        engine_is_thinking,
                        thinking_elapsed,
                    );
                    window.request_redraw();
                }
            }
            Event::UserEvent(AppEvent::EngineMoveReady {
                request_id,
                best_move,
            }) => {
                if request_id != active_request_id {
                    return;
                }

                engine_is_thinking = false;
                #[cfg(target_arch = "wasm32")]
                animation_loop.stop();
                if let Some(best_move) = best_move {
                    position.do_move(best_move);
                }
                write_gpu_state(
                    &queue,
                    &uniform_buf,
                    &position,
                    engine_is_thinking,
                    0.0,
                );
                window.request_redraw();
                #[cfg(target_arch = "wasm32")]
                sync_debug_state(&position, engine_is_thinking, active_request_id);
            }
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
                if button == MouseButton::Left && !engine_is_thinking {
                    if state == winit::event::ElementState::Released {
                        if let Some(mouse_pos) = current_mouse_pos {
                            mouse_positions[mouse_index] = mouse_pos;
                            pending_move_selection = mouse_index == 1;
                            mouse_index = (mouse_index + 1) % 2;
                        }
                    }

                    if pending_move_selection {
                        pending_move_selection = false;
                        let window_size = window.inner_size();
                        if let (Some(from), Some(to)) = (
                            mouse_to_square(mouse_positions[0], window_size),
                            mouse_to_square(mouse_positions[1], window_size),
                        ) {
                            if try_player_move(&mut position, from, to) {
                                engine_is_thinking = true;
                                // Reset the pulse phase so black pieces always start fully opaque.
                                thinking_started_time_seconds =
                                    animation_time_seconds(animation_start);
                                active_request_id = active_request_id.wrapping_add(1);
                                #[cfg(target_arch = "wasm32")]
                                animation_loop.start();
                                write_gpu_state(
                                    &queue,
                                    &uniform_buf,
                                    &position,
                                    engine_is_thinking,
                                    0.0,
                                );
                                window.request_redraw();
                                #[cfg(target_arch = "wasm32")]
                                sync_debug_state(&position, engine_is_thinking, active_request_id);
                                engine_driver.request_move(active_request_id, position);
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
    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event().build();
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
