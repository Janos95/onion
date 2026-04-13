use bytemuck::{Pod, Zeroable};
use onion::engine::{
    is_selectable_piece, legal_move_destinations, player_move, Move, Piece, Position, Square,
    SEARCH_TIME_BUDGET_MS,
};
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

fn square_center(square: Square) -> [f32; 2] {
    [
        (square % 8) as f32 + 0.5,
        (square / 8) as f32 + 0.5,
    ]
}

#[derive(Debug, Copy, Clone)]
struct MovingPieceVisual {
    origin: Square,
    piece: Piece,
    board_position: [f32; 2],
    target: Square,
    captured_piece: Piece,
    capture_progress: f32,
}

#[derive(Debug, Copy, Clone)]
struct MoveAnimation {
    from: Square,
    to: Square,
    piece: Piece,
    captured_piece: Piece,
    move_to_apply: Move,
    started_time_seconds: f32,
    start_engine_after: bool,
}

const MOVE_ANIMATION_DURATION_SECONDS: f32 = 0.63;
const MOVING_PIECE_SMOOTH_UNION_RADIUS: f32 = 0.35;

fn eased_progress(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn move_animation_visual(
    animation: MoveAnimation,
    now_seconds: f32,
) -> MovingPieceVisual {
    let progress = eased_progress(
        (now_seconds - animation.started_time_seconds) / MOVE_ANIMATION_DURATION_SECONDS,
    );
    let from = square_center(animation.from);
    let to = square_center(animation.to);
    let dx = to[0] - from[0];
    let dy = to[1] - from[1];
    let is_knight = matches!(animation.piece, Piece::WhiteKnight | Piece::BlackKnight);
    let board_position = if is_knight {
        let (corner, first_leg_fraction) = if dx.abs() > dy.abs() {
            ([to[0], from[1]], 2.0 / 3.0)
        } else {
            ([from[0], to[1]], 2.0 / 3.0)
        };

        if progress < first_leg_fraction {
            let local_t = progress / first_leg_fraction;
            [
                from[0] + (corner[0] - from[0]) * local_t,
                from[1] + (corner[1] - from[1]) * local_t,
            ]
        } else {
            let local_t = (progress - first_leg_fraction) / (1.0 - first_leg_fraction);
            [
                corner[0] + (to[0] - corner[0]) * local_t,
                corner[1] + (to[1] - corner[1]) * local_t,
            ]
        }
    } else {
        [
            from[0] + dx * progress,
            from[1] + dy * progress,
        ]
    };
    let capture_progress = if animation.captured_piece != Piece::Empty {
        eased_progress(((progress - 0.6) / 0.4).clamp(0.0, 1.0))
    } else {
        0.0
    };
    MovingPieceVisual {
        origin: animation.from,
        piece: animation.piece,
        board_position,
        target: animation.to,
        captured_piece: animation.captured_piece,
        capture_progress,
    }
}

fn move_origin_square(m: Move) -> Square {
    (m >> 6) & 0x3F
}

fn move_destination_square(m: Move) -> Square {
    m & 0x3F
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct GpuState {
    pieces: [i32; 64],
    markers: [i32; 64],
    status: [f32; 4],
    moving_piece_state: [f32; 4],
    moving_piece: [i32; 4],
}

unsafe impl Zeroable for GpuState {}
unsafe impl Pod for GpuState {}

impl GpuState {
    fn new(
        position: &Position,
        selected_square: Option<Square>,
        legal_move_destinations: u64,
        moving_piece: Option<MovingPieceVisual>,
        engine_is_thinking: bool,
        animation_time_seconds: f32,
    ) -> GpuState {
        let mut board = GpuState {
            pieces: [0; 64],
            markers: [0; 64],
            status: [
                if engine_is_thinking { 1.0 } else { 0.0 },
                animation_time_seconds,
                0.0,
                0.0,
            ],
            moving_piece_state: [0.0, 0.0, 0.0, MOVING_PIECE_SMOOTH_UNION_RADIUS],
            moving_piece: [0; 4],
        };
        for square in 0..64 {
            board.pieces[square] = position.piece_at(square as Square) as i32;
            if (legal_move_destinations & (1u64 << square)) != 0 {
                board.markers[square] = 1;
            }
        }

        if let Some(square) = selected_square {
            board.markers[square as usize] = 2;
        }

        if let Some(moving_piece) = moving_piece {
            board.pieces[moving_piece.origin as usize] = Piece::Empty as i32;
            board.moving_piece_state = [
                1.0,
                moving_piece.board_position[0],
                moving_piece.board_position[1],
                moving_piece.capture_progress,
            ];
            board.moving_piece = [
                moving_piece.piece as i32,
                moving_piece.target as i32,
                moving_piece.captured_piece as i32,
                0,
            ];
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
    selected_square: Option<Square>,
    legal_move_destinations: u64,
    moving_piece: Option<MovingPieceVisual>,
    engine_is_thinking: bool,
    animation_time_seconds: f32,
) {
    let gpu_state = GpuState::new(
        position,
        selected_square,
        legal_move_destinations,
        moving_piece,
        engine_is_thinking,
        animation_time_seconds,
    );
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
    let gpu_state = GpuState::new(&position, None, 0, None, false, 0.0);

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
    let mut selected_square = None;
    let mut selected_moves_mask = 0u64;
    let mut move_animation: Option<MoveAnimation> = None;
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
            *control_flow = if engine_is_thinking || move_animation.is_some() {
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
                let now_seconds = animation_time_seconds(animation_start);
                let mut completed_animation = false;
                if let Some(active_animation) = move_animation {
                    if now_seconds - active_animation.started_time_seconds
                        >= MOVE_ANIMATION_DURATION_SECONDS
                    {
                        move_animation = None;
                        position.do_move(active_animation.move_to_apply);
                        completed_animation = true;
                        if active_animation.start_engine_after {
                            engine_is_thinking = true;
                            thinking_started_time_seconds = now_seconds;
                            active_request_id = active_request_id.wrapping_add(1);
                            #[cfg(target_arch = "wasm32")]
                            sync_debug_state(&position, engine_is_thinking, active_request_id);
                            engine_driver.request_move(active_request_id, position);
                        } else {
                            #[cfg(target_arch = "wasm32")]
                            sync_debug_state(&position, engine_is_thinking, active_request_id);
                        }
                    }
                }

                if engine_is_thinking || move_animation.is_some() || completed_animation {
                    let thinking_elapsed = if engine_is_thinking {
                        now_seconds - thinking_started_time_seconds
                    } else {
                        0.0
                    };
                    write_gpu_state(
                        &queue,
                        &uniform_buf,
                        &position,
                        selected_square,
                        selected_moves_mask,
                        move_animation.map(|animation| move_animation_visual(animation, now_seconds)),
                        engine_is_thinking,
                        thinking_elapsed,
                    );
                    window.request_redraw();
                } else {
                    animation_loop.stop();
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            Event::MainEventsCleared => {
                let now_seconds = animation_time_seconds(animation_start);
                let mut completed_animation = false;
                if let Some(active_animation) = move_animation {
                    if now_seconds - active_animation.started_time_seconds
                        >= MOVE_ANIMATION_DURATION_SECONDS
                    {
                        move_animation = None;
                        position.do_move(active_animation.move_to_apply);
                        completed_animation = true;
                        if active_animation.start_engine_after {
                            engine_is_thinking = true;
                            thinking_started_time_seconds = now_seconds;
                            active_request_id = active_request_id.wrapping_add(1);
                            engine_driver.request_move(active_request_id, position);
                        }
                    }
                }

                if engine_is_thinking || move_animation.is_some() || completed_animation {
                    let thinking_elapsed = if engine_is_thinking {
                        now_seconds - thinking_started_time_seconds
                    } else {
                        0.0
                    };
                    write_gpu_state(
                        &queue,
                        &uniform_buf,
                        &position,
                        selected_square,
                        selected_moves_mask,
                        move_animation.map(|animation| move_animation_visual(animation, now_seconds)),
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
                selected_square = None;
                selected_moves_mask = 0;
                if let Some(best_move) = best_move {
                    let now_seconds = animation_time_seconds(animation_start);
                    let origin = move_origin_square(best_move);
                    move_animation = Some(MoveAnimation {
                        from: origin,
                        to: move_destination_square(best_move),
                        piece: position.piece_at(origin),
                        captured_piece: position.piece_at(move_destination_square(best_move)),
                        move_to_apply: best_move,
                        started_time_seconds: now_seconds,
                        start_engine_after: false,
                    });
                } else {
                    move_animation = None;
                    #[cfg(target_arch = "wasm32")]
                    animation_loop.stop();
                    #[cfg(target_arch = "wasm32")]
                    sync_debug_state(&position, engine_is_thinking, active_request_id);
                }
                let now_seconds = animation_time_seconds(animation_start);
                write_gpu_state(
                    &queue,
                    &uniform_buf,
                    &position,
                    selected_square,
                    selected_moves_mask,
                    move_animation.map(|animation| move_animation_visual(animation, now_seconds)),
                    engine_is_thinking,
                    0.0,
                );
                window.request_redraw();
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
                if button != MouseButton::Left
                    || state != winit::event::ElementState::Released
                    || engine_is_thinking
                    || move_animation.is_some()
                {
                    return;
                }

                let window_size = window.inner_size();
                let clicked_square =
                    current_mouse_pos.and_then(|mouse_pos| mouse_to_square(mouse_pos, window_size));

                if let Some(clicked_square) = clicked_square {
                    if selected_square == Some(clicked_square) {
                        selected_square = None;
                        selected_moves_mask = 0;
                    } else if is_selectable_piece(&position, clicked_square) {
                        selected_square = Some(clicked_square);
                        selected_moves_mask = legal_move_destinations(&position, clicked_square);
                    } else if let Some(from) = selected_square {
                        if (selected_moves_mask & (1u64 << clicked_square)) != 0 {
                            if let Some(move_to_apply) = player_move(&position, from, clicked_square)
                            {
                                let now_seconds = animation_time_seconds(animation_start);
                                move_animation = Some(MoveAnimation {
                                    from,
                                    to: clicked_square,
                                    piece: position.piece_at(from),
                                    captured_piece: position.piece_at(clicked_square),
                                    move_to_apply,
                                    started_time_seconds: now_seconds,
                                    start_engine_after: true,
                                });
                                selected_square = None;
                                selected_moves_mask = 0;
                                #[cfg(target_arch = "wasm32")]
                                animation_loop.start();
                                write_gpu_state(
                                    &queue,
                                    &uniform_buf,
                                    &position,
                                    selected_square,
                                    selected_moves_mask,
                                    move_animation
                                        .map(|animation| move_animation_visual(animation, now_seconds)),
                                    engine_is_thinking,
                                    0.0,
                                );
                                window.request_redraw();
                                return;
                            }
                        }
                    }
                }

                write_gpu_state(
                    &queue,
                    &uniform_buf,
                    &position,
                    selected_square,
                    selected_moves_mask,
                    None,
                    engine_is_thinking,
                    0.0,
                );
                window.request_redraw();
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
