#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("capture_lab is native-only");
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use bytemuck::{Pod, Zeroable};
    use onion::engine::{Piece, Square};
    use std::{
        borrow::Cow,
        fs,
        path::{Path, PathBuf},
        time::{Duration, Instant, SystemTime},
    };
    use wgpu::util::DeviceExt;
    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::Window,
    };

    const MOVE_ANIMATION_DURATION_SECONDS: f32 = 1.0;
    const CAPTURE_CONTACT_TIME_SECONDS: f32 = 0.2;
    const SCENARIO_LEAD_IN_SECONDS: f32 = 0.35;
    const SCENARIO_HOLD_OUT_SECONDS: f32 = 0.9;
    const SCENARIO_DURATION_SECONDS: f32 =
        SCENARIO_LEAD_IN_SECONDS + MOVE_ANIMATION_DURATION_SECONDS + SCENARIO_HOLD_OUT_SECONDS;
    const MOVING_PIECE_SMOOTH_UNION_RADIUS: f32 = 0.5;
    const SHADER_RELOAD_POLL_INTERVAL: Duration = Duration::from_millis(150);

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

    #[derive(Debug, Copy, Clone)]
    struct MovingPieceVisual {
        origin: Square,
        piece: Piece,
        board_position: [f32; 2],
        target: Square,
        captured_piece: Piece,
    }

    #[derive(Debug, Copy, Clone)]
    struct PiecePlacement {
        square: Square,
        piece: Piece,
    }

    #[derive(Debug, Clone)]
    struct LabScenario {
        name: String,
        from: Square,
        to: Square,
        moving_piece: Piece,
        target_piece: Piece,
        extras: Vec<PiecePlacement>,
    }

    fn scenarios_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("capture_lab_moves.txt")
    }

    fn parse_square_name(name: &str) -> Result<Square, String> {
        let bytes = name.trim().as_bytes();
        if bytes.len() != 2 {
            return Err(format!("invalid square '{name}'"));
        }
        let file = bytes[0].to_ascii_lowercase();
        let rank = bytes[1];
        if !(b'a'..=b'h').contains(&file) || !(b'1'..=b'8').contains(&rank) {
            return Err(format!("invalid square '{name}'"));
        }
        Ok((file - b'a') as Square + (rank - b'1') as Square * 8)
    }

    fn parse_piece_name(name: &str) -> Result<Piece, String> {
        match name.trim() {
            "." | "-" | "_" | "Empty" | "empty" => Ok(Piece::Empty),
            "P" | "WP" | "WhitePawn" => Ok(Piece::WhitePawn),
            "p" | "BP" | "BlackPawn" => Ok(Piece::BlackPawn),
            "N" | "WN" | "WhiteKnight" => Ok(Piece::WhiteKnight),
            "n" | "BN" | "BlackKnight" => Ok(Piece::BlackKnight),
            "B" | "WB" | "WhiteBishop" => Ok(Piece::WhiteBishop),
            "b" | "BB" | "BlackBishop" => Ok(Piece::BlackBishop),
            "R" | "WR" | "WhiteRook" => Ok(Piece::WhiteRook),
            "r" | "BR" | "BlackRook" => Ok(Piece::BlackRook),
            "Q" | "WQ" | "WhiteQueen" => Ok(Piece::WhiteQueen),
            "q" | "BQ" | "BlackQueen" => Ok(Piece::BlackQueen),
            "K" | "WK" | "WhiteKing" => Ok(Piece::WhiteKing),
            "k" | "BK" | "BlackKing" => Ok(Piece::BlackKing),
            _ => Err(format!("invalid piece '{name}'")),
        }
    }

    fn parse_extra_placements(spec: &str) -> Result<Vec<PiecePlacement>, String> {
        let trimmed = spec.trim();
        if trimmed.is_empty() || trimmed == "-" {
            return Ok(Vec::new());
        }

        trimmed
            .split(',')
            .map(|entry| {
                let (square_name, piece_name) = entry
                    .split_once(':')
                    .ok_or_else(|| format!("invalid extra placement '{entry}'"))?;
                Ok(PiecePlacement {
                    square: parse_square_name(square_name)?,
                    piece: parse_piece_name(piece_name)?,
                })
            })
            .collect()
    }

    fn load_scenarios(path: &Path) -> Result<(Vec<LabScenario>, SystemTime), String> {
        let source =
            fs::read_to_string(path).map_err(|error| format!("failed to read scenarios: {error}"))?;
        let metadata =
            fs::metadata(path).map_err(|error| format!("failed to stat scenarios: {error}"))?;
        let modified = metadata
            .modified()
            .map_err(|error| format!("failed to read scenario mtime: {error}"))?;

        let mut scenarios = Vec::new();
        for (line_index, raw_line) in source.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<_> = line.split('|').map(str::trim).collect();
            if parts.len() < 5 || parts.len() > 6 {
                return Err(format!(
                    "scenario line {} must have 5 or 6 '|' fields",
                    line_index + 1
                ));
            }

            let extras = if parts.len() == 6 {
                parse_extra_placements(parts[5])?
            } else {
                Vec::new()
            };

            scenarios.push(LabScenario {
                name: parts[0].to_string(),
                moving_piece: parse_piece_name(parts[1])?,
                from: parse_square_name(parts[2])?,
                to: parse_square_name(parts[3])?,
                target_piece: parse_piece_name(parts[4])?,
                extras,
            });
        }

        if scenarios.is_empty() {
            return Err("scenario file did not contain any scenarios".to_string());
        }

        Ok((scenarios, modified))
    }

    impl GpuState {
        fn from_scenario(
            scenario: &LabScenario,
            moving_piece: Option<MovingPieceVisual>,
            animation_time_seconds: f32,
        ) -> GpuState {
            let mut board = GpuState {
                pieces: [0; 64],
                markers: [0; 64],
                status: [0.0, animation_time_seconds, 0.0, 0.0],
                moving_piece_state: [0.0, 0.0, 0.0, MOVING_PIECE_SMOOTH_UNION_RADIUS],
                moving_piece: [0; 4],
            };

            for placement in &scenario.extras {
                if placement.piece != Piece::Empty {
                    board.pieces[placement.square as usize] = placement.piece as i32;
                }
            }
            board.pieces[scenario.from as usize] = scenario.moving_piece as i32;
            board.pieces[scenario.to as usize] = scenario.target_piece as i32;

            if let Some(moving_piece) = moving_piece {
                board.pieces[moving_piece.origin as usize] = Piece::Empty as i32;
                board.moving_piece_state = [
                    1.0,
                    moving_piece.board_position[0],
                    moving_piece.board_position[1],
                    MOVING_PIECE_SMOOTH_UNION_RADIUS,
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

    fn hermite_segment(x0: f32, y0: f32, m0: f32, x1: f32, y1: f32, m1: f32, x: f32) -> f32 {
        let h = x1 - x0;
        let t = ((x - x0) / h).clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * h * m0
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * h * m1
    }

    fn smooth_capture_path_progress(
        raw_progress: f32,
        contact_time_fraction: f32,
        contact_path_progress: f32,
    ) -> f32 {
        let x = raw_progress.clamp(0.0, 1.0);
        let x1 = contact_time_fraction.clamp(0.001, 0.999);
        let y1 = contact_path_progress.clamp(0.0, 1.0);

        let h0 = x1;
        let h1 = 1.0 - x1;
        let d0 = y1 / h0;
        let d1 = (1.0 - y1) / h1;

        let mut m0 = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
        if m0.signum() != d0.signum() {
            m0 = 0.0;
        } else if m0.abs() > 3.0 * d0.abs() {
            m0 = 3.0 * d0;
        }

        let m1 = if d0 > 0.0 && d1 > 0.0 {
            let w0 = 2.0 * h1 + h0;
            let w1 = h1 + 2.0 * h0;
            (w0 + w1) / (w0 / d0 + w1 / d1)
        } else {
            0.0
        };

        let mut m2 = ((2.0 * h1 + h0) * d1 - h1 * d0) / (h0 + h1);
        if m2.signum() != d1.signum() {
            m2 = 0.0;
        } else if m2.abs() > 3.0 * d1.abs() {
            m2 = 3.0 * d1;
        }

        if x <= x1 {
            hermite_segment(0.0, 0.0, m0, x1, y1, m1, x)
        } else {
            hermite_segment(x1, y1, m1, 1.0, 1.0, m2, x)
        }
    }

    fn square_center(square: Square) -> [f32; 2] {
        [(square % 8) as f32 + 0.5, (square / 8) as f32 + 0.5]
    }

    fn move_path_position(scenario: &LabScenario, progress: f32) -> [f32; 2] {
        let from = square_center(scenario.from);
        let to = square_center(scenario.to);
        let dx = to[0] - from[0];
        let dy = to[1] - from[1];
        let is_knight = matches!(scenario.moving_piece, Piece::WhiteKnight | Piece::BlackKnight);
        if is_knight {
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
            [from[0] + dx * progress, from[1] + dy * progress]
        }
    }

    fn square_boxes_touch(position: [f32; 2], target_center: [f32; 2]) -> bool {
        (position[0] - target_center[0]).abs() <= 1.0
            && (position[1] - target_center[1]).abs() <= 1.0
    }

    fn capture_contact_path_progress(scenario: &LabScenario) -> f32 {
        let target_center = square_center(scenario.to);
        let mut low = 0.0f32;
        let mut high = 1.0f32;
        for _ in 0..24 {
            let mid = 0.5 * (low + high);
            if square_boxes_touch(move_path_position(scenario, mid), target_center) {
                high = mid;
            } else {
                low = mid;
            }
        }
        high
    }

    fn scenario_moving_piece(
        scenario: &LabScenario,
        elapsed_seconds: f32,
    ) -> Option<MovingPieceVisual> {
        if elapsed_seconds < SCENARIO_LEAD_IN_SECONDS {
            return None;
        }

        let local_animation_seconds = elapsed_seconds - SCENARIO_LEAD_IN_SECONDS;
        if local_animation_seconds >= MOVE_ANIMATION_DURATION_SECONDS {
            return None;
        }

        let raw_progress =
            (local_animation_seconds / MOVE_ANIMATION_DURATION_SECONDS).clamp(0.0, 1.0);
        let path_progress = if scenario.target_piece != Piece::Empty {
            let contact_path_progress = capture_contact_path_progress(scenario);
            let contact_time_fraction =
                CAPTURE_CONTACT_TIME_SECONDS / MOVE_ANIMATION_DURATION_SECONDS;
            smooth_capture_path_progress(
                raw_progress,
                contact_time_fraction,
                contact_path_progress,
            )
        } else {
            let t = raw_progress.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };

        Some(MovingPieceVisual {
            origin: scenario.from,
            piece: scenario.moving_piece,
            board_position: move_path_position(scenario, path_progress),
            target: scenario.to,
            captured_piece: scenario.target_piece,
        })
    }

    fn scenario_gpu_state(scenario: LabScenario, elapsed_seconds: f32) -> GpuState {
        let moving_piece = scenario_moving_piece(&scenario, elapsed_seconds);
        let board = if elapsed_seconds
            >= SCENARIO_LEAD_IN_SECONDS + MOVE_ANIMATION_DURATION_SECONDS
        {
            let mut board = GpuState::from_scenario(&scenario, None, elapsed_seconds);
            board.pieces[scenario.from as usize] = Piece::Empty as i32;
            board.pieces[scenario.to as usize] = scenario.moving_piece as i32;
            board
        } else {
            GpuState::from_scenario(&scenario, moving_piece, elapsed_seconds)
        };
        board
    }

    struct HotReloadPipeline {
        render_pipeline: wgpu::RenderPipeline,
        shader_mtime: SystemTime,
        last_poll: Instant,
    }

    struct HotReloadScenarios {
        scenarios: Vec<LabScenario>,
        scenarios_mtime: SystemTime,
        last_poll: Instant,
    }

    fn shader_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/shader.wgsl")
    }

    fn load_render_pipeline(
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        path: &Path,
    ) -> Result<(wgpu::RenderPipeline, SystemTime), String> {
        let source =
            fs::read_to_string(path).map_err(|error| format!("failed to read shader: {error}"))?;
        let metadata =
            fs::metadata(path).map_err(|error| format!("failed to stat shader: {error}"))?;
        let modified = metadata
            .modified()
            .map_err(|error| format!("failed to read shader mtime: {error}"))?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("capture_lab_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(source)),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("capture_lab_pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        device.poll(wgpu::Maintain::Wait);

        if let Some(error) = pollster::block_on(device.pop_error_scope()) {
            return Err(format!("{error}"));
        }

        Ok((pipeline, modified))
    }

    fn maybe_reload_pipeline(
        hot_pipeline: &mut HotReloadPipeline,
        device: &wgpu::Device,
        pipeline_layout: &wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
        path: &Path,
    ) {
        if hot_pipeline.last_poll.elapsed() < SHADER_RELOAD_POLL_INTERVAL {
            return;
        }
        hot_pipeline.last_poll = Instant::now();

        let Ok(metadata) = fs::metadata(path) else {
            return;
        };
        let Ok(modified) = metadata.modified() else {
            return;
        };
        if modified <= hot_pipeline.shader_mtime {
            return;
        }

        match load_render_pipeline(device, pipeline_layout, format, path) {
            Ok((render_pipeline, shader_mtime)) => {
                hot_pipeline.render_pipeline = render_pipeline;
                hot_pipeline.shader_mtime = shader_mtime;
                eprintln!("reloaded shader: {}", path.display());
            }
            Err(error) => {
                eprintln!("shader reload failed: {error}");
            }
        }
    }

    fn maybe_reload_scenarios(hot_scenarios: &mut HotReloadScenarios, path: &Path) {
        if hot_scenarios.last_poll.elapsed() < SHADER_RELOAD_POLL_INTERVAL {
            return;
        }
        hot_scenarios.last_poll = Instant::now();

        let Ok(metadata) = fs::metadata(path) else {
            return;
        };
        let Ok(modified) = metadata.modified() else {
            return;
        };
        if modified <= hot_scenarios.scenarios_mtime {
            return;
        }

        match load_scenarios(path) {
            Ok((scenarios, scenarios_mtime)) => {
                hot_scenarios.scenarios = scenarios;
                hot_scenarios.scenarios_mtime = scenarios_mtime;
                eprintln!("reloaded scenarios: {}", path.display());
            }
            Err(error) => {
                eprintln!("scenario reload failed: {error}");
            }
        }
    }

    fn configure_surface(
        surface: &wgpu::Surface,
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
        width: u32,
        height: u32,
    ) -> (wgpu::SurfaceConfiguration, wgpu::TextureFormat) {
        let capabilities = surface.get_capabilities(adapter);
        let format = capabilities.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(device, &config);
        (config, format)
    }

    async fn run(event_loop: EventLoop<()>, window: Window) {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("failed to find a suitable adapter");

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
            .expect("failed to create device");

        let (mut config, surface_format) =
            configure_surface(&surface, &device, &adapter, size.width, size.height);

        let scenarios_path = scenarios_path();
        let (scenarios, scenarios_mtime) =
            load_scenarios(&scenarios_path).expect("failed to load initial scenarios");
        let mut hot_scenarios = HotReloadScenarios {
            scenarios,
            scenarios_mtime,
            last_poll: Instant::now(),
        };

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("capture_lab_uniforms"),
            contents: GpuState::from_scenario(&hot_scenarios.scenarios[0], None, 0.0).as_byte_slice(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("capture_lab_bind_group_layout"),
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
            label: Some("capture_lab_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("capture_lab_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let path = shader_path();
        let (render_pipeline, shader_mtime) =
            load_render_pipeline(&device, &pipeline_layout, surface_format, &path)
                .expect("failed to build initial shader");
        let mut hot_pipeline = HotReloadPipeline {
            render_pipeline,
            shader_mtime,
            last_poll: Instant::now(),
        };

        let start_time = Instant::now();
        let mut current_scenario_index = usize::MAX;
        window.request_redraw();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::MainEventsCleared => {
                    maybe_reload_pipeline(
                        &mut hot_pipeline,
                        &device,
                        &pipeline_layout,
                        surface_format,
                        &path,
                    );
                    maybe_reload_scenarios(&mut hot_scenarios, &scenarios_path);

                    let elapsed_seconds = start_time.elapsed().as_secs_f32();
                    let scenario_index =
                        ((elapsed_seconds / SCENARIO_DURATION_SECONDS) as usize)
                            % hot_scenarios.scenarios.len();
                    if scenario_index != current_scenario_index {
                        current_scenario_index = scenario_index;
                        window.set_title(&format!(
                            "Onion Capture Lab - {}",
                            hot_scenarios.scenarios[scenario_index].name
                        ));
                    }

                    let scenario_elapsed = elapsed_seconds % SCENARIO_DURATION_SECONDS;
                    let gpu_state = scenario_gpu_state(
                        hot_scenarios.scenarios[scenario_index].clone(),
                        scenario_elapsed,
                    );
                    queue.write_buffer(&uniform_buf, 0, gpu_state.as_byte_slice());
                    window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    let frame = match surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            surface.configure(&device, &config);
                            return;
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            *control_flow = ControlFlow::Exit;
                            return;
                        }
                        Err(wgpu::SurfaceError::Timeout) => {
                            return;
                        }
                    };
                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("capture_lab_encoder"),
                        });

                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("capture_lab_pass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                    store: true,
                                },
                            })],
                            depth_stencil_attachment: None,
                        });
                        rpass.set_bind_group(0, &bind_group, &[]);
                        rpass.set_pipeline(&hot_pipeline.render_pipeline);
                        rpass.draw(0..6, 0..1);
                    }

                    queue.submit(Some(encoder.finish()));
                    frame.present();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    if size.width == 0 || size.height == 0 {
                        return;
                    }
                    (config, _) =
                        configure_surface(&surface, &device, &adapter, size.width, size.height);
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                _ => {}
            }
        });
    }

    pub fn main() {
        env_logger::init();
        let event_loop = EventLoop::new();
        let window = winit::window::WindowBuilder::new()
            .with_title("Onion Capture Lab")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 800.0))
            .build(&event_loop)
            .unwrap();
        pollster::block_on(run(event_loop, window));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main();
}
