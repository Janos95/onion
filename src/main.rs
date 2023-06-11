use std::{borrow::Cow, cmp::max, cmp::min, iter::Iterator};
use winit::{
    event::{Event, WindowEvent, MouseButton},
    event_loop::{ControlFlow, EventLoop},
    window::Window, dpi::PhysicalPosition,
};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

const INITIAL_PIECES: [Piece; 64] = [
    Piece::WhiteRook, Piece::WhiteKnight, Piece::WhiteBishop, Piece::WhiteQueen, Piece::WhiteKing, Piece::WhiteBishop, Piece::WhiteKnight, Piece::WhiteRook, 
    Piece::WhitePawn, Piece::WhitePawn, Piece::WhitePawn, Piece::WhitePawn, Piece::WhitePawn, Piece::WhitePawn, Piece::WhitePawn, Piece::WhitePawn,
    Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty,
    Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty,
    Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty,
    Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty, Piece::Empty,
    Piece::BlackPawn, Piece::BlackPawn, Piece::BlackPawn, Piece::BlackPawn, Piece::BlackPawn, Piece::BlackPawn, Piece::BlackPawn, Piece::BlackPawn,
    Piece::BlackRook, Piece::BlackKnight, Piece::BlackBishop, Piece::BlackQueen, Piece::BlackKing, Piece::BlackBishop, Piece::BlackKnight, Piece::BlackRook,
];

const FILE_MASKS: [u64; 8] = [
    0x0101010101010101, /* File A */ 
    0x0202020202020202, /* File B */ 
    0x0404040404040404, /* File C */ 
    0x0808080808080808, /* File D */ 
    0x1010101010101010, /* File E */ 
    0x2020202020202020, /* File F */ 
    0x4040404040404040, /* File G */ 
    0x8080808080808080, /* File H */ 
];

const RANK_MASKS: [u64; 8] = [
    0x00000000000000FF, // Rank 1
    0x000000000000FF00, // Rank 2
    0x0000000000FF0000, // Rank 3
    0x00000000FF000000, // Rank 4
    0x000000FF00000000, // Rank 5
    0x0000FF0000000000, // Rank 6
    0x00FF000000000000, // Rank 7
    0xFF00000000000000, // Rank 8
];

const DOUBLE_PUSH_MASK: [u64; 2] = [
    0x0000000000FF0000, // rank 3
    0x0000FF0000000000, // rank 6
];

// chess colors
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Color {
    White,
    Black,
}

impl Color {
    fn opposite(&self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
enum PieceKind {
    Empty,
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    fn from_u8(p : u8) -> PieceKind {
        match p {
            0 => PieceKind::Empty,
            1 => PieceKind::Pawn,
            2 => PieceKind::Knight,
            3 => PieceKind::Bishop,
            4 => PieceKind::Rook,
            5 => PieceKind::Queen,
            6 => PieceKind::King,
            _ => panic!("Invalid piece kind: {}", p),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Piece {
    Empty,
    WhitePawn,
    BlackPawn,
    WhiteKnight,
    BlackKnight,
    WhiteBishop,
    BlackBishop,
    WhiteRook,
    BlackRook,
    WhiteQueen,
    BlackQueen,
    WhiteKing,
    BlackKing,
}

enum MoveType {
    Normal,
    Promotion,
    EnPassant,
    Castling,
}

impl Piece {
    fn kind(&self) -> PieceKind {
        match *self {
            Piece::Empty => PieceKind::Empty,
            Piece::WhitePawn | Piece::BlackPawn => PieceKind::Pawn,
            Piece::WhiteKnight | Piece::BlackKnight => PieceKind::Knight,
            Piece::WhiteBishop | Piece::BlackBishop => PieceKind::Bishop,
            Piece::WhiteRook | Piece::BlackRook => PieceKind::Rook,
            Piece::WhiteQueen | Piece::BlackQueen => PieceKind::Queen,
            Piece::WhiteKing | Piece::BlackKing => PieceKind::King,
        }
    }

    fn color(&self) -> Color {
        let is_odd = *self as u8 & 1;
        if is_odd == 0 { Color::Black } else { Color::White }
    }
}

type Bitboard = u64;
type Square = u32;

/// bit  0-5: destination square (from 0 to 63)
/// bit  6-11: origin square (from 0 to 63)
/// bit 12-13: promotion piece type - 2 (from KNIGHT-2 to QUEEN-2)
/// bit 14-15: special move flag: promotion (1), en passant (2), castling (3)
/// NOTE: en passant bit is set only when a pawn can be captured
type Move = u32;

fn destination_square(m : Move) -> Square {
    return m & 0x3F;
}

fn origin_square(m : Move) -> Square {
    return (m >> 6) & 0x3F;
}

#[derive(Copy, Clone)]
struct Position {
    positions: [Bitboard; 7], // empty, pawn, knight, bishop, rook, queen, king
    colors: [Bitboard; 2], // white, black
    pieces: [Piece; 64],
    side_to_move : Color,
}

fn set_square(bitboard : &mut Bitboard, square : Square) {
    *bitboard |= 1 << square;
}

fn unset_square(bitboard : &mut Bitboard, square : Square) {
    *bitboard &= !(1 << square);
}

impl Position {
    fn new() -> Position {
        let mut position = Position {
            positions: [0; 7],
            colors: [0; 2],
            pieces: INITIAL_PIECES,
            side_to_move: Color::White,
        };

        for i in 0..64 {
            let square = i as Square;
            let piece = INITIAL_PIECES[i];
            let pos = &mut position.positions[piece.kind() as usize];
            set_square(pos, square);
            if piece != Piece::Empty {
                set_square(&mut position.colors[piece.color() as usize], square);
            }
        }

        println!("white pieces {}", position.colors[0]);
        println!("black pieces {}", position.colors[1]);

        position
    }

    fn do_move(&mut self, m : Move) {
        let origin = origin_square(m);
        let destination = destination_square(m);

        //println!("doing move {} -> {}", origin, destination);

        let us = self.side_to_move as usize;
        let them = self.side_to_move.opposite() as usize;

        // color bitboards
        unset_square(&mut self.colors[them], destination);
        unset_square(&mut self.colors[us], origin);
        set_square(&mut self.colors[us], destination);

        // set origin to empty and 
        set_square(&mut self.positions[PieceKind::Empty as usize], origin);

        let moving_piece_kind = self.pieces[origin as usize].kind() as usize;
        let target_piece_kind = self.pieces[destination as usize].kind() as usize;

        unset_square(&mut self.positions[moving_piece_kind], origin);
        unset_square(&mut self.positions[target_piece_kind], destination);
        set_square(&mut self.positions[moving_piece_kind], destination);

        // explicit piece array
        self.pieces[destination as usize] = self.pieces[origin as usize];
        self.pieces[origin as usize] = Piece::Empty;

        self.side_to_move = self.side_to_move.opposite();

        assert!(self.is_consistent());
    }

    fn value(&self) -> u32 {
        let us = self.side_to_move as usize;

        let mut value = 0;
        let color_mask = self.colors[us];

        value += (self.positions[PieceKind::Pawn as usize] & color_mask).count_ones() * 100;
        value += (self.positions[PieceKind::Knight as usize] & color_mask).count_ones() * 320;
        value += (self.positions[PieceKind::Bishop as usize] & color_mask).count_ones() * 330;
        value += (self.positions[PieceKind::Rook as usize] & color_mask).count_ones() * 500;
        value += (self.positions[PieceKind::Queen as usize] & color_mask).count_ones() * 900;
        value += (self.positions[PieceKind::King as usize] & color_mask).count_ones() * 20000;

        value
    }

    fn is_consistent(&self) -> bool {
        for i in 0..64 {
            let piece = self.pieces[i as usize];
            let bb : u64 = 1 << i as u64;
            if self.positions[piece.kind() as usize] & bb == 0 {
                return false;
            }
            if piece.kind() != PieceKind::Empty {
                let c = piece.color();
                if self.colors[c as usize] & bb == 0 {
                    return false;
                }
            }
        }
        true
    }    
}

fn pop_square(board : &mut Bitboard) -> Square {
    let s = board.trailing_zeros();
    *board = *board & (*board - 1);
    s
}

// TODO: en-passant
fn pawn_attacks(us : usize, square : Square) -> Bitboard {
    let mut attacks = 0;
    //if square % 8 != 0 {
    //    attacks |= 1 << (square as Bitboard - 9 + 16 * us as Bitboard);
    //}
    //if square % 8 != 7 {
    //    attacks |= 1 << (square as Bitboard - 7 + 16 * us as Bitboard);
    //}
    attacks
}

#[derive(Copy, Clone)]
struct Moves {
    moves : [Move; 256],
    num_moves : usize,
}

fn create_move(origin : Square, destination : Square, promotion : PieceKind, move_type : MoveType) -> Move {
        let mut m = destination;
        m |= origin << 6;
        m |= (promotion as u32) << 12;
        m |= (move_type as u32) << 14;
        m
    }

impl Moves {
    fn new() -> Moves {
        Moves{moves : [0; 256], num_moves : 0}
    }
    fn push(&mut self, m : Move) {
        self.moves[self.num_moves] = m;
        self.num_moves += 1;
    }

    fn iter(&self) -> MoveIter {
        MoveIter{moves : self, idx : 0}
    }
}

struct MoveIter<'a> {
    moves : &'a Moves,
    idx : usize,
}

impl<'a> Iterator for MoveIter<'a> {
    type Item = Move;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.moves.num_moves {
            return None;
        }
        let i = self.idx;
        self.idx += 1;
        Some(self.moves.moves[i])
    }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Direction {
    North =  8,
    East  =  1,
    South = -8,
    West  = -1,
    NorthEast = 9,
    SouthEast = -7,
    SouthWest = -9,
    NorthWest = 7,
}

const FILE_ABB : Bitboard = 0x0101010101010101;
const FILE_HBB : Bitboard = FILE_ABB << 7;
fn shift(bb : Bitboard, dir : Direction) -> Bitboard {
    match dir {
        Direction::North => bb << 8,
        Direction::South => bb >> 8,
        Direction::East => (bb & !FILE_HBB) << 1,
        Direction::West => (bb & !FILE_ABB) >> 1,
        Direction::NorthEast => (bb & !FILE_HBB) << 9,
        Direction::NorthWest => (bb & !FILE_ABB) << 7,
        Direction::SouthEast => (bb & !FILE_HBB) >> 7,
        Direction::SouthWest => (bb & !FILE_ABB) >> 9,
    }
}

fn pawn_push(c : Color) -> Direction {
    match c {
        Color::White => Direction::North,
        Color::Black => Direction::South,
    }
}

fn generate_pawn_moves(position : &Position, moves : &mut Moves) {
    let us_color = position.side_to_move;
    let us = us_color as usize;
    let them = us_color.opposite() as usize;

    let up = pawn_push(us_color);
    let up_right = if us_color == Color::White {Direction::NorthEast} else {Direction::SouthEast};
    let up_left = if us_color == Color::White {Direction::NorthWest} else {Direction::SouthWest};

    println!("us {:?}", us);

    let empty = position.positions[PieceKind::Empty as usize];
    let pawns = position.colors[us] & position.positions[PieceKind::Pawn as usize];
    let enemies = position.colors[them];

    let b = shift(pawns, up);

    // captures
    {
        let mut b1 = shift(b, Direction::East) & enemies;
        while b1 != 0 {
            let destination = pop_square(&mut b1);
            let origin = destination as i32 - up_right as i32; 
            println!("generating pawn capture {} -> {}", origin, destination);
            moves.push(create_move(origin as u32, destination, PieceKind::Empty, MoveType::Normal));
        }

        let mut b2 = shift(b, Direction::West) & enemies;
        while b2 != 0 {
            let destination = pop_square(&mut b2);
            let origin = destination as i32 - up_left as i32; 
            println!("generating pawn capture {} -> {}", origin, destination);
            moves.push(create_move(origin as u32, destination, PieceKind::Empty, MoveType::Normal));
        }
    }

    {
        // single pawn push
        let mut b1 = b & empty;

        while b1 != 0 {
            let destination = pop_square(&mut b1);
            let origin = destination as i32 - up as i32;
            println!("generating single push move {} -> {}", origin, destination);
            moves.push(create_move(origin as u32, destination, PieceKind::Empty, MoveType::Normal));
        }

        // double pawn push
        let mut b2 = shift(b & empty & DOUBLE_PUSH_MASK[us], up) & empty;

        println!("num double pushes {}",b2.count_ones());

        while b2 != 0 {
            let destination = pop_square(&mut b2);
            let origin = destination as i32 - 2*(up as i32);
            println!("generating two push move {} -> {}", origin, destination);
            moves.push(create_move(origin as u32, destination, PieceKind::Empty, MoveType::Normal));
        }
    }
}

fn generate_moves(position : &Position) -> Moves {
    let mut moves = Moves::new();
    generate_pawn_moves(position, &mut moves);
    moves
}

fn alpha_beta_search(position : &Position, depth : usize, alpha_ : i32, beta_ : i32, maximizing_player : bool) -> i32 {
    if depth == 0 {
        return position.value() as i32;
    }
    let moves = generate_moves(position);
    if moves.num_moves == 0  {
        return position.value() as i32;
    }

    let mut alpha = alpha_;
    let mut beta = beta_;

    if maximizing_player {
        let mut value = i32::MIN;
        for m in moves.iter() {
            let mut new_position: Position = position.clone();
            new_position.do_move(m);
            value = max(value, alpha_beta_search(&new_position, depth - 1, alpha, beta, false));
            if value > beta {
                break;
            }
            alpha = max(alpha, value);
        }
        return value;
    }
    else {
        let mut value = i32::MAX;
        for m in moves.iter() {
            let mut new_position: Position = position.clone();
            new_position.do_move(m);
            value = max(value, alpha_beta_search(&new_position, depth - 1, alpha, beta, true));
            if value < alpha {
                break;
            }
            beta = min(beta, value);
        }
        return value;
    }
}

fn make_move(position : &mut Position) {
    let moves = generate_moves(position);
    let mut best_move = 0;
    let mut best_value = i32::MAX;
    for m in moves.iter() {
        //let v = alpha_beta_search(position, 5, i32::MIN, i32::MAX, true);
        let mut p = position.clone();
        p.do_move(m);
        let v = p.value() as i32;
        // choose position which has the lowest valuation for the opponent
        if v < best_value {
            best_move = m;
            best_value = v;
        }
    }
    position.do_move(best_move);
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone)]
struct GpuBoard {
    pieces : [i32; 64],
}

unsafe impl Zeroable for GpuBoard {}
unsafe impl Pod for GpuBoard {}

impl GpuBoard {
    fn new(pos : &Position) -> GpuBoard {
        let mut board = GpuBoard{ pieces : [0; 64]};
        for i in 0..64 {
           board.pieces[i] = pos.pieces[i] as i32;
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
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let mut position = Position::new();
    let mut gpu_board = GpuBoard::new(&position);

    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: gpu_board.as_byte_slice(),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(18 * 4),
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
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

    let mut mouse_down = false;
    let mut current_mouse_pos = None;

    // array of two mouse positions
    let mut i = 0;
    let mut mouse_positions: [PhysicalPosition<f64>; 2] = [winit::dpi::PhysicalPosition::default(); 2];
    let mut move_piece = false;

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = size.width;
                config.height = size.height;
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::MouseInput { button, state, .. },
                ..
            } => {
                if button == MouseButton::Left {
                    mouse_down = state == winit::event::ElementState::Pressed;
                    if !mouse_down {
                        mouse_positions[i] = current_mouse_pos.unwrap();
                        move_piece = i == 1;
                        i = (i + 1) % 2;
                    }
                    if move_piece {
                        move_piece = false;
                        let last_pos = mouse_positions[0];
                        let current_pos = mouse_positions[1];
                        let window_size = window.inner_size();

                        let height = window_size.height as f64;
                        let width = window_size.width as f64;

                        println!("{}, {}", last_pos.x, last_pos.y);
                        println!("{}, {}", current_pos.x, current_pos.y);
                        println!("{}, {}", width, height);

                        let square_size = 1./8.;

                        let last_x = (last_pos.x / width / square_size).floor() as i32;
                        let mut last_y = (last_pos.y / height / square_size).floor() as i32;
                        last_y = 7 - last_y;

                        let current_x = (current_pos.x / width / square_size).floor() as i32;
                        let mut current_y = (current_pos.y / height / square_size).floor() as i32;
                        current_y = 7 - current_y;

                        let from = last_x + last_y * 8;
                        let to = current_x + current_y * 8;
                        
                        //println!("{} -> {}", from, to);

                        let m = create_move(from as u32, to as u32, PieceKind::Bishop, MoveType::Normal);
                        position.do_move(m);

                        //println!("computing moves for color {:?}", position.side_to_move);
                        //for m in generate_moves(&position).iter() {
                        //    println!("{} -> {}", origin_square(m), destination_square(m));
                        //}

                        make_move(&mut position);

                        gpu_board = GpuBoard::new(&position);

                        queue.write_buffer(
                            &uniform_buf,
                            0,
                            gpu_board.as_byte_slice(),
                        );
                        window.request_redraw();
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
        // On wasm, append the canvas to the document body
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
