alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>; 

@group(0)
@binding(0)
var<uniform> pieces: array<vec4<i32>, 16>;

struct vertexoutput {
    @location(0) tex_coord: v2,
    @builtin(position) position: v4,
};

@vertex
fn vs_main(@builtin(vertex_index) j: u32) -> vertexoutput {
    let i = i32(j);
    var result: vertexoutput;
    if i == 0 {
        result.position = v4(-1., -1., 0., 1.);
        result.tex_coord = v2(0.,0.);
    }
    if i == 1 {
        result.position = v4(1., -1., 0., 1.);
        result.tex_coord = v2(1.,0.);
    }
    if i == 2 {
        result.position = v4(-1., 1., 0., 1.);
        result.tex_coord = v2(0.,1.);
    }
    if i == 3 {
        result.position = v4(1., -1., 0., 1.);
        result.tex_coord = v2(1.,0.);
    }
    if i == 4 {
        result.position = v4(1., 1., 0., 1.);
        result.tex_coord = v2(1.,1.);
    }
    if i == 5 { 
        result.position = v4(-1., 1., 0., 1.);
        result.tex_coord = v2(0.,1.);
    }
    return result;
}

fn dot2(p : v2) -> f32 
{
    return dot(p,p);
}

fn sd_circle(p : v2, r : f32) -> f32
{
    return length(p) - r;
}

fn sd_box(p : v2, b : v2) -> f32
{
    let d = abs(p)-b;
    return length(max(d,v2(0.0))) + min(max(d.x,d.y),0.0);
}

fn sd_rounded_box(p : v2, b : v2, r : v4) -> f32
{
    var r1 = r;
    if p.x <= 0.0 {
        r1.x = r1.z;
        r1.y = r1.w;
    }
    if p.y <= 0.0 {
       r1.x = r1.y; 
    }
    let q = abs(p)-b+r1.x;
    return min(max(q.x,q.y),0.0) + length(max(q,v2(0.0))) - r1.x;
}

fn sd_trapezoid(p1 : v2, r1 : f32, r2 : f32, he : f32) -> f32
{
    var p = p1;
    let k1 = v2(r2,he);
    let k2 = v2(r2-r1,2.0*he);
    p.x = abs(p.x);
    var r = r2;
    if p.y < 0.0 {
        r = r1;    
    }
    let ca = v2(p.x-min(p.x,r), abs(p.y)-he);
    let cb = p - k1 + k2*clamp( dot(k1-p,k2)/dot2(k2), 0.0, 1.0 );
    var s = 1.0;
    if cb.x<0.0 && ca.y<0.0 {
        s = -1.0;
    }
    return s*sqrt( min(dot2(ca),dot2(cb)) );
}

fn sd_egg(p1 : v2, ra : f32, rb : f32) -> f32
{
    var p = p1;
    let k = sqrt(3.0);
    p.x = abs(p.x);
    let r = ra - rb;
    if p.y < 0.0 {
       return length(v2(p.x, p.y)) - r - rb;
    }
    if k*(p.x+r)<p.y {
        return length(v2(p.x, p.y-k*r)) - rb;
    }
    return (length(v2(p.x+r,p.y)) - 2.0*r) - rb;
}

fn sd_oriented_box(p : v2, a : v2, b : v2, th : f32) -> f32
{
    let l = length(b-a);
    let  d = (b-a)/l;
    var  q = (p-(a+b)*0.5);
    q = mat2x2<f32>(d.x,-d.y,d.y,d.x)*q;
    q = abs(q)-v2(l,th)*0.5;
    return length(max(q,v2(0.0))) + min(max(q.x,q.y),0.0);    
}

fn sd_polygon(p : v2, v : ptr<function, array<v2, 9>>) -> f32
{
    var d = dot(p-(*v)[0],p-(*v)[0]);
    var s = 1.0;
    for(var i : i32 = 0; i < 9; i = i + 1)
    {
        let j = (i + 8) % 9;
        let e = (*v)[j] - (*v)[i];
        let w =    p - (*v)[i];
        let b = w - e*clamp( dot(w,e)/dot(e,e), 0.0, 1.0 );
        d = min(d, dot(b,b) );

        let cond = vec3<bool>(p.y>=(*v)[i].y, p.y < (*v)[j].y, e.x*w.y>e.y*w.x);
        if all(cond) || all(!cond) {
            s = -s;
        }
    }
    
    return s*sqrt(d);
} 

fn sd_heart(p1 : v2) -> f32
{
    var p = p1;
    p.x = abs(p.x);
    if p.y+p.x > 1.0 {
        return sqrt(dot2(p-v2(0.25,0.75))) - sqrt(2.0)/4.0;
    }
    let a = dot2(p-v2(0.00,1.00));
    let c = 0.5*max(p.x+p.y,0.0);
    let b = dot2(p-v2(c));
    return sqrt(min(a, b)) * sign(p.x-p.y);
}

fn op_smooth_union(d1 : f32, d2 : f32, k : f32) -> f32
{
    let h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix(d2, d1, h) - k*h*(1.0-h); 
}

fn op_union(d1 : f32, d2 : f32) -> f32
{ 
    return min(d1,d2); 
}

fn op_subtraction(d1 : f32, d2 : f32) -> f32
{ 
    return max(-d1,d2); 
}

fn pawn(p : v2) -> f32 
{
    let bottom = sd_rounded_box(p + v2(0., 0.7), v2(0.6, 0.2), v4(0.2,0.,0.2,0.));
    let middle = sd_trapezoid(p + vec2(0., 0.3), 0.4, 0.08, 0.4);
    let base = op_smooth_union(bottom, middle, 0.2);
    let top = sd_circle(p - v2(0.,0.3), 0.3);
    let tie1 = sd_oriented_box(p, v2(0.3,0.), v2(0.,0.1), 0.08) - 0.02;
    let tie2 = sd_oriented_box(p, v2(-0.3,0.), v2(0.,0.1), 0.08) - 0.02;
    let tie3 = sd_box(p + v2(0., 0.007), v2(0.3, 0.05));
    let tie = min(min(tie1, tie2), tie3);
    let a = op_subtraction(p.y + 0.825, min(base, top));
    return min(a, tie);
}

fn bishop(p : v2) -> f32 {
    let bottom = sd_rounded_box(p + v2(0., 0.7), v2(0.6, 0.15), v4(0.15,0.,0.15,0.));
    let middle = sd_egg(0.8 * p + v2(0., 0.15), 0.4, 0.08) / 0.8;
    let top = sd_circle(p - v2(0., 0.7), 0.2);
    let rect = sd_oriented_box(p, v2(0.,-0.1), v2(0.25, 0.9), 0.1);
    let b = op_union(op_union(bottom, middle), top);
    return op_subtraction(rect, b);
}

fn rook(p : v2) -> f32 {
    let bottom = sd_rounded_box(p + v2(0., 0.7), v2(0.6, 0.15), vec4(0.15,0.,0.15,0.));
    let middle = sd_trapezoid(p, 0.45, 0.3, 0.6);
    let top = sd_rounded_box(p - v2(0., 0.6), v2(0.5, 0.2), vec4(0.,0.1,0.,0.1));
   
    let nick1 = sd_box(p - v2(0.2, 0.75), v2(0.05, 0.1));
    let nick2 = sd_box(p - v2(-0.2, 0.75), v2(0.05, 0.1));
    let nicks = op_union(nick1, nick2);
    
    let base = op_union(op_union(bottom, middle), top);
    return op_subtraction(nicks, base);
}

fn queen(p : v2) -> f32 
{
    let bottom = sd_rounded_box(p + v2(0., 0.7), v2(0.6, 0.15), vec4(0.15,0.,0.15,0.));
    
    let tip1 = v2(-0.75, 0.5);
    let tip2 = v2(-0.25, 0.7);
    let tip3 = v2(0.25, 0.7);
    let tip4 = v2(0.75, 0.5);
    
    let ridge1 = v2(-0.35, 0.);
    let ridge2 = v2(0., 0.);
    let ridge3 = v2(0.35, 0.);
    
    let start = v2(-0.45, -0.7);
    let end = v2(.45, -0.7);
    
    var polygon = array<v2, 9>(
        start,
        tip1,
        ridge1,
        tip2,
        ridge2,
        tip3,
        ridge3,
        tip4,
        end
    );
    
    let middle = sd_polygon(p, &polygon);
    //let middle = 1.;
    
    var crown = op_union(bottom, middle);
    crown = op_union(crown, sd_circle(p - v2(-0.7, 0.4), 0.13));
    crown = op_union(crown, sd_circle(p - v2(-0.25, 0.63), 0.13));
    crown = op_union(crown, sd_circle(p - v2(.25, 0.63), 0.13));
    crown = op_union(crown, sd_circle(p - v2(.7, 0.4), 0.13));
    
    return crown;
}

fn king(p : v2) -> f32 
{
    let bottom = sd_rounded_box(p + v2(0., 0.7), v2(0.6, 0.15), vec4(0.15,0.0,0.15,0.0));
    
    //let translation = -0.5;
    //let middle = op_subtraction(sd_box(p + v2(0.,0.4 - translation), v2(0.5, 0.3)), sd_heart(0.7*p + v2(0.,0.2 - translation)) / 0.7);
    let translation = -0.55;
    let middle = op_subtraction(sd_box(p + v2(0.,0.4 - translation), v2(0.5, 0.3)), sd_heart(v2(p.x * 0.7, p.y * 0.8) + v2(0.,0.2 - translation)) / 0.75);
    
    let base = op_union(bottom, middle);

    let cross_base = sd_circle(p - v2(0., 0.3), 0.2);
    let cross_horizontal = sd_box(p - v2(0., 0.63), v2(0.18, 0.07));
    let cross_vertical = sd_box(p - v2(0., 0.63), v2(0.07, 0.18));
    let full_cross = op_union(op_union(cross_base, cross_horizontal), cross_vertical);
    
    return op_union(base, full_cross);
}

fn piece_color(square_id : i32) -> v3 {
    let piece = pieces[square_id / 4][square_id % 4];
    let is_even = (piece & 1) == 0;
    if is_even {
        // black
        return pow(v3(85.,83.,82.) / 255., v3(2.2));
    }
    // white
    return pow(v3(248.,248.,248.) / 255., v3(2.2));
}

fn dispatch_piece(p : v2, square_id : i32) -> f32 {
    let piece = pieces[square_id / 4][square_id % 4];
    if piece == 0 {
        return 1.;
    }
    let piece_kind = (piece - 1) / 2;
    switch piece_kind {
        case 0: { return pawn(p); }
        case 1: { return sd_circle(p, 0.8); }
        case 2: { return bishop(p); }
        case 3: { return rook(p); }
        case 4: { return queen(p); }
        case 5: { return king(p); } 
        default: { return 1.; }
    }
    return 1.;
}

@fragment
fn fs_main(vertex: vertexoutput) -> @location(0) vec4<f32> {
    let uv = vertex.tex_coord;
    let a = 1. / 8.;
    let x = uv.x / a;
    let y = uv.y / a;

    let x_trunc = trunc(x);
    let y_trunc = trunc(y);

    let i = i32(x_trunc);
    let j = i32(y_trunc);

    let px = 2.*(x - x_trunc) - 1.;
    let py = 2.*(y - y_trunc) - 1.;
    let p = v2(px, py);

    let square_id = 8 * j + i;
    let d = dispatch_piece(p, square_id);
    var col : v3;
    if d < 0.0 {
        col = piece_color(square_id);
    }
    else {
        let is_white = ((i + j) % 2) == 0;
        if is_white {
            col = pow(v3(238.,238.,210.) / 255., v3(2.2));
        } 
        else {
            col = pow(v3(117.,150.,86.) / 255., v3(2.2));
        }
    }
    col = mix(col, vec3(0.0), 1.0-smoothstep(0.0,0.03,abs(d)) );
    return v4(col.x, col.y, col.z, 1.);
}

