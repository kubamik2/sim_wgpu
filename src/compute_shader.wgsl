struct LinkedParticle {
    position: vec2f,
    link: i32,
}

struct Cell {
    mass_option: u32,
    center_of_mass: vec2f,
    particle_index: u32
}

@group(0) @binding(0) var<storage, read> input: array<LinkedParticle>;
@group(0) @binding(1) var<storage, write> result: array<vec2f>;
@group(0) @binding(2) var<storage, read> grid: array<Cell>;
@group(0) @binding(3) var<uniform> grid_size: vec2i;
@group(1) @binding(0) var<uniform> G: f32;

@compute @workgroup_size(128)
fn cmp_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * global_id.z;
    if index > arrayLength(&input) + u32(1) { return; }
    var p1 = input[index].position;
    var acc = vec2f(0.0, 0.0);
    for (var i = 0; i < i32(arrayLength(&input)); i++) {
        let i = u32(i);
        if i == index { continue; }
        let p2 = input[i].position;
        let position_vector = p2 - p1;
        let distance_2 = position_vector.x * position_vector.x + position_vector.y * position_vector.y;
        if distance_2 < 1.0 { continue; }
        let gravity_acceleration = (position_vector * G) / distance_2;
        acc += gravity_acceleration;
    }
    result[index] = acc;
}