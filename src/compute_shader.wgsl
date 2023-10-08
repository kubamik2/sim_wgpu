struct Particle {
    position: vec2f,
    velocity: vec2f
}

struct A {
    value: f32,
    node: A
}

@group(0) @binding(0) var<storage, read> input: array<vec2f>;
@group(0) @binding(1) var<storage, write> result: array<vec2f>;
@group(1) @binding(0) var<uniform> G: f32;

@compute @workgroup_size(128)
fn cmp_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * global_id.z;
    if index > arrayLength(&input) + u32(1) { return; }
    var p1 = input[index];
    var acc = vec2f(0.0, 0.0);
    for (var i = 0; i < i32(arrayLength(&input)); i++) {
        let i = u32(i);
        if i == index { continue; }
        let p2 = input[i];
        let position_vector = p2 - p1;
        let distance_2 = position_vector.x * position_vector.x + position_vector.y * position_vector.y;
        if distance_2 < 1.0 { continue; }
        let gravity_velocity = (position_vector * G) / distance_2;
        acc += gravity_velocity;
    }
    result[index] = acc;
}