struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) vert_pos: vec3f,
    @location(5) color: vec3f,
    @location(6) radius: f32,
    @location(7) center: vec3f
}

struct VertexInput {
    @location(0) position: vec3f
}

struct Instance {
    @location(1) instance_matrix_0: vec4f,
    @location(2) instance_matrix_1: vec4f,
    @location(3) instance_matrix_2: vec4f,
    @location(4) instance_matrix_3: vec4f,
    @location(5) color: vec3f,
    @location(6) radius: f32,
    @location(7) center: vec2f
}

struct WindowProperties {
    dimensions: vec2f,
    aspect: f32
}

@group(0) @binding(0) var<uniform> window_properties: WindowProperties;

@vertex
fn vs_main(in: VertexInput, instance: Instance) -> VertexOutput {
    var out: VertexOutput;
    let instance_matrix = mat4x4f(
        instance.instance_matrix_0,
        instance.instance_matrix_1,
        instance.instance_matrix_2,
        instance.instance_matrix_3
    );
    let clip_pos = instance_matrix * vec4f(in.position, 1.0) / vec4f(window_properties.dimensions / 2.0, 1.0, 1.0);
    out.clip_position = clip_pos;
    out.vert_pos = clip_pos.xyz;
    out.color = instance.color;
    out.radius = instance.radius / window_properties.dimensions.x / 2.0;
    out.center = vec3f(instance.center / (window_properties.dimensions / 2.0), 0.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var diff = (in.center - in.vert_pos);
    diff.y /= window_properties.aspect;
    let distance_2 = diff.x * diff.x + diff.y * diff.y;
    let radius_2 = in.radius * in.radius;
    if distance_2 > radius_2 { discard; }
    return vec4(in.color, 1.0 - pow(sqrt(distance_2), 0.005) / pow(sqrt(radius_2), 0.005));
}