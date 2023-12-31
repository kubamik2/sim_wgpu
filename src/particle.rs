use cgmath::{Point2, Vector3, Matrix4, Vector4, Vector2};
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub position: Point2<f32>,
    pub radius: f32,
    pub color: Vector3<f32>,
    pub acceleration: Vector2<f32>,
    pub prev_position: Point2<f32>
}

impl Particle {
    pub fn new<P: Clone + Into<Point2<f32>>, C: Into<Vector3<f32>>>(position: P, radius: f32, color: C) -> Self {
        Self { 
            position: position.clone().into(),
            radius,
            color: color.into(),
            acceleration: Vector2::new(0.0, 0.0),
            prev_position: position.into()
        }
    }
}

impl Into<ParticleRaw> for Particle {
    fn into(self) -> ParticleRaw {
        let translation_matrix = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            self.position.x - self.radius, self.position.y - self.radius, 0.0, 1.0
        );

        let scale = self.radius * 2.0;
        let scale_matrix = Matrix4::new(
            scale, 0.0, 0.0, 0.0,
            0.0, scale, 0.0, 0.0,
            0.0, 0.0, scale, 0.0,
            0.0, 0.0, 0.0, 1.0
        );

        let instance_matrix = translation_matrix * scale_matrix;
        ParticleRaw { instance_matrix: instance_matrix.into(), color: self.color.into(), radius: self.radius, center: self.position.into(), _padding: [0; 2] }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleRaw {
    pub instance_matrix: [[f32; 4]; 4],
    pub color: [f32; 3],
    pub radius: f32,
    pub center: [f32; 2],
    _padding: [u32; 2]
}

impl ParticleRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 0 as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    shader_location: 2,
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    shader_location: 3,
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    shader_location: 4,
                    offset: std::mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute {
                    shader_location: 5,
                    offset: std::mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute {
                    shader_location: 6,
                    offset: std::mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32
                },
                wgpu::VertexAttribute {
                    shader_location: 7,
                    offset: std::mem::size_of::<[f32; 20]>() as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x2
                },
            ]
        }
    }
}

const INDICES: &[u16] = &[
    0, 1, 3,
    1, 2, 3
];

pub struct ParticleMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_elements: u32
}

impl ParticleMesh {
    pub fn new(device: &wgpu::Device, aspect: f32) -> Self {
        let vertices = &[
            Vertex::new([ 0.0, 0.0, 0.0 ]),
            Vertex::new([ 1.0, 0.0, 0.0 ]),
            Vertex::new([ 1.0, 1.0, 0.0 ]),
            Vertex::new([ 0.0, 1.0, 0.0 ]),
        ];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Vertex Buffer"),
            usage: wgpu::BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(vertices)
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Index Buffer"),
            usage: wgpu::BufferUsages::INDEX,
            contents: bytemuck::cast_slice(INDICES)
        });

        let num_elements = INDICES.len() as u32;

        Self { vertex_buffer, index_buffer, num_elements }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
}

impl Vertex {
    pub fn new(position: [f32; 3]) -> Self {
        Self { position }
    }

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0 as wgpu::BufferAddress,
                    format: wgpu::VertexFormat::Float32x3
                },
            ]
        }
    }
}
// #[repr(C)]
// #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
// pub struct ParticleCompute {
//     pub position: [f32; 2],
//     pub acceleration: [f32; 2],
// }

// impl Into<ParticleCompute> for Particle {
//     fn into(self) -> ParticleCompute {
//         ParticleCompute { position: self.position.into(), acceleration: self.velocity.into() }
//     }
// }

// impl ParticleCompute {
//     pub fn reconstruct(bytes: &[u8]) -> Vec<Self> {
//         if bytes.len() % 16 != 0 { panic!("ParticleCompute incorrect bytes len"); }
//         let mut particles = vec![];
//         for i in 0..bytes.len() / 16 {
//             let offset = i * 16;
//             let x = f32::from_le_bytes([bytes[0 + offset], bytes[1 + offset], bytes[2 + offset], bytes[3 + offset]]);
//             let y = f32::from_le_bytes([bytes[4 + offset], bytes[5 + offset], bytes[6 + offset], bytes[7 + offset]]);
//             let position = [x, y];

//             let x = f32::from_le_bytes([bytes[8 + offset], bytes[9 + offset], bytes[10 + offset], bytes[11 + offset]]);
//             let y = f32::from_le_bytes([bytes[12 + offset], bytes[13 + offset], bytes[14 + offset], bytes[15 + offset]]);
//             let velocity = [x, y];
//             particles.push(Self { position, acceleration: velocity });
//         }

//         particles
//     }
// }

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub mass: u32,
    pub center_of_mass: Point2<f32>,
    pub particle_index: usize
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct CellCompute {
    pub mass_option: u32,
    pub center_of_mass: [f32; 2],
    pub particle_index: u32
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeLinkedParticle {
    pub position: [f32; 2],
    pub link: i32,
    pub _padding: u32
}

const GRID_SIZE: usize = 3;

use cgmath::prelude::*;
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<Vec<Option<Cell>>>,
    
}

impl Grid {
    #[inline]
    pub fn new(linked_particles: &mut Vec<(Particle, Option<usize>)>) -> Self {
        let positions = linked_particles.iter().map(|f| f.0.position.cast().unwrap()).collect::<Vec<Point2<i32>>>();

        let min_x = positions.iter().min_by_key(|f| f.x).map(|f| f.x).unwrap();
        let max_x = positions.iter().max_by_key(|f| f.x).map(|f| f.x).unwrap();
        let max_y = positions.iter().max_by_key(|f| f.y).map(|f| f.y).unwrap();
        let min_y = positions.iter().min_by_key(|f| f.y).map(|f| f.y).unwrap();

        let cols = ((max_x - min_x).abs() as f32 / GRID_SIZE as f32).ceil() as usize + 1;
        let rows = ((max_y - min_y).abs() as f32 / GRID_SIZE as f32).ceil() as usize + 1;

        let mut cells: Vec<Vec<Option<Cell>>> = vec![vec![None; cols]; rows];

        for ((i, linked_particle), position) in linked_particles.iter_mut().enumerate().zip(positions) {
            let col_id = (position.x - min_x) as usize / GRID_SIZE;
            let row_id = (max_y - position.y) as usize / GRID_SIZE;
    
            let cell = cells.get_mut(row_id).unwrap().get_mut(col_id).unwrap();
            if let Some(cell) = cell {
                linked_particle.1 = Some(cell.particle_index);
                cell.center_of_mass.add_assign_element_wise(linked_particle.0.position);
                cell.mass += 1;
                cell.particle_index = i;
            } else {
                *cell = Some(Cell { mass: 1, center_of_mass: linked_particle.0.position, particle_index: i });
            }
        }

        for row in cells.iter_mut() {
            for cell in row.iter_mut() {
                if let Some(cell) = cell {
                    cell.center_of_mass /= cell.mass as f32;
                }
            }
        }
        
        Self { cells, rows, cols }
    }
}

impl Into<Vec<CellCompute>> for Grid {
    fn into(self) -> Vec<CellCompute> {
        self.cells.iter().cloned().map(|rows| {
            rows.iter().cloned().map(|f| 
                match f {
                    Some(cell) => {
                        CellCompute { mass_option: cell.mass | (1 << 31), center_of_mass: cell.center_of_mass.into(), particle_index: cell.particle_index as u32 }
                    },
                    None => CellCompute::default()
                }
            ).collect::<Vec<CellCompute>>()
        }).flatten().collect::<Vec<CellCompute>>()
    }
}