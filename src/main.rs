mod particle;
mod quadtree;
use std::time::{Instant, Duration};

use cgmath::{Point2, Vector3, Vector2};
use particle::*;

use wgpu::util::DeviceExt;
use winit::{event::{Event, WindowEvent, DeviceEvent, VirtualKeyCode, ElementState, MouseButton}, dpi::PhysicalSize};
use imgui_winit_support::WinitPlatform;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WindowProperties {
    dimensions: [f32; 2],
    aspect: f32,
    _padding: u32,

}

struct Data {
    avg_dt: [Duration; 144],
    avg_compute_dt: [Duration; 144],
    avg_render_dt: [Duration; 144],
    gravity_constant: f32
}

impl Data {
    fn avg_compute_dt(&self) -> Duration {
        let sum: Duration = self.avg_compute_dt.iter().sum();
        sum / 144
    }

    fn avg_render_dt(&self) -> Duration {
        let sum: Duration = self.avg_render_dt.iter().sum();
        sum / 144
    }

    fn avg_dt(&self) -> Duration {
        let sum: Duration = self.avg_dt.iter().sum();
        sum / 144
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: winit::window::Window,
    render_pipeline: wgpu::RenderPipeline,
    imgui_renderer: imgui_wgpu::Renderer,
    imgui: imgui::Context,
    imgui_platform: imgui_winit_support::WinitPlatform,
    particle_mesh: ParticleMesh,
    particles: Vec<Particle>,
    window_properties_bind_group: wgpu::BindGroup,
    data: Data,
    gravity_constant_buffer: wgpu::Buffer,
    gravity_constant_bind_group_layout: wgpu::BindGroupLayout,
    gravity_constant_bind_group: wgpu::BindGroup
}

impl State {
    async fn new(window: winit::window::Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default()
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false
        }).await.unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            features: wgpu::Features::BUFFER_BINDING_ARRAY | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY,
            label: None,
            limits: wgpu::Limits::default()
        }, None).await.unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied().find(|p| p.is_srgb()).unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![]
        };

        surface.configure(&device, &config);

        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle_shader.wgsl").into())
        });

        let window_properties = WindowProperties { dimensions: [size.width as f32, size.height as f32], aspect: size.width as f32 / size.height as f32, _padding: 0 };
        let window_properties_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WindowProperties Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
            }]
        });

        let window_properties_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("WindowProperties Buffer"),
            contents: bytemuck::cast_slice(&[window_properties]),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let window_properties_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WindowProperties Bind Group"),
            layout: &window_properties_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: window_properties_buffer.as_entire_binding()
                }
            ]
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &window_properties_bind_group_layout
            ],
            push_constant_ranges: &[]
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            depth_stencil: None,
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc(),
                    ParticleRaw::desc(),
                ]
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL
                })],

            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false
            },
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false
            },
            multiview: None
        });

        // let compute_particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("ComputeParticle Buffer"),
        //     contents: &[],
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        // });

        // let compute_particle_result_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("ComputeParticle Result Buffer"),
        //     contents: &[],
        //     usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        // });

        // let compute_particle_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //     label: Some("ComputeParticle Bind Group Layout"),
        //     entries: &[wgpu::BindGroupLayoutEntry {
        //         binding: 0,
        //         count: None,
        //         ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
        //         visibility: wgpu::ShaderStages::COMPUTE
        //     }]
        // });

        // let compute_particle_result_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //     label: Some("ComputeParticle Bind Group Layout"),
        //     entries: &[wgpu::BindGroupLayoutEntry {
        //         binding: 1,
        //         count: None,
        //         ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
        //         visibility: wgpu::ShaderStages::COMPUTE
        //     }]
        // });
        
        // let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { 
        //     label: Some("Compute Shader"),
        //     source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into())
        // });

        // let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //     label: Some("Compute Pipeline Layout"),
        //     bind_group_layouts: &[
        //         &compute_particle_bind_group_layout,
        //         &compute_particle_result_bind_group_layout
        //     ],
        //     push_constant_ranges: &[] 
        // });

        // let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        //     label: Some("Compute Pipeline"),
        //     layout: Some(&compute_pipeline_layout),
        //     module: &compute_shader,
        //     entry_point: "cmp_main"
        // });

        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        let mut imgui_platform = WinitPlatform::init(&mut imgui);
        imgui.fonts().build_rgba32_texture();
        imgui_platform.attach_window(imgui.io_mut(), &window, imgui_winit_support::HiDpiMode::Default);
        let hidpi_factor = window.scale_factor();

        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        imgui.fonts().add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                oversample_h: 1,
                pixel_snap_h: true,
                size_pixels: font_size,
                ..Default::default()
            })
        }]);
        let renderer_config = imgui_wgpu::RendererConfig {
            texture_format: surface_format,
            ..Default::default()
        };

        let imgui_renderer = imgui_wgpu::Renderer::new(&mut imgui, &device, &queue, renderer_config);

        let particle_mesh = ParticleMesh::new(&device, size.width as f32 / size.height as f32);

        let particles = vec![];

        let gravity_constant_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gravity Constant Buffer"),
            contents: bytemuck::cast_slice(&[0.6674 as f32]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
        });

        let gravity_constant_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gravity Constant Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    has_dynamic_offset: false,
                    min_binding_size: None,
                    ty: wgpu::BufferBindingType::Uniform
                },
                visibility: wgpu::ShaderStages::COMPUTE
            }]
        });

        let gravity_constant_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gravity Constant Bind Group"),
            layout: &gravity_constant_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gravity_constant_buffer.as_entire_binding()
            }]
        });

        let data = Data { avg_dt: [Duration::ZERO; 144], avg_compute_dt: [Duration::ZERO; 144], avg_render_dt: [Duration::ZERO; 144], gravity_constant: 0.6674 as f32 };
        Self { surface, device, queue, config, size, window, render_pipeline, imgui_renderer, imgui, imgui_platform, particle_mesh, particles, window_properties_bind_group, data, gravity_constant_buffer, gravity_constant_bind_group_layout, gravity_constant_bind_group }
    }

    fn update(&mut self, dt: Duration, frame: usize) {
        self.imgui.io_mut().update_delta_time(dt);
        if self.particles.len() > 0 {
            let now = Instant::now();
            self.queue.write_buffer(&self.gravity_constant_buffer, 0, bytemuck::cast_slice(&[self.data.gravity_constant]));
            let compute_shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { 
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("compute_shader.wgsl").into())
            });
            
            let particle_compute_vec = self.particles.iter().map(|f| { let pos = f.position.clone(); [pos.x, pos.y]}).collect::<Vec<[f32; 2]>>();
            let particle_compute_bytes = bytemuck::cast_slice(particle_compute_vec.as_slice());
            let particle_compute_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ParticleCompute Buffer"),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                contents: particle_compute_bytes
            });

            let particle_compute_result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ParticleCompute Result Buffer"),
                mapped_at_creation: false,
                size: particle_compute_bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
            });

            let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Read Buffer"),
                mapped_at_creation: false,
                size: particle_compute_bytes.len() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST
            });

            let compute_bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                        visibility: wgpu::ShaderStages::COMPUTE
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                        visibility: wgpu::ShaderStages::COMPUTE
                    },
                ]
            });

            let compute_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Bind Group"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: particle_compute_buffer.as_entire_binding()
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_compute_result_buffer.as_entire_binding()
                    },
                ]
            });
    
            let compute_pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &compute_bind_group_layout,
                    &self.gravity_constant_bind_group_layout
                ],
                push_constant_ranges: &[] 
            });
    
            let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &compute_shader,
                entry_point: "cmp_main"
            });
            let n = Instant::now();
            let min_x = self.particles.iter().min_by_key(|f| f.position.x as i32);
            let max_x = self.particles.iter().max_by_key(|f| f.position.x as i32);
            let max_y = self.particles.iter().max_by_key(|f| f.position.y as i32);
            let min_y = self.particles.iter().max_by_key(|f| f.position.y as i32);
            dbg!(n.elapsed());
            let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Compute Encoder") });
            {
                let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Compute Pass") });
                compute_pass.set_pipeline(&compute_pipeline);
                compute_pass.set_bind_group(0, &compute_bind_group, &[]);
                compute_pass.set_bind_group(1, &self.gravity_constant_bind_group, &[]);
                compute_pass.dispatch_workgroups(1024, 1, 1);
            }
            
            command_encoder.copy_buffer_to_buffer(&particle_compute_result_buffer, 0, &read_buffer, 0, particle_compute_bytes.len() as u64);
            self.queue.submit(std::iter::once(command_encoder.finish()));
            
            
            let (tx, rx) = futures_channel::oneshot::channel();
            read_buffer.slice(..).map_async(wgpu::MapMode::Read,|f| {tx.send(f);});

            self.device.poll(wgpu::Maintain::Wait);
            
            futures::executor::block_on(rx).unwrap().unwrap();
            let bytes = read_buffer.slice(..).get_mapped_range().to_vec();

            let mut acceleration_vec = vec![];
            for i in 0..bytes.len() / 8 {
                let offset = i * 8;
                let x = f32::from_le_bytes([bytes[0 + offset], bytes[1 + offset], bytes[2 + offset], bytes[3 + offset]]);
                let y = f32::from_le_bytes([bytes[4 + offset], bytes[5 + offset], bytes[6 + offset], bytes[7 + offset]]);
                acceleration_vec.push(Vector2::new(x, y));
            }
            for (i, particle) in self.particles.iter_mut().enumerate() {
                particle.acceleration = acceleration_vec[i];
                let new_pos = (particle.position * 2.0) - particle.prev_position + (particle.acceleration * dt.as_secs_f32() * dt.as_secs_f32());
                particle.prev_position = particle.position;
                particle.position = Point2::new(new_pos.x, new_pos.y);
            }

            self.data.avg_compute_dt[frame] = now.elapsed();
        }
    }   

    fn render(&mut self, frame: usize) -> Result<(), wgpu::SurfaceError> {
        let now = Instant::now();
        self.imgui_platform.prepare_frame(self.imgui.io_mut(), &self.window).unwrap();
        let dt = self.imgui.io().delta_time;
        let mouse_pos = self.imgui.io().mouse_pos;

        let ui = self.imgui.frame();
        {
            let window = ui.window("Debug");
            window
                .size([300.0, 150.0], imgui::Condition::FirstUseEver)
                .no_decoration()
                .movable(false)
                .position([0.0, 0.0], imgui::Condition::Always)
                .bg_alpha(0.0)
                .build(|| {
                    ui.text(format!("dt: {:?}   {:.0} fps", self.data.avg_dt(), 1.0 / self.data.avg_dt().as_secs_f32()));
                    ui.text(format!("compute_dt: {:?} ({:.0}%)", self.data.avg_compute_dt(), (self.data.avg_compute_dt().as_secs_f32() / dt) * 100.0));
                    ui.text(format!("render_dt: {:?} ({:.0}%)", self.data.avg_render_dt(), (self.data.avg_render_dt().as_secs_f32() / dt) * 100.0));
                    ui.separator();
                    ui.text(format!("mouse_pos: {:?}", mouse_pos));
                    ui.separator();
                    ui.text(format!("entities: {}", self.particles.len()));
                    ui.separator();
                    ui.slider("G", 0.0, 6.0, &mut self.data.gravity_constant)
                });
        }
        self.imgui_platform.prepare_render(ui, &self.window);

        let particle_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(self.particles.iter().cloned().map(|f| f.into()).collect::<Vec<ParticleRaw>>().as_slice())
        });

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });
        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0005, g: 0.00, b: 0.001, a: 1.0}), store: true }
                })],
                depth_stencil_attachment: None
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.window_properties_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.particle_mesh.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, particle_buffer.slice(..));
            render_pass.set_index_buffer(self.particle_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(0..self.particle_mesh.num_elements, 0, 0..self.particles.len() as u32);
        }

        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: true }
                })],
                depth_stencil_attachment: None
            });
            self.imgui_renderer.render(self.imgui.render(), &self.queue, &self.device, &mut render_pass).expect("Gui Render Failed");
        }
        self.queue.submit(std::iter::once(command_encoder.finish()));
        output.present();
        self.data.avg_render_dt[frame] = now.elapsed();
        Ok(())
    }
}

use cgmath::prelude::*;
fn main() {
    env_logger::init();
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new().with_title("sim").build(&event_loop).unwrap();

    let mut state = futures::executor::block_on(State::new(window));
    
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    for _ in 0..30_000 {
        let radius = 300.0 * rng.gen::<f32>();
        let radians = 2.0 * std::f32::consts::PI * rng.gen::<f32>();
        let x = radians.cos() * radius;
        let y = radians.sin() * radius;
        let mut particle = Particle::new((x, y), 15.0, (0.1, 0.4, 1.0));
        //particle.acceleration = ((rng.gen::<f32>() - 0.5) / 10.0, (rng.gen::<f32>() - 0.5) / 10.0).into();
        state.particles.push(particle);
    }

    let mut last_render_time = Instant::now();
    let mut frame = 0;
    event_loop.run(move |event, _, control_flow|  {
        control_flow.set_poll();
        state.imgui_platform.handle_event(state.imgui.io_mut(), &state.window, &event);
        
        match event {
            Event::WindowEvent { window_id, ref event } if window_id == state.window.id() => {
                match event {
                    WindowEvent::CloseRequested => control_flow.set_exit(),
                    WindowEvent::KeyboardInput { input, .. } if input.virtual_keycode == Some(VirtualKeyCode::Escape) => {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::Escape) => control_flow.set_exit(),
                            _ => ()
                        }
                    },
                    WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                        let mouse_pos = state.imgui.io().mouse_pos;
                        let mut screen_space_pos = [(mouse_pos[0] - state.size.width as f32 / 2.0) * 1.0, 1.0 * (state.size.height as f32 - mouse_pos[1] - state.size.height as f32 / 2.0)];
                        state.particles.push(Particle::new(screen_space_pos, 15.0, (0.1, 0.4, 1.0)));
                        screen_space_pos[0] += 16.1;
                        state.particles.push(Particle::new(screen_space_pos, 15.0, (0.1, 0.4, 1.0)));
                        screen_space_pos[1] -= 12.2;
                        state.particles.push(Particle::new(screen_space_pos, 15.0, (0.1, 0.4, 1.0)));
                        screen_space_pos[0] -= 20.0;
                        state.particles.push(Particle::new(screen_space_pos, 15.0, (0.1, 0.4, 1.0)));
                    },
                    _ => ()
                }
            },
            Event::RedrawRequested(window_id) if window_id == state.window.id() => {
                let now = Instant::now();
                let dt = now - last_render_time;
                state.data.avg_dt[frame] = dt;
                last_render_time = now;

                state.update(dt, frame);
                state.render(frame);
                frame += 1;
                if frame > 143 { frame = 0; }
            },
            Event::MainEventsCleared => {
                state.window.request_redraw();
            }
            _ => ()
        }
    });
}