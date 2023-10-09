use cgmath::Point2;
use cgmath::prelude::*;
use crate::Particle;

pub struct BoundingBox {
    pub position: Point2<f32>,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn contains(&self, point: Point2<f32>) -> bool {
        !(
            point.x < self.position.x || point.x > self.position.x + self.width ||
            point.y > self.position.y || point.y < self.position.y - self.height
        )
    }

    pub fn intersects_radius(&self, bounding_circle: BoundingCircle) -> bool {
        let clamped_point = Point2::new(
            bounding_circle.position.x.clamp(self.position.x, self.position.x + self.width),
            bounding_circle.position.y.clamp(self.position.y - self.height, self.position.y)
        );
        let vector = clamped_point - bounding_circle.position;
        let distance = vector.magnitude();
        return distance < bounding_circle.radius;
    }
}

pub struct BoundingCircle {
    pub position: Point2<f32>,
    pub radius: f32
}

pub struct QuadTreeNode {
    pub bounding_box: BoundingBox,
    pub particle_index: i32,
    pub ul: i32,
    pub ur: i32,
    pub dl: i32,
    pub dr: i32,
}

impl QuadTreeNode {
    pub fn new(bounding_box: BoundingBox) -> Self {
        Self { bounding_box, particle_index: -1, ul: -1, ur: -1, dl: -1, dr: -1 }
    }

    pub fn insert(&mut self, particle_index: usize, particles: &Vec<Particle>, nodes: &mut Vec<QuadTreeNode>) -> bool {
        let particle = particles[particle_index];
        if !self.bounding_box.contains(particle.position) { return false; }
        todo!()
    }

    pub fn subdivide(&mut self, particles: &Vec<Particle>, nodes: &mut Vec<QuadTreeNode>) {
        let half_width = self.bounding_box.width / 2.0;
        let half_height = self.bounding_box.height / 2.0;
        
        let nodes_len = nodes.len() as i32;

        let mut ul = Self::new(BoundingBox { position: self.bounding_box.position, width: half_width, height: half_height });
        self.ul = nodes_len;

        let mut ur = Self::new(BoundingBox { position: self.bounding_box.position.add_element_wise(Point2::new(half_width, 0.0)), width: half_width, height: half_height });
        self.ur = nodes_len + 1;

        let mut dl = Self::new(BoundingBox { position: self.bounding_box.position.add_element_wise(Point2::new(0.0, -half_height)), width: half_width, height: half_height });
        self.dl = nodes_len + 2;

        let mut dr = Self::new(BoundingBox { position: self.bounding_box.position.add_element_wise(Point2::new(half_width, -half_height)), width: half_width, height: half_height });
        self.dr = nodes_len + 3;

        if self.particle_index > 0 {
            let particle = particles[self.particle_index as usize];
            if ul.bounding_box.contains(particle.position) {
                ul.particle_index = self.particle_index;
            } else if ur.bounding_box.contains(particle.position) {
                ur.particle_index = self.particle_index;
            } else if dl.bounding_box.contains(particle.position) {
                dl.particle_index = self.particle_index;
            } else if dr.bounding_box.contains(particle.position) {
                dr.particle_index = self.particle_index;
            }
            self.particle_index = -1;
        }
        
        nodes.push(ul);
        nodes.push(ur);
        nodes.push(dl);
        nodes.push(dr);
    }
}