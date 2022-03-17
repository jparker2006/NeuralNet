extern crate rand;
extern crate serde;
extern crate image;

use rand::Rng;
use serde::{Serialize, Deserialize};
use image::GenericImageView;
use std::time::Duration;
use std::time::Instant;

#[derive(Serialize, Deserialize, Debug)]
pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    pub data: Vec<Vec<f32>>
}

impl Matrix {
    // constructors
    pub fn r_new(r: i32, c: i32) -> Self { // new randomized matrix
        let mut mat = Matrix {
            rows: r,
            cols: c,
            data: Vec::new()
        };
        mat.randomize();
        return mat
    }
    pub fn from_vector_new(d: Vec<f32>) -> Self { // new matrix from 1d vector
        let mut mat = Matrix {
            rows: d.len() as i32,
            cols: 1,
            data: Vec::new()
        };
        for i in 0..d.len() {
            mat.data.push(Vec::new());
            mat.data[i].push(d[i]);
        }
        return mat
    }

    pub fn print(&self) {
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                print!("{}\t", self.data[x][y]);
            }
            println!();
        }
        println!();
    }

    // scalar operations
    pub fn s_mult(&mut self, n: f32) {
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] *= n;
            }
        }
    }

    // element wise operations
    pub fn e_add(&mut self, m: &Matrix) -> &mut Matrix { // returning a reference might be an error
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] += m.data[x][y];
            }
        }
        return self
    }
    pub fn e_sub(&mut self, m: &Matrix) -> &mut Matrix {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] -= m.data[x][y];
            }
        }
        return self
    }
    pub fn e_mult(&mut self, m: &Matrix) -> &mut Matrix {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] *= m.data[x][y];
            }
        }
        return self
    }

    pub fn transpose(m: &Matrix) -> Matrix {
        let mut new_data = Vec::new();
        for x in 0..m.cols {
            let mut v: Vec<f32> = Vec::new();
            for y in 0..m.rows {
                v.push(m.data[y as usize][x as usize]);
            }
            new_data.push(v);
        }

        return Matrix {
            rows: m.cols,
            cols: m.rows,
            data: new_data
        }
    }

    pub fn dot_product(&mut self, m: &Matrix) -> Matrix {
        let mut nm = Matrix {
            rows: self.rows,
            cols: m.cols,
            data: Vec::new()
        };

        if self.cols != m.rows {
            println!("dot product could not be performed");
            return nm
        }

        for x in 0..self.rows as usize {
            for y in 0..m.cols as usize {
                let mut sum: f32 = 0.0;
                for z in 0..self.cols as usize {
                    sum += self.data[x][z] * m.data[z][y];
                }
                nm.data.push(Vec::new());
                nm.data[x].push(sum);
            }
        }
        return nm
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for _x in 0..self.rows {
            let mut vector: Vec<f32> = Vec::new();
            for _y in 0..self.cols {
                vector.push(rng.gen::<f32>() * 2.0 - 1.0);
            }
            self.data.push(vector);
        }
    }

    pub fn map_to_function(&mut self, func: &dyn Fn(f32) -> f32) {
        for x in 0..self.rows {
            for y in 0..self.cols {
                self.data[x as usize][y as usize] = func(self.data[x as usize][y as usize]);
            }
        }
    }

    pub fn static_map_to_function(m: &Matrix, func: &dyn Fn(f32) -> f32) -> Matrix {
        let mut new_data = Vec::new();
        for x in 0..m.rows {
            let mut v: Vec<f32> = Vec::new();
            for y in 0..m.cols {
                v.push(func(m.data[x as usize][y as usize]));
            }
            new_data.push(v);
        }

        return Matrix {
            rows: m.rows,
            cols: m.cols,
            data: new_data
        }
    }

    pub fn to_vector(&mut self) -> Vec<f32> {
        let mut vector: Vec<f32> = Vec::new();
        for x in 0..self.rows {
            for y in 0..self.cols {
                vector.push(self.data[x as usize][y as usize]);
            }
        }
        return vector
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNet {
    ih_weight: Matrix,
    ho_weight: Matrix,
    h_bias: Matrix,
    o_bias: Matrix,
    learning_rate: f32,
    clock_epoch: Duration
}

impl NeuralNet {
    pub fn new(i: i32, h: i32, o: i32) -> Self {
        NeuralNet {
            ih_weight: Matrix::r_new(h, i),
            ho_weight: Matrix::r_new(o, h),
            h_bias: Matrix::r_new(h, 1),
            o_bias: Matrix::r_new(o, 1),
            learning_rate: 0.1,
            clock_epoch: Duration::from_secs_f32(0.0)
        }
    }

    pub fn feed_foward(&mut self, input_data: Vec<f32>) -> Vec<f32> {
        let m_input_data = Matrix::from_vector_new(input_data);
        let m_hidden = self.calc_ih(&m_input_data);
        return self.calc_ho(&m_hidden).to_vector()
    }

    pub fn calc_ih(&mut self, m_input_data: &Matrix) -> Matrix {
        let mut m_hidden: Matrix = self.ih_weight.dot_product(&m_input_data);
        m_hidden.e_add(&self.h_bias);
        m_hidden.map_to_function(&sigmoid);
        return m_hidden
    }

    pub fn calc_ho(&mut self, m_hidden: &Matrix) -> Matrix {
        let mut m_output: Matrix = self.ho_weight.dot_product(&m_hidden);
        m_output.e_add(&self.o_bias);
        m_output.map_to_function(&sigmoid);
        return m_output
    }

    pub fn train(&mut self, input_data: Vec<Vec<f32>>) {
        let m_input_data = Matrix::from_vector_new(input_data[0].clone());
        let m_hidden: Matrix = self.calc_ih(&m_input_data);
        let output_data: Matrix = self.calc_ho(&m_hidden);
        let mut m_target_data = Matrix::from_vector_new(input_data[1].clone());
        let err_output = m_target_data.e_sub(&output_data);
        let mut o_gradient = Matrix::static_map_to_function(&output_data, &d_sigmoid);
        o_gradient.e_mult(err_output);
        o_gradient.s_mult(self.learning_rate);
        let t_hidden = Matrix::transpose(&m_hidden);
        let ho_dweights = o_gradient.dot_product(&t_hidden);
        self.ho_weight.e_add(&ho_dweights);
        self.o_bias.e_add(&o_gradient);
        let mut ho_weight_t = Matrix::transpose(&self.ho_weight);
        let err_hidden = ho_weight_t.dot_product(&err_output);
        let mut h_gradient = Matrix::static_map_to_function(&m_hidden, &d_sigmoid);
        h_gradient.e_mult(&err_hidden);
        h_gradient.s_mult(self.learning_rate);
        let t_input = Matrix::transpose(&m_input_data);
        let ih_dweights = h_gradient.dot_product(&t_input);
        self.ih_weight.e_add(&ih_dweights);
        self.h_bias.e_add(&h_gradient);
    }

    pub fn calc_lr(&mut self, nepoch: i32, nindex: i32) {
/*      let fpercent: f32 = nindex as f32 / nepoch as f32;
        self.learning_rate = f32::powf(self.learning_rate, 1.24-fpercent);
        let numerator: f32 = (0.5 * f32::powf(nepoch as f32, 2.8)) * 0.018;
        let denomenator: f32 = f32::powf(nindex as f32 * 60.0, 1.9) + (2.0 * (f32::powf(nepoch as f32 * 5.0, 1.8)));
        self.learning_rate = numerator / denomenator;
        f(x) = 8(a)^3/x^2+4(a)^2
        self.learning_rate = (f32::powf(8.0 * fpercent, 3.0)) / (f32::powf(self.learning_rate, 2.0)) + 4.0 * (fpercent * fpercent);*/
        self.learning_rate = 5.0 + -1.0/(0.2*nepoch as f32) * nindex as f32;
//         println!("{}", self.learning_rate);
    }
}

pub fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-x));
}

pub fn d_sigmoid(x: f32) -> f32 {
    return x * (1.0 - x);
}

fn main() {
    #[allow(non_snake_case)]
    let e_CIRCLE = 0.0;
    #[allow(non_snake_case)]
    let e_SQUARE = 0.5;
    #[allow(non_snake_case)]
    let e_TRIANGLE = 1.0;
    let mut data: Vec<Vec<Vec<f32>>> = vec![];
    println!("starting");
    let start = Instant::now();
    for i in 1..100 { // circle data
        let path: &str;
        if i < 10 {
            path = "/Users/jparker/NeuralNet/data/circle000";
        }
        else {
            path = "/Users/jparker/NeuralNet/data/circle00";
        }
        let pic_number: &str = &i.to_string();
        let extention = ".png";
        let final_path = [path, pic_number, extention].concat();
        let img = image::open(final_path).unwrap();
        let mut pix_vec: Vec<f32> = vec![];
        for pixel in img.pixels() {
            let mut img_sum: f32 = 0.0;
            img_sum += pixel.2[0] as f32 + pixel.2[1] as f32 + pixel.2[2] as f32;
            pix_vec.push(img_sum / 765.0);
        }
        data.push(vec![pix_vec, vec![e_CIRCLE]]);
    }

    for i in 1..100 { // square data
        let path: &str;
        if i < 10 {
            path = "/Users/jparker/NeuralNet/data/square000";
        }
        else {
            path = "/Users/jparker/NeuralNet/data/square00";
        }
        let pic_number: &str = &i.to_string();
        let extention = ".png";
        let final_path = [path, pic_number, extention].concat();
        let img = image::open(final_path).unwrap();
        let mut pix_vec: Vec<f32> = vec![];
        for pixel in img.pixels() {
            let mut img_sum: f32 = 0.0;
            img_sum += pixel.2[0] as f32 + pixel.2[1] as f32 + pixel.2[2] as f32;
            pix_vec.push(img_sum / 765.0);
        }
        data.push(vec![pix_vec, vec![e_SQUARE]]);
    }

    for i in 1..100 { // triangle data
        let path: &str;
        if i < 10 {
            path = "/Users/jparker/NeuralNet/data/triangle000";
        }
        else {
            path = "/Users/jparker/NeuralNet/data/triangle00";
        }
        let pic_number: &str = &i.to_string();
        let extention = ".png";
        let final_path = [path, pic_number, extention].concat();
        let img = image::open(final_path).unwrap();
        let mut pix_vec: Vec<f32> = vec![];
        for pixel in img.pixels() {
            let mut img_sum: f32 = 0.0;
            img_sum += pixel.2[0] as f32 + pixel.2[1] as f32 + pixel.2[2] as f32;
            pix_vec.push(img_sum / 765.0);
        }
        data.push(vec![pix_vec, vec![e_TRIANGLE]]);
    }

    let mut network = NeuralNet::new(4096, 256, 1);

    let epoch: i32 = 5000;
    let rand_len: usize = 296; // change
    for i in 0..epoch {
        let index = rand::thread_rng().gen_range(0..rand_len);
        network.train(data[index as usize].clone());

        if i % 10 == 0 {
            println!("{:.2}%", i as f32 / epoch as f32 * 100.0);
        }
    }

    let elapsed = start.elapsed();
    network.clock_epoch = elapsed;

    println!("elapsed: {:.2?}\n", elapsed);
    println!("\n{}", network.feed_foward(data[0][0].clone())[0]);
    println!("\n{}", network.feed_foward(data[120][0].clone())[0]);
    println!("\n{}", network.feed_foward(data[260][0].clone())[0]);
}

