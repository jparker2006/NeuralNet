extern crate rand;
use rand::Rng;

pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    pub data: Vec<Vec<f32>>
}

impl Matrix {
    // constructors
    pub fn d_new(d: Vec<Vec<f32>>) -> Self { // new defined matrix
        let mat = Matrix {
            rows: d.len() as i32,
            cols: d[0].len() as i32,
            data: d
        };
        return mat
    }
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
    pub fn s_add(&mut self, n: f32) { // also sub
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] += n;
            }
        }
    }
    pub fn s_mult(&mut self, n: f32) { // also div
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] *= n;
            }
        }
    }

    // element wise operations
    pub fn e_add(&mut self, m: &Matrix) { // passing as reference (might need to take out later)
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
            return
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] += m.data[x][y];
            }
        }
    }
    pub fn e_sub(&mut self, m: Matrix) {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
            return;
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] -= m.data[x][y];
            }
        }
    }
    pub fn e_mult(&mut self, m: Matrix) {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
            return;
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] *= m.data[x][y];
            }
        }
    }
    pub fn e_div(&mut self, m: Matrix) {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
            return;
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] /= m.data[x][y];
            }
        }
    }

    pub fn transpose(&mut self) {
        let mut new_data = Vec::new();
        for x in 0..self.cols {
            let mut v: Vec<f32> = Vec::new();
            for y in 0..self.rows {
                v.push(self.data[y as usize][x as usize]);
            }
            new_data.push(v);
        }
        self.data = new_data;
        let t_rows = self.rows;
        self.rows = self.cols;
        self.cols = t_rows;
    }

    pub fn dot_product(&mut self, m: Matrix) -> Matrix {
        let mut nm = Matrix {
            rows: m.cols,
            cols: self.rows,
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

    pub fn randomize(&mut self) { // might have error
        let mut rng = rand::thread_rng();
        for _x in 0..self.rows {
            let mut vector: Vec<f32> = Vec::new();
            for _y in 0..self.cols {
                vector.push(rng.gen::<f32>() * 2.0 - 1.0);
            }
            self.data.push(vector);
        }
    }

    pub fn map_to_sigmoid(&mut self) { // might be outta bounds
        for x in 0..self.rows {
            for y in 0..self.cols {
                self.data[x as usize][y as usize] = sigmoid(self.data[x as usize][y as usize]);
            }
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

pub struct NeuralNet {
    i_nodes: i32,
    h_nodes: i32,
    o_nodes: i32,
    ih_weight: Matrix,
    ho_weight: Matrix,
    h_bias: Matrix,
    o_bias: Matrix
}

impl NeuralNet {
    pub fn new(i: i32, h: i32, o: i32) -> Self {
        NeuralNet {
            i_nodes: i,
            h_nodes: h,
            o_nodes: o,
            ih_weight: Matrix::r_new(i, h),
            ho_weight: Matrix::r_new(o, h),
            h_bias: Matrix::r_new(h, 1),
            o_bias: Matrix::r_new(o, 1)
        }
    }

    pub fn feed_foward(&mut self, input_data: Vec<f32>) -> Vec<f32> {
        // i -> h layer
        let m_input_data = Matrix::from_vector_new(input_data);
        let mut m_hidden: Matrix = self.ih_weight.dot_product(m_input_data);

        m_hidden.print();
        self.h_bias.print();

        println!("{}", &self.h_bias.rows);
        println!("{}", &self.h_bias.cols);
        println!("{}", m_hidden.rows);
        println!("{}", m_hidden.cols);

        m_hidden.e_add(&self.h_bias); // issue
//         m_hidden.map_to_sigmoid();

        // h -> o layer
        let mut m_output = self.ho_weight.dot_product(m_hidden);
        m_output.e_add(&self.o_bias);
//         m_output.map_to_sigmoid();
        return m_output.to_vector()
    }
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-x));
}

fn main() {
    let mut net = NeuralNet::new(2, 2, 1);
    let input = vec![1.0, 0.0];
    let output = net.feed_foward(input);
    println!("{}", output[0]);
}

