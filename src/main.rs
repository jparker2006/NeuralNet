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
    pub fn e_div(&mut self, m: &Matrix) -> &mut Matrix {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
        }
        for x in 0..self.data.len() {
            for y in 0..self.data[x].len() {
                self.data[x][y] /= m.data[x][y];
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

pub struct NeuralNet {
    ih_weight: Matrix,
    ho_weight: Matrix,
    h_bias: Matrix,
    o_bias: Matrix,
    learning_rate: f32
}

impl NeuralNet {
    pub fn new(i: i32, h: i32, o: i32) -> Self {
        NeuralNet {
            ih_weight: Matrix::r_new(i, h),
            ho_weight: Matrix::r_new(o, h),
            h_bias: Matrix::r_new(h, 1),
            o_bias: Matrix::r_new(o, 1),
            learning_rate: 0.1
        }
    }

    pub fn feed_foward(&mut self, input_data: Vec<f32>) -> Vec<f32> {
        let m_input_data = Matrix::from_vector_new(input_data);
        // i -> h layer
        let mut m_hidden: Matrix = self.ih_weight.dot_product(&m_input_data);
        m_hidden.e_add(&self.h_bias);
        m_hidden.map_to_function(&sigmoid);
        // h -> o layer
        let mut m_output: Matrix = self.ho_weight.dot_product(&m_hidden);
        m_output.e_add(&self.o_bias);
        m_output.map_to_function(&sigmoid);
        return m_output.to_vector()
    }

    pub fn train(&mut self, input_data: Vec<f32>, v_target_data: Vec<f32>) {
        // feeding foward (refactor)
        // i -> h layer
        let m_input_data = Matrix::from_vector_new(input_data);
        let mut m_hidden: Matrix = self.ih_weight.dot_product(&m_input_data);
        m_hidden.e_add(&self.h_bias);
        m_hidden.map_to_function(&sigmoid);

        // h -> o layer
        let mut output_data: Matrix = self.ho_weight.dot_product(&m_hidden);
        output_data.e_add(&self.o_bias);
        output_data.map_to_function(&sigmoid);

        let mut m_target_data = Matrix::from_vector_new(v_target_data);
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
}

pub fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + f32::exp(-x));
}

pub fn d_sigmoid(x: f32) -> f32 {
    return x * (1.0 - x);
}

fn main() {
    let mut network = NeuralNet::new(2, 2, 1);
    // xor data set
    let data = vec![vec![1.0, 1.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
    for _i in 0..50000 {
        let index = rand::thread_rng().gen_range(0..4);
        network.train(data[index as usize].clone(), targets[index as usize].clone()); // fix to vector ! impl copy
    }
    Matrix::from_vector_new(network.feed_foward(vec![1.0, 0.0])).print();
    Matrix::from_vector_new(network.feed_foward(vec![0.0, 1.0])).print();
    Matrix::from_vector_new(network.feed_foward(vec![1.0, 1.0])).print();
    Matrix::from_vector_new(network.feed_foward(vec![0.0, 0.0])).print();
}








