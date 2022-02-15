// tester

fn main() {
    let mut mat = Matrix::new(2, 3);
    let mut mat2 = Matrix::new(3, 2);

    mat.data = vec![
        vec![1.0,2.0,3.0],
        vec![3.0,2.0,1.0]
    ];

    mat2.data = vec![
        vec![1.0,2.0],
        vec![3.0,2.0],
        vec![2.0,7.0]
    ];

    let mut mat3 = mat.dot_product(mat2);
    mat3.print();
}

pub struct Matrix {
    rows: i32,
    cols: i32,
    data: Vec<Vec<f32>>
}

impl Matrix {
    pub fn new(r: i32, c: i32) -> Self {
        let mut mat = Matrix {
            rows: r,
            cols: c,
            data: Vec::new()
        };
        for _x in 0..r {
            let mut v: Vec<f32> = Vec::new();
            for _y in 0..c {
                v.push(0.0);
            }
            mat.data.push(v);
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

    // scalar functions
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

    // element wise functions (try and combine sub add, mult div)
    pub fn e_add(&mut self, m: Matrix) {
        if m.cols != self.cols || m.rows != self.rows {
            println!("cannot perform element wise operation");
            return;
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

    pub fn transpose(&mut self) -> Matrix {
        let mut mat = Matrix {
            rows: self.cols,
            cols: self.rows,
            data: Vec::new()
        };
        for x in 0..self.cols {
            let mut v: Vec<f32> = Vec::new();
            for y in 0..self.rows {
                v.push(self.data[y as usize][x as usize]);
            }
            mat.data.push(v);
        }
        return mat;
    }

    pub fn dot_product(&mut self, m: Matrix) -> Matrix {
        let mut nm = Matrix {
            rows: 0, // figure out
            cols: 0,
            data: Vec::new()
        };

        if self.data[0].len() != m.data.len() {
            println!("dot product could not be performed");
            return nm
        }

        for x in 0..self.data.len() {
            for y in 0..m.data[0].len() {
                let mut sum: f32 = 0.0;
                for z in 0..self.data[0].len() {
                    sum += self.data[x][z] * m.data[z][y];
                }
                nm.data.push(Vec::new());
                nm.data[x].push(sum);
            }
        }
        nm.rows = nm.data.len() as i32;
        nm.cols = nm.data[0].len() as i32;
        return nm
    }
}
