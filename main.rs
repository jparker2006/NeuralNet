fn main() {
    println!("HELLO RUST");
    let mut vec: Vec<i32> = Vec::new();

    for i in 0..10 {
        vec.push(i);
    }

    for i in 0..vec.len() {
        println!("{}", vec[i]);
    }


    let mut ex = Matrix {
        row: 3,
        col: 4
//         data: Vec::new()
    };
}

pub struct Matrix {
    row: i32,
    col: i32
//     data: Vec<f32>
}

impl Matrix {
    pub fn new() -> Matrix {
        Matrix {row: 4, col: 3}
    }
}
