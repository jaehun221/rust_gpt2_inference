use std::fs::read;
use safetensors::SafeTensors;
use ndarray::{ Array1, Array2 };

#[derive(Debug)]
pub struct Weights {
    w: Vec<u8>,
}

impl Weights {
    pub fn weights_load(weights_path: &str) -> Self {
        Self { w: read(weights_path).expect("file open failed") }
    }

    pub fn get_tensor1(&self, name: &str) -> Array1<f32> {
        let tensors = SafeTensors::deserialize(&self.w).expect("deserialize failed");
        let tensor = tensors.tensor(name).expect("tensor not found");

        let v: Vec<f32> = tensor.data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("u8 -> f32 failed")))
            .collect();

        Array1::from_shape_vec(tensor.shape()[0], v).expect("Array1 shape err")
    }

    pub fn get_tensor2(&self, name: &str) -> Array2<f32> {
        let tensors = SafeTensors::deserialize(&self.w).expect("deserialize failed");
        let tensor = tensors.tensor(name).expect("tensor not found");

        let v: Vec<f32> = tensor.data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("u8 -> f32 failed")))
            .collect();

        Array2::from_shape_vec((tensor.shape()[0], tensor.shape()[1]), v).expect("Array2 shape err")
    }

    pub fn tensor_list(&self) {
        let tensors = SafeTensors::deserialize(&self.w).expect("deserialize failed");
        
        let mut t_list = tensors.names();
        t_list.sort();
        for i in t_list {
            println!("{}", i);
        }
    }

}