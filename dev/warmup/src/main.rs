
use candle_core::{ /* DType, */ Device, Tensor};

fn tensorfun() -> Result<(), Box<dyn std::error::Error>> {

    let data: [f32; 3] = [1.0f32, 2., 3.];
    println!("data: {:?}", data);

    let t1 = Tensor::new(&data, &Device::Cpu)?;
    println!("t1: {:?}", t1.to_vec1::<f32>()?);

    let t2 = Tensor::new(&data, &Device::Cpu)?;
    println!("t2: {:?}", t2.to_vec1::<f32>()?);

    let t3 = t1.add(&t2)?;
    println!("t3: {:?}", t3.to_vec1::<f32>()?);

    let t4 = t1.mul(&t2)?;
    println!("t4: {:?}", t4.to_vec1::<f32>()?);

    let zero_tensor = t4.zeros_like()?;
    println!("zero_tensor: {:?}", zero_tensor.to_vec1::<f32>()?);

    let ones_tensor = t4.ones_like()?;
    println!("ones_tensor: {:?}", ones_tensor.to_vec1::<f32>()?);

    let random_tensor = t4.rand_like(0.0, 1.0)?;
    println!("random_tensor: {:?}", random_tensor.to_vec1::<f32>()?);


    let nested_data: [[f32; 3]; 3] = [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]];
    let nested_tensor = Tensor::new(&nested_data, &Device::Cpu)?;
    println!("nested_tensor: {:?}", nested_tensor.to_vec2::<f32>()?);


    Ok(())

}

fn main() {

    println!("Let's have fun with Tensors!");

    let _ = tensorfun();

}
