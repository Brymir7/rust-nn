use crate::tensor::{Tensor, TensorHandle};
use byteorder::{BigEndian, ReadBytesExt};
use std::fs::{self, File};
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};

const TRAIN_IMAGES: &str = "train-images-idx3-ubyte";
const TRAIN_LABELS: &str = "train-labels-idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images-idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels-idx1-ubyte";

pub struct MnistDataset {
    pub images: Vec<Vec<f32>>,
    pub labels: Vec<u8>,
    image_size: usize,
}

impl MnistDataset {
    pub fn new(images: Vec<Vec<f32>>, labels: Vec<u8>, image_size: usize) -> Self {
        Self {
            images,
            labels,
            image_size,
        }
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn get_batch(&self, batch_indices: &[usize]) -> (TensorHandle, Vec<u8>) {
        let batch_size = batch_indices.len();

        let mut image_data = Vec::with_capacity(batch_size * self.image_size);
        for &idx in batch_indices {
            image_data.extend_from_slice(&self.images[idx]);
        }
        let image_tensor =
            Tensor::with_shape_f32(image_data, vec![batch_size, self.image_size], false);

        let labels: Vec<u8> = batch_indices.iter().map(|&idx| self.labels[idx]).collect();

        (image_tensor, labels)
    }
}

pub fn check_mnist_dataset() -> io::Result<PathBuf> {
    let data_dir = PathBuf::from("data/mnist");

    let files = [TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS];

    let all_exist = files.iter().all(|file| data_dir.join(file).exists());

    if all_exist {
        println!("MNIST dataset files found");
        return Ok(data_dir);
    }

    return Err(io::Error::new(
        io::ErrorKind::NotFound,
        "MNIST dataset files not found in data/mnist directory",
    ));
}

fn read_file(file_path: &Path) -> io::Result<Vec<u8>> {
    println!("Attempting to open file: {:?}", file_path);
    match fs::read(file_path) {
        Ok(data) => {
            println!("Read {} bytes from file", data.len());
            Ok(data)
        }
        Err(e) => {
            println!("Error reading file: {:?}", e);
            Err(e)
        }
    }
}
pub fn load_mnist_images(file_path: &Path, percentage: usize) -> io::Result<(Vec<Vec<f32>>, usize)> {
    let data = read_file(file_path)?;
    let mut reader = BufReader::new(&data[..]);

    let magic = reader.read_u32::<BigEndian>()?;
    if magic != 2051 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid magic number for images: {}", magic),
        ));
    }

    let num_images = reader.read_u32::<BigEndian>()? as usize * percentage / 100;
    let num_rows = reader.read_u32::<BigEndian>()? as usize;
    let num_cols = reader.read_u32::<BigEndian>()? as usize;
    let image_size = num_rows * num_cols;

    let mut images = Vec::with_capacity(num_images);
    let mut image_buffer = vec![0u8; image_size];

    for _ in 0..num_images {
        reader.read_exact(&mut image_buffer)?;

        let image_data: Vec<f32> = image_buffer
            .iter()
            .map(|&pixel| pixel as f32 / 255.0)
            .collect();

        images.push(image_data);
    }

    Ok((images, image_size))
}

pub fn load_mnist_labels(file_path: &Path) -> io::Result<Vec<u8>> {
    let data = read_file(file_path)?;
    let mut reader = BufReader::new(&data[..]);

    // Read header
    let magic = reader.read_u32::<BigEndian>()?;
    if magic != 2049 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid magic number for labels: {}", magic),
        ));
    }

    let num_labels = reader.read_u32::<BigEndian>()? as usize;

    let mut labels = vec![0u8; num_labels];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}
pub fn load_mnist_dataset(data_dir: &Path, percentage: usize) -> io::Result<(MnistDataset, MnistDataset)> {
    println!("Starting to load training images...");
    let train_images_path = data_dir.join(TRAIN_IMAGES);
    println!("Loading from path: {:?}", train_images_path);

    let (train_images, image_size) = match load_mnist_images(&train_images_path, percentage) {
        Ok(result) => { 
            println!("Successfully loaded {} training images", result.0.len());
            result
        }
        Err(e) => {
            println!("Error loading training images: {:?}", e);
            return Err(e);
        }
    };

    println!("Loading training labels...");
    let train_labels_path = data_dir.join(TRAIN_LABELS);
    let train_labels = match load_mnist_labels(&train_labels_path) {
        Ok(labels) => {
            println!("Successfully loaded {} training labels", labels.len());
            labels
        }
        Err(e) => {
            println!("Error loading training labels: {:?}", e);
            return Err(e);
        }
    };

    println!("Loading test images...");
    let test_images_path = data_dir.join(TEST_IMAGES);
    let (test_images, _) = match load_mnist_images(&test_images_path, percentage) {
        Ok(result) => {
            println!("Successfully loaded {} test images", result.0.len());
            result
        }
        Err(e) => {
            println!("Error loading test images: {:?}", e);
            return Err(e);
        }
    };

    println!("Loading test labels...");
    let test_labels_path = data_dir.join(TEST_LABELS);
    let test_labels = match load_mnist_labels(&test_labels_path) {
        Ok(labels) => {
            println!("Successfully loaded {} test labels", labels.len());
            labels
        }
        Err(e) => {
            println!("Error loading test labels: {:?}", e);
            return Err(e);
        }
    };

    let train_dataset = MnistDataset::new(train_images, train_labels, image_size);
    let test_dataset = MnistDataset::new(test_images, test_labels, image_size);

    println!("Dataset loading complete");
    Ok((train_dataset, test_dataset))
}
