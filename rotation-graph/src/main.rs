use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::thread;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    n: usize,

    /// Number of times to greet
    #[arg(short, long)]
    k: i64,

    #[arg(short, long)]
    t: f64,

    #[arg(short, long, default_value_t = 1)]
    p: usize,

    #[arg(short, long, default_value_t = String::new("result.json"))]
    file_name: String,
}

// generate all vectors in the n-dimensional space with absolute value of each element <= m
fn generate_vectors(n: usize, k: i64) -> Vec<Vec<i64>> {
    let mut vectors = Vec::new();
    let mut vector = vec![-k; n];
    let mut i;

    loop {
        vectors.push(vector.clone());
        i = 0;
        while i < n && vector[i] == k {
            vector[i] = -k;
            i += 1;
        }
        if i == n {
            break;
        }
        vector[i] += 1;
    }

    vectors
}

// Given n and p, split the workload of half of the n*n matrix (half by diagonal)
// into p parts, and return the indexes of the splitting rows
fn split_half_matrix(n: usize, p: usize) -> Vec<usize> {
    let mut indexes = vec![]; // indexes of the splitting rows
    let len = n * (n + 1) / 2;
    let avg = len / p;
    let mut current_row = 0;
    let mut current_row_len = n;
    let mut flag = p;
    for i in 0..p {
        indexes.push(current_row);
        let mut current_sum = current_row_len;
        current_row += 1;
        current_row_len -= 1;
        if current_row == n {
            // You don't need that many processes
            flag = i;
            break;
        }
        let lim = avg * (i + 1);
        while current_sum < lim {
            current_sum += current_row_len;
            current_row += 1;
            current_row_len -= 1;
        }
    }
    for _ in flag..p + 1 {
        indexes.push(n);
    }
    indexes
}

// helper function, convert a vector of tuples to a hashmap
fn vec_to_map(v: &Vec<(usize, usize)>) -> HashMap<usize, HashSet<usize>> {
    let mut hashmap: HashMap<usize, HashSet<usize>> = HashMap::new();

    for (key, value) in v.iter() {
        hashmap.entry(*key).or_insert(HashSet::new()).insert(*value);
    }

    hashmap
}

// helper function, get the vector at index
fn index_to_vec(vector: &Vec<Vec<i64>>, index: usize) -> Vec<i64> {
    vector[index].clone()
}

#[allow(dead_code)]
fn output_to_file(v: &Vec<(usize, usize)>, file_path: &str) {
    let hashmap = vec_to_map(v);
    let serialized = serde_json::to_string(&hashmap).expect("Serialization failed");
    
    let mut file = File::create(file_path).expect("File creation failed");
    file.write_all(serialized.as_bytes()).expect("Write failed");
}

// output to a file in dict format
fn output_to_file_readable(v: &Vec<(usize, usize)>, vector: &Vec<Vec<i64>>, file_path: &str) {
    let hashmap = vec_to_map(v);
    let hashmap: HashMap<String, Vec<Vec<i64>>> = hashmap.iter().map(|(key, value)| {
        (
            format!("{:?}",index_to_vec(vector, *key)),
            value.iter().map(|x| index_to_vec(vector, *x)).collect::<Vec<_>>(),
        )
    }).collect();

    let mut file = File::create(file_path).expect("File creation failed");
    let serialized = serde_json::to_string_pretty(&hashmap).expect("Serialization failed");
    file.write_all(serialized.as_bytes()).expect("Write failed");
}

fn main() {
    let args = Args::parse();
    let (n, k, p, t) = (args.n, args.k, args.p, args.t);
    let pb = Arc::new(ProgressBar::new(p as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} (ETA {eta})")
            .unwrap()
            .progress_chars("=> "),
    );
    let vectors = generate_vectors(n, k);
    let vectors = Arc::new(vectors);
    let split_rows = split_half_matrix(vectors.len() as usize, p as usize);
    let split_rows = Arc::new(split_rows);
    // Now t is in angle, convert it to radian, then compute the cosine
    let t = t.to_radians().cos();
    println!("t = {}", t);

    let mut receivers = vec![];
    for tid in 0..p {
        let (sender, receiver) = std::sync::mpsc::channel();
        let pb = pb.clone();
        let split_indexes = split_rows.clone();
        let vectors = vectors.clone();

        let _ = thread::spawn(move || {
            let mut result = Vec::new();
            for i in split_indexes[tid]..split_indexes[tid + 1] {
                for j in 0..vectors.len() {
                    if j < i {
                        continue;
                    }
                    // compute the angle between vectors[i] and vectors[j]
                    let mut dot_product = 0;
                    let mut norm1 = 0;
                    let mut norm2 = 0;
                    for k in 0..n {
                        dot_product += vectors[i][k] * vectors[j][k];
                        norm1 += vectors[i][k] * vectors[i][k];
                        norm2 += vectors[j][k] * vectors[j][k];
                    }
                    let angle = dot_product as f64 / (norm1 as f64 * norm2 as f64).sqrt();
                    if angle >= t {
                        // println!("{:?} {:?}", index_to_vec(&vectors, i), index_to_vec(&vectors, j));
                        result.push((i, j));
                    }
                    pb.inc(1);
                }
            }
            sender.send(result).unwrap();
        });

        receivers.push(receiver);
    }

    let mut result = Vec::new();
    for r in receivers {
        result.append(&mut r.recv().unwrap());
    }
    pb.finish();

    output_to_file_readable(&result, &vectors, "result.txt");
}
