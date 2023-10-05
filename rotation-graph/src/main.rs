use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::{thread, vec};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    n: usize,

    #[arg(short, long)]
    k: i64,

    #[arg(short, long)]
    t: f64,

    #[arg(short, long, default_value_t = 1)]
    p: usize,

    #[arg(short, long, default_value = "result.json")]
    file_name: String,
}

// generate all vectors in the n-dimensional space with absolute value of each element <= m
fn generate_vectors(n: usize, k: i64, base:i64) -> Vec<Vec<i64>> {
    let mut vectors = Vec::new();
    let mut vector = vec![base; n];
    let mut i;

    loop {
        // push if gcd(vector) == 1
        let mut gcd = vector[0];
        let mut flag = if vector[0] == 0 { true } else { false };
        for i in 1..n {
            gcd = num::integer::gcd(gcd, vector[i]);
            if vector[i] != 0 {
                flag = false;
            }
        }
        if gcd == 1 && (!flag) {
            vectors.push(vector.clone());
        }
        i = 0;
        while i < n && vector[i] == k {
            vector[i] = base;
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
    let mut indexes = vec![0]; // indexes of the splitting rows
    let len = n * (n + 1) / 2;
    let avg = (len as f64 / p as f64).ceil() as usize;
    let mut current_row = 0;
    let mut current_row_len = n;

    for _ in 0..p {
        if current_row >= n {
            break;
        }

        let mut current_sum = current_row_len;
        current_row += 1;
        current_row_len -= 1;
        while current_sum < avg {
            if current_row >= n {
                break;
            }
            current_sum += current_row_len;
            current_row += 1;
            current_row_len -= 1;
        }
        indexes.push(current_row);
    }
    let len = indexes.len();
    for _ in len..p + 1 {
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
fn output_to_file(v: &Vec<(usize, usize)>, file_path: &String) {
    let hashmap = vec_to_map(v);
    let serialized = serde_json::to_string(&hashmap).expect("Serialization failed");

    let mut file = File::create(file_path).expect("File creation failed");
    file.write_all(serialized.as_bytes()).expect("Write failed");
}

// output to a file in dict format
fn output_to_file_readable(
    v: &Vec<(usize, usize)>,
    vectors: &Vec<Vec<i64>>,
    snd_vectors: &Vec<Vec<i64>>,
    file_path: &str,
) {
    let hashmap = vec_to_map(v);
    let hashmap: HashMap<String, Vec<Vec<i64>>> = hashmap
        .iter()
        .map(|(key, value)| {
            (
                format!("{:?}", index_to_vec(vectors, *key)),
                value
                    .iter()
                    .map(|x| index_to_vec(snd_vectors, *x))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    let mut file = File::create(file_path).expect("File creation failed");
    let serialized = serde_json::to_string_pretty(&hashmap).expect("Serialization failed");
    file.write_all(serialized.as_bytes()).expect("Write failed");
}


fn main() {
    let args = Args::parse();
    let (n, k, p, t) = (args.n, args.k, args.p, args.t);

    let vectors = generate_vectors(n, k, 0);
    let vectors = Arc::new(vectors);
    let split_rows = split_half_matrix(vectors.len() as usize, p as usize);
    let split_rows = Arc::new(split_rows);
    let snd_vectors = generate_vectors(n, k, -k);
    let snd_vectors = Arc::new(snd_vectors);
    // Now t is in angle, convert it to radian, then compute the cosine
    let t = t.to_radians().cos();
    println!("t = {}", t);

    let mut receivers = vec![];
    for tid in 0..p {
        let (sender, receiver) = std::sync::mpsc::channel();
        let split_indexes = split_rows.clone();
        let vectors = vectors.clone();
        let snd_vectors = snd_vectors.clone();

        let _ = thread::spawn(move || {
            let len_pb = (split_indexes[tid + 1] - split_indexes[tid]) * snd_vectors.len();
            let pb = ProgressBar::new(len_pb as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] [{bar:40}] {pos}/{len} (ETA {eta})")
                    .unwrap()
                    .progress_chars("=> "),
            );

            let mut result = Vec::new();
            for i in split_indexes[tid]..split_indexes[tid + 1] {
                for j in 0..snd_vectors.len() {
                    // if j < i {
                    // continue;
                    // }
                    // compute the angle between vectors[i] and vectors[j]
                    let mut dot_product = 0;
                    let mut norm1 = 0;
                    let mut norm2 = 0;
                    for k in 0..n {
                        dot_product += vectors[i][k] * snd_vectors[j][k];
                        norm1 += vectors[i][k] * vectors[i][k];
                        norm2 += snd_vectors[j][k] * snd_vectors[j][k];
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
            pb.finish();
        });

        receivers.push(receiver);
    }

    let mut result = Vec::new();
    for r in receivers {
        result.append(&mut r.recv().unwrap());
    }

    output_to_file_readable(&result, &vectors, &snd_vectors, "result.txt");
}


mod test {
    #[test]
    fn test_split_half_matrix() {
        fn check_valid_indexes(p: usize, n: usize, v: &Vec<usize>) -> bool {
            let nn = n;
            let n = n * (n + 1) / 2;
            let avg: usize = n / p;
            if v.len() != p + 1 {
                println!("1:{:?} {} {}", v, p, n);
                return false;
            }
            if v[0] != 0 {
                println!("2:{:?} {} {}", v, p, n);
                return false;
            }
            if v[p] != nn {
                println!("3:{:?} nn:{}, p:{} n:{} avg:{}", v, nn, p, n, avg);
                return false;
            }
            for i in 0..p {
                if v[i] > v[i + 1] {
                    println!("1:{:?} {} {} {}", v, v[i], v[i + 1], n);
                    return false;
                }
                if (nn - (v[i + 1] - 1) + nn - v[i]) * (v[i + 1] - v[i]) / 2 > avg + nn {
                    println!("2:{:?} {} {} {}, {}, nn: {}", v, v[i], v[i + 1], n, avg, nn);
                    return false;
                }
            }

            true
        }
        // Here n is the len of generated vectors, p is the number of processes
        for n in 1..25 * 25 * 25 * 25 * 25 {
            for p in 1..48 {
                let v = super::split_half_matrix(n, p);
                assert!(check_valid_indexes(p, n, &v));
            }
        }
    }
}
