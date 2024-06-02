use std::env;
use std::path::PathBuf;

fn main() {
    let lib_path = PathBuf::from("../../build/src")
        .canonicalize()
        .expect("cannot canonicalize path");
    let include_path = PathBuf::from("../../src")
        .join("include")
        .canonicalize()
        .expect("cannot canonicalize path");
    let lib_dir = lib_path.to_str().unwrap();
    let include_dir = include_path.to_str().unwrap();

    println!("cargo:rustc-link-search={}", lib_dir);
    println!("cargo:rustc-link-search=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=chiavec");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=stdc++");

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", include_dir))
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");
}
