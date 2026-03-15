use std::path::PathBuf;

fn find_libtorch_lib_dir() -> PathBuf {
    // 1. LIBTORCH environment variable (explicit override)
    if let Ok(libtorch) = std::env::var("LIBTORCH") {
        let lib_dir = PathBuf::from(libtorch).join("lib");
        if lib_dir.exists() {
            return lib_dir;
        }
    }

    // 2. LIBTORCH_USE_PYTORCH: get path from Python's torch package
    if std::env::var("LIBTORCH_USE_PYTORCH").is_ok() {
        if let Ok(output) = std::process::Command::new("python3")
            .args(["-c", "import torch; print(torch.__path__[0])"])
            .output()
        {
            let torch_path = String::from_utf8(output.stdout).unwrap().trim().to_string();
            let lib_dir = PathBuf::from(&torch_path).join("lib");
            if lib_dir.exists() {
                return lib_dir;
            }
        }
    }

    // 3. DEP_TCH_LIBTORCH_LIB: lib path exported by torch-sys via links="tch"
    //    Build order is guaranteed because torch-sys is in [dependencies],
    //    so the download completes before this build.rs runs
    if let Ok(lib_dir) = std::env::var("DEP_TCH_LIBTORCH_LIB") {
        let lib_dir = PathBuf::from(lib_dir);
        if lib_dir.exists() {
            return lib_dir;
        }
    }

    panic!("Could not find libtorch lib directory. Set LIBTORCH or enable download-libtorch feature.");
}

fn main() {
    let libtorch_lib_dir = find_libtorch_lib_dir();
    let libtorch = libtorch_lib_dir.parent().unwrap().to_path_buf();

    // Compile mps_helper.cpp
    let include1 = libtorch.join("include");
    let include2 = libtorch
        .join("include")
        .join("torch")
        .join("csrc")
        .join("api")
        .join("include");
    cc::Build::new()
        .cpp(true)
        .file("csrc/mps_helper.cpp")
        .include(&include1)
        .include(&include2)
        .flag("-std=c++17")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-missing-field-initializers")
        .compile("mps_helper");

    // macOS: add @executable_path/lib to rpath
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/lib");

    // Copy dylibs to lib/ next to the binary
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // OUT_DIR = target/<profile>/build/<crate>/out
    // ancestors: out -> <crate> -> build -> <profile>
    let target_profile_dir = out_dir.ancestors().nth(3).unwrap();
    let target_lib_dir = target_profile_dir.join("lib");
    std::fs::create_dir_all(&target_lib_dir).ok();

    let required_dylibs = [
        "libtorch_cpu.dylib",
        "libtorch.dylib",
        "libc10.dylib",
        "libomp.dylib",
    ];
    for name in &required_dylibs {
        let src = libtorch_lib_dir.join(name);
        if src.exists() {
            let dst = target_lib_dir.join(name);
            std::fs::copy(&src, &dst).ok();
        }
    }
}
