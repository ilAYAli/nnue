#[cfg(not(feature = "cuda"))]
compile_error!("build tools/bullet/train with --features cuda");

use std::{
    env,
    ffi::OsStr,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    mem,
    path::{Path, PathBuf},
    process::{self, Command},
    time::Instant,
};

use bullet_lib::{
    game::formats::bulletformat::ChessBoard,
    value::loader::{
        DataLoader, SfBinpackLoader,
        sfbinpack::{MoveType, PieceType, TrainingDataEntry},
    },
};
use serde_json::Value;

mod trainer_main {
    include!("../main.rs");

    pub fn run_from_env() {
        main();
    }
}

const FORBIDDEN_KEYS: &[&str] = &[
    "init_weights",
    "bullet_init_weights",
    "init_from_nn",
    "export_init_only",
];

const FORBIDDEN_TEXT: &[&str] = &[
    "default.net",
    "berserk",
    "native-28",
    "native-29",
    "weights.bin",
];

const ENYO_SUPPORTED_INPUT_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 32];
const ENYO_SUPPORTED_FEATURE_CHANNELS: &[usize] = &[11, 12];
const ENYO_SUPPORTED_OUTPUT_BUCKETS: &[usize] = &[1, 2, 4, 8];
const ENYO_LEGACY_BUCKET_FOR_32: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 8, 9, 10, 11,
    12, 12, 13, 13, 12, 12, 13, 13,
    14, 14, 15, 15, 14, 14, 15, 15,
];

const TRAINING_DEFAULT_KEYS: &[&str] = &[
    "loader",
    "net_id",
    "batches",
    "batch_size",
    "superbatches",
    "threads",
    "wdl",
    "lr",
    "final_lr",
    "save_rate",
    "trainable",
    "weight_decay",
    "sfbinpack",
];

const SFBINPACK_DEFAULT_KEYS: &[&str] =
    &["buffer_mb", "offset", "min_ply", "max_abs_cp", "quiet_only"];

const BUILD_METADATA_KEYS: &[&str] = &[
    "run",
    "lineage",
    "continue_from",
    "init_net",
    "reference",
    "hypothesis",
    "data",
    "net",
    "out",
    "changed_variables",
];

const BUILD_DATA_KEYS: &[&str] = &[
    "source_binpack",
    "bullet_output",
    "limit",
    "offset",
    "threads",
];

#[derive(Clone)]
struct Config {
    arch: Value,
    defaults: Value,
    build: Value,
}

#[derive(Clone)]
struct DataConfig {
    source_binpack: String,
    bullet_output: String,
    offset: u64,
    limit: u64,
    threads: usize,
    buffer_mb: usize,
    min_ply: u16,
    max_abs_cp: u32,
    quiet_only: bool,
}

#[derive(Clone, Copy)]
struct Filter {
    min_ply: u16,
    max_abs_cp: u32,
    quiet_only: bool,
}

impl Filter {
    fn keep(self, entry: &TrainingDataEntry) -> bool {
        if entry.ply < self.min_ply {
            return false;
        }
        if i32::from(entry.score).unsigned_abs() > self.max_abs_cp {
            return false;
        }
        if self.quiet_only
            && (entry.mv.mtype() != MoveType::Normal
                || entry.pos.piece_at(entry.mv.to()).piece_type() != PieceType::None)
        {
            return false;
        }
        if entry.pos.is_checked(entry.pos.side_to_move()) {
            return false;
        }
        true
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: tools/bullet/train <plan|data|run|export|all> \
         [--build build.json] [--arch architecture.json] [--defaults defaults.json] [--force]"
    );
    process::exit(2);
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("unexpected crate path")
        .to_owned()
}

fn expand_path(value: &str) -> PathBuf {
    let expanded = if let Some(rest) = value.strip_prefix("~/") {
        match env::var("HOME") {
            Ok(home) => format!("{home}/{rest}"),
            Err(_) => value.to_owned(),
        }
    } else {
        value.to_owned()
    };
    let expanded = env::vars().fold(expanded, |text, (key, val)| {
        text.replace(&format!("${key}"), &val)
            .replace(&format!("${{{key}}}"), &val)
    });
    let path = PathBuf::from(expanded);
    if path.is_absolute() {
        path
    } else {
        root().join(path)
    }
}

fn load_json(path: &str) -> Value {
    let path = expand_path(path);
    let text = fs::read_to_string(&path).unwrap_or_else(|err| {
        eprintln!("error: cannot read {}: {err}", path.display());
        process::exit(1);
    });
    serde_json::from_str(&text).unwrap_or_else(|err| {
        eprintln!("error: invalid JSON in {}: {err}", path.display());
        process::exit(1);
    })
}

fn string_at<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn required_string(value: &Value, key: &str) -> String {
    string_at(value, key)
        .unwrap_or_else(|| {
            eprintln!("error: missing string key `{key}`");
            process::exit(1);
        })
        .to_owned()
}

fn usize_at(value: &Value, key: &str, default: usize) -> usize {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn u64_at(value: &Value, key: &str, default: u64) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn f64_at(value: &Value, key: &str, default: f64) -> f64 {
    value.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn bool_at(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn object_at<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or(&Value::Null)
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn config_type_error(path: &str, expected: &str, value: &Value) -> ! {
    eprintln!("error: {path} must be {expected}, got {}", json_kind(value));
    process::exit(2);
}

fn same_scalar_kind(left: &Value, right: &Value) -> bool {
    matches!(
        (left, right),
        (Value::Bool(_), Value::Bool(_))
            | (Value::Number(_), Value::Number(_))
            | (Value::String(_), Value::String(_))
            | (Value::Array(_), Value::Array(_))
            | (Value::Null, Value::Null)
    )
}

fn validate_override_shape(
    build_value: &Value,
    default_value: &Value,
    path: &str,
    errors: &mut Vec<String>,
) {
    match (build_value, default_value) {
        (Value::Object(build), Value::Object(defaults)) => {
            for (key, child) in build {
                let child_path = format!("{path}.{key}");
                match defaults.get(key) {
                    Some(default_child) => {
                        validate_override_shape(child, default_child, &child_path, errors)
                    }
                    None => errors.push(format!(
                        "{child_path}: build override has no matching defaults.json key"
                    )),
                }
            }
        }
        (Value::Object(_), _) | (_, Value::Object(_)) => errors.push(format!(
            "{path}: build/defaults shape mismatch (build {}, defaults {})",
            json_kind(build_value),
            json_kind(default_value)
        )),
        _ if !same_scalar_kind(build_value, default_value) => errors.push(format!(
            "{path}: build/defaults type mismatch (build {}, defaults {})",
            json_kind(build_value),
            json_kind(default_value)
        )),
        _ => {}
    }
}

fn validate_build_data(value: &Value, errors: &mut Vec<String>) {
    let Some(data) = value.as_object() else {
        errors.push(format!(
            "build.data: expected object, got {}",
            json_kind(value)
        ));
        return;
    };
    for key in data.keys() {
        if !BUILD_DATA_KEYS.contains(&key.as_str()) {
            errors.push(format!(
                "build.data.{key}: not an allowed data key; put Bullet training knobs in build.sfbinpack or top-level build.json"
            ));
        }
    }
}

fn config_contract_errors(config: &Config) -> Vec<String> {
    let mut errors = Vec::new();
    for key in TRAINING_DEFAULT_KEYS {
        if config.defaults.get(*key).is_none() {
            errors.push(format!(
                "defaults.{key}: missing Bullet training parameter default"
            ));
        }
    }
    let sfbinpack = object_at(&config.defaults, "sfbinpack");
    if !sfbinpack.is_object() {
        errors.push(format!(
            "defaults.sfbinpack: expected object, got {}",
            json_kind(sfbinpack)
        ));
    } else {
        for key in SFBINPACK_DEFAULT_KEYS {
            if sfbinpack.get(*key).is_none() {
                errors.push(format!(
                    "defaults.sfbinpack.{key}: missing Bullet training parameter default"
                ));
            }
        }
    }

    let Some(build) = config.build.as_object() else {
        errors.push(format!(
            "build.json root must be object, got {}",
            json_kind(&config.build)
        ));
        return errors;
    };
    let Some(defaults) = config.defaults.as_object() else {
        errors.push(format!(
            "defaults.json root must be object, got {}",
            json_kind(&config.defaults)
        ));
        return errors;
    };

    for (key, value) in build {
        if BUILD_METADATA_KEYS.contains(&key.as_str()) {
            if key == "data" {
                validate_build_data(value, &mut errors);
            }
            continue;
        }
        match defaults.get(key) {
            Some(default_value) => {
                validate_override_shape(value, default_value, &format!("build.{key}"), &mut errors)
            }
            None => errors.push(format!(
                "build.{key}: no matching defaults.json key and not allowed build metadata"
            )),
        }
    }
    errors
}

fn validate_config_contract(config: &Config) {
    let errors = config_contract_errors(config);
    if !errors.is_empty() {
        eprintln!("config contract rejected:");
        for error in errors {
            eprintln!("  {error}");
        }
        process::exit(2);
    }
}

fn resolved_training_value<'a>(config: &'a Config, key: &str) -> Option<(&'a Value, String)> {
    if let Some(value) = config.build.get(key) {
        return Some((value, format!("build.{key}")));
    }
    config
        .defaults
        .get(key)
        .map(|value| (value, format!("defaults.{key}")))
}

fn resolved_training_nested_value<'a>(
    config: &'a Config,
    parent: &str,
    key: &str,
) -> Option<(&'a Value, String)> {
    if let Some(parent_value) = config.build.get(parent) {
        let path = format!("build.{parent}");
        let object = parent_value
            .as_object()
            .unwrap_or_else(|| config_type_error(&path, "object", parent_value));
        if let Some(value) = object.get(key) {
            return Some((value, format!("{path}.{key}")));
        }
    }
    let parent_value = config.defaults.get(parent)?;
    let path = format!("defaults.{parent}");
    let object = parent_value
        .as_object()
        .unwrap_or_else(|| config_type_error(&path, "object", parent_value));
    object
        .get(key)
        .map(|value| (value, format!("{path}.{key}")))
}

fn value_string(value: &Value, path: &str) -> String {
    value
        .as_str()
        .unwrap_or_else(|| config_type_error(path, "string", value))
        .to_owned()
}

fn value_usize(value: &Value, path: &str) -> usize {
    value
        .as_u64()
        .map(|v| v as usize)
        .unwrap_or_else(|| config_type_error(path, "integer", value))
}

fn value_u64(value: &Value, path: &str) -> u64 {
    value
        .as_u64()
        .unwrap_or_else(|| config_type_error(path, "integer", value))
}

fn value_f64(value: &Value, path: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| config_type_error(path, "number", value))
}

fn value_bool(value: &Value, path: &str) -> bool {
    value
        .as_bool()
        .unwrap_or_else(|| config_type_error(path, "bool", value))
}

fn training_string(config: &Config, key: &str, default: &str) -> String {
    resolved_training_value(config, key)
        .map(|(value, path)| value_string(value, &path))
        .unwrap_or_else(|| default.to_owned())
}

fn training_usize(config: &Config, key: &str, default: usize) -> usize {
    resolved_training_value(config, key)
        .map(|(value, path)| value_usize(value, &path))
        .unwrap_or(default)
}

fn training_f64(config: &Config, key: &str, default: f64) -> f64 {
    resolved_training_value(config, key)
        .map(|(value, path)| value_f64(value, &path))
        .unwrap_or(default)
}

fn training_nested_usize(config: &Config, parent: &str, key: &str, default: usize) -> usize {
    resolved_training_nested_value(config, parent, key)
        .map(|(value, path)| value_usize(value, &path))
        .unwrap_or(default)
}

fn training_nested_u64(config: &Config, parent: &str, key: &str, default: u64) -> u64 {
    resolved_training_nested_value(config, parent, key)
        .map(|(value, path)| value_u64(value, &path))
        .unwrap_or(default)
}

fn training_nested_bool(config: &Config, parent: &str, key: &str, default: bool) -> bool {
    resolved_training_nested_value(config, parent, key)
        .map(|(value, path)| value_bool(value, &path))
        .unwrap_or(default)
}

fn walk_poison(value: &Value, path: &str, hits: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let child_path = if path.is_empty() {
                    key.to_owned()
                } else {
                    format!("{path}.{key}")
                };
                if FORBIDDEN_KEYS.contains(&key.as_str()) {
                    hits.push(child_path.clone());
                }
                walk_poison(child, &child_path, hits);
            }
        }
        Value::Array(items) => {
            for (idx, child) in items.iter().enumerate() {
                walk_poison(child, &format!("{path}[{idx}]"), hits);
            }
        }
        Value::String(text) => {
            let lower = text.to_ascii_lowercase();
            for token in FORBIDDEN_TEXT {
                if lower.contains(token) {
                    hits.push(path.to_owned());
                }
            }
        }
        _ => {}
    }
}

fn reject_poison(config: &Config) {
    let mut hits = Vec::new();
    walk_poison(&config.arch, "architecture", &mut hits);
    walk_poison(&config.defaults, "defaults", &mut hits);
    walk_poison(&config.build, "build", &mut hits);
    if !hits.is_empty() {
        eprintln!("scratch provenance rejected:");
        for hit in hits {
            eprintln!("  {hit}");
        }
        process::exit(2);
    }
}

fn load_config(args: &[String]) -> (String, Config, bool) {
    let command = args.first().cloned().unwrap_or_else(|| usage());
    let mut build = "build.json".to_owned();
    let mut arch = "architecture.json".to_owned();
    let mut defaults = "defaults.json".to_owned();
    let mut force = false;

    let mut idx = 1;
    while idx < args.len() {
        match args[idx].as_str() {
            "--build" => {
                idx += 1;
                build = args.get(idx).cloned().unwrap_or_else(|| usage());
            }
            "--arch" => {
                idx += 1;
                arch = args.get(idx).cloned().unwrap_or_else(|| usage());
            }
            "--defaults" => {
                idx += 1;
                defaults = args.get(idx).cloned().unwrap_or_else(|| usage());
            }
            "--force" => force = true,
            _ => usage(),
        }
        idx += 1;
    }

    let config = Config {
        arch: load_json(&arch),
        defaults: load_json(&defaults),
        build: load_json(&build),
    };
    reject_poison(&config);
    validate_config_contract(&config);

    if string_at(&config.arch, "lineage") != Some("scratch-native")
        || string_at(&config.build, "lineage") != Some("scratch-native")
    {
        eprintln!("error: architecture.json and build.json lineage must be scratch-native");
        process::exit(2);
    }

    (command, config, force)
}

fn data_config(config: &Config) -> DataConfig {
    let data = object_at(&config.build, "data");
    DataConfig {
        source_binpack: required_string(data, "source_binpack"),
        bullet_output: string_at(data, "bullet_output")
            .map(str::to_owned)
            .unwrap_or_else(|| format!("data/bullet/{}.bullet", run_name(config))),
        offset: u64_at(
            data,
            "offset",
            training_nested_u64(config, "sfbinpack", "offset", 0),
        ),
        limit: u64_at(data, "limit", 0),
        threads: usize_at(data, "threads", training_usize(config, "threads", 4)),
        buffer_mb: training_nested_usize(config, "sfbinpack", "buffer_mb", 1024),
        min_ply: training_nested_usize(config, "sfbinpack", "min_ply", 16) as u16,
        max_abs_cp: training_nested_usize(config, "sfbinpack", "max_abs_cp", 10000) as u32,
        quiet_only: training_nested_bool(config, "sfbinpack", "quiet_only", true),
    }
}

fn run_name(config: &Config) -> String {
    required_string(&config.build, "run")
}

fn out_dir(config: &Config) -> String {
    string_at(&config.build, "out")
        .map(str::to_owned)
        .unwrap_or_else(|| format!("runs/{}/checkpoints", run_name(config)))
}

fn net_path(config: &Config) -> String {
    string_at(&config.build, "net")
        .map(str::to_owned)
        .unwrap_or_else(|| format!("~/assets/nets/{}.nn", run_name(config)))
}

fn net_id(config: &Config) -> String {
    training_string(config, "net_id", "scratch_native")
}

fn continue_from(config: &Config) -> Option<String> {
    string_at(&config.build, "continue_from").map(str::to_owned)
}

fn init_net(config: &Config) -> Option<String> {
    string_at(&config.build, "init_net").map(str::to_owned)
}

fn init_weights_path(config: &Config) -> PathBuf {
    expand_path(&format!(
        "runs/{}/init/optimiser_state/weights.bin",
        run_name(config)
    ))
}

fn command_path(value: &str) -> PathBuf {
    if value.starts_with("~/")
        || value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
    {
        expand_path(value)
    } else {
        PathBuf::from(value)
    }
}

fn python_command() -> PathBuf {
    if let Ok(value) = env::var("PYTHON") {
        let value = value.trim();
        if !value.is_empty() {
            return command_path(value);
        }
    }
    let venv = root().join(".venv/bin/python");
    if venv.exists() {
        venv
    } else {
        PathBuf::from("python3")
    }
}

fn convert_init_net(config: &Config, init_net: &str) -> PathBuf {
    let input = expand_path(init_net);
    if !input.exists() {
        eprintln!("error: missing init_net: {}", input.display());
        process::exit(1);
    }
    let output = init_weights_path(config);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    let status = Command::new(python_command())
        .arg(root().join("tools/bullet/enyo_nn_to_bullet_weights.py"))
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(&output)
        .arg("--eval-scale")
        .arg(f64_at(&config.arch, "eval_scale", 400.0).to_string())
        .arg("--l1-export-scale")
        .arg(f64_at(&config.arch, "l1_export_scale", 1.0).to_string())
        .arg("--input-buckets")
        .arg(usize_at(&config.arch, "input_buckets", 1).to_string())
        .arg("--feature-channels")
        .arg(usize_at(&config.arch, "feature_channels", 12).to_string())
        .arg("--output-buckets")
        .arg(usize_at(&config.arch, "output_buckets", 1).to_string())
        .status()
        .unwrap_or_else(|err| {
            eprintln!("error: cannot run init_net converter: {err}");
            process::exit(1);
        });
    if !status.success() {
        eprintln!("error: init_net conversion failed");
        process::exit(1);
    }
    output
}
fn training_loader(config: &Config) -> String {
    training_string(config, "loader", "direct")
}

fn training_lr(config: &Config) -> f64 {
    training_f64(config, "lr", 1e-3)
}

fn training_final_lr(config: &Config) -> f64 {
    training_f64(config, "final_lr", 1e-4)
}

fn training_wdl(config: &Config) -> f64 {
    training_f64(config, "wdl", 0.3)
}

fn training_superbatches(config: &Config) -> usize {
    training_usize(config, "superbatches", 64)
}

fn training_batch_size(config: &Config) -> usize {
    training_usize(config, "batch_size", 2048)
}

fn training_batches(config: &Config) -> usize {
    training_usize(config, "batches", 64)
}

fn training_threads(config: &Config) -> usize {
    training_usize(config, "threads", 4)
}

fn training_save_rate(config: &Config) -> usize {
    training_usize(config, "save_rate", 64)
}

fn training_trainable(config: &Config) -> String {
    training_string(config, "trainable", "all")
}

fn training_weight_decay(config: &Config) -> f64 {
    training_f64(config, "weight_decay", 0.0)
}

fn validate_layout(config: &Config) {
    let input_buckets = usize_at(&config.arch, "input_buckets", 1);
    let runtime_input_buckets = usize_at(&config.arch, "runtime_input_buckets", input_buckets);
    let feature_channels = usize_at(&config.arch, "feature_channels", 12);
    let output_buckets = usize_at(&config.arch, "output_buckets", 1);
    if !ENYO_SUPPORTED_INPUT_BUCKETS.contains(&input_buckets)
        || !ENYO_SUPPORTED_INPUT_BUCKETS.contains(&runtime_input_buckets)
    {
        eprintln!("error: unsupported Enyo input bucket count");
        process::exit(2);
    }
    if input_buckets > runtime_input_buckets {
        eprintln!("error: runtime_input_buckets cannot be smaller than input_buckets");
        process::exit(2);
    }
    if !ENYO_SUPPORTED_FEATURE_CHANNELS.contains(&feature_channels) {
        eprintln!("error: unsupported Enyo feature_channels={feature_channels}");
        process::exit(2);
    }
    if feature_channels == 11 && (input_buckets != 32 || runtime_input_buckets != 32) {
        eprintln!("error: 11-channel layout requires 32 input/runtime buckets");
        process::exit(2);
    }
    if !ENYO_SUPPORTED_OUTPUT_BUCKETS.contains(&output_buckets) {
        eprintln!("error: unsupported Enyo output_buckets={output_buckets}");
        process::exit(2);
    }
}

fn cmd_plan(config: &Config) {
    let data = data_config(config);
    let init_net = init_net(config);
    println!("run={}", run_name(config));
    println!("lineage={}", required_string(&config.build, "lineage"));
    if let Some(previous_run) = continue_from(config) {
        println!("continue_from={previous_run}");
        if init_net.is_none() {
            println!(
                "init_weights={}",
                latest_weight_checkpoint(config, &previous_run).display()
            );
        }
    }
    if let Some(net) = init_net {
        println!("init_net={net}");
        println!("init_weights={}", init_weights_path(config).display());
    }
    println!("source_binpack={}", data.source_binpack);
    println!("bullet_output={}", data.bullet_output);
    println!("offset={}", data.offset);
    println!("limit={}", data.limit);
    println!("net={}", net_path(config));
    println!();
    println!("commands:");
    println!("  tools/bullet/train plan --build build.json");
    println!("  tools/bullet/train all --build build.json");
    println!("debug:");
    println!("  tools/bullet/train data --build build.json");
    println!("  tools/bullet/train run --build build.json");
    println!("  tools/bullet/train export --build build.json");
    println!();
    println!("resolved:");
    println!(
        "  layout={} buckets, {} channels, hidden={}, output_buckets={}",
        usize_at(&config.arch, "input_buckets", 1),
        usize_at(&config.arch, "feature_channels", 12),
        usize_at(&config.arch, "hidden", 1024),
        usize_at(&config.arch, "output_buckets", 1),
    );
    println!(
        "  loader={}, net_id={}, threads={}, save_rate={}",
        training_loader(config),
        net_id(config),
        training_threads(config),
        training_save_rate(config),
    );
    println!(
        "  dose={} superbatches, batch_size={}, batches={}",
        training_superbatches(config),
        training_batch_size(config),
        training_batches(config),
    );
    println!(
        "  wdl={}, lr={}, final_lr={}, trainable={}, weight_decay={}",
        training_wdl(config),
        training_lr(config),
        training_final_lr(config),
        training_trainable(config),
        training_weight_decay(config),
    );
    println!(
        "  sfbinpack buffer_mb={}, offset={}, min_ply={}, max_abs_cp={}, quiet_only={}",
        training_nested_usize(config, "sfbinpack", "buffer_mb", 1024),
        training_nested_u64(config, "sfbinpack", "offset", 0),
        training_nested_usize(config, "sfbinpack", "min_ply", 16),
        training_nested_usize(config, "sfbinpack", "max_abs_cp", 10000),
        training_nested_bool(config, "sfbinpack", "quiet_only", true),
    );
}

fn write_chunk(writer: &mut BufWriter<File>, chunk: &[ChessBoard]) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            chunk.as_ptr() as *const u8,
            mem::size_of_val(chunk),
        )
    };
    writer.write_all(bytes)
}

fn is_bullet_data_path(path: &Path) -> bool {
    matches!(path.extension().and_then(OsStr::to_str), Some("bullet" | "data"))
}

fn copy_bullet_data(source: &Path, output: &Path, offset: u64, limit: u64) {
    let record_size = mem::size_of::<ChessBoard>() as u64;
    let source_size = source
        .metadata()
        .unwrap_or_else(|err| {
            eprintln!("error: cannot stat {}: {err}", source.display());
            process::exit(1);
        })
        .len();
    if source_size % record_size != 0 {
        eprintln!(
            "error: Bullet data size is not a multiple of record size: {}",
            source.display()
        );
        process::exit(1);
    }
    let total = source_size / record_size;
    if offset >= total {
        eprintln!("error: offset {offset} >= Bullet data records {total}");
        process::exit(1);
    }
    let available = total - offset;
    let records = if limit == 0 { available } else { limit.min(available) };
    if records == 0 {
        eprintln!("error: selected 0 Bullet data records");
        process::exit(1);
    }
    if source == output {
        if offset != 0 || records != total {
            eprintln!("error: cannot slice Bullet data in place: {}", source.display());
            process::exit(1);
        }
        eprintln!("using existing Bullet data: {} records from {}", records, source.display());
        return;
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    let mut input = BufReader::new(File::open(source).unwrap_or_else(|err| {
        eprintln!("error: cannot open {}: {err}", source.display());
        process::exit(1);
    }));
    input.seek(SeekFrom::Start(offset * record_size)).unwrap_or_else(|err| {
        eprintln!("error: cannot seek {}: {err}", source.display());
        process::exit(1);
    });
    let mut output_file = BufWriter::new(File::create(output).unwrap_or_else(|err| {
        eprintln!("error: cannot create {}: {err}", output.display());
        process::exit(1);
    }));
    let bytes = records * record_size;
    let copied = std::io::copy(&mut input.take(bytes), &mut output_file).unwrap_or_else(|err| {
        eprintln!("error: copy failed: {err}");
        process::exit(1);
    });
    if copied != bytes {
        eprintln!("error: copied {copied} bytes, expected {bytes}");
        process::exit(1);
    }
    eprintln!("copied {} Bullet data records to {}", records, output.display());
}

fn cmd_data(config: &Config) {
    let data = data_config(config);
    let source = expand_path(&data.source_binpack);
    if !source.exists() {
        eprintln!("error: missing source binpack: {}", source.display());
        process::exit(1);
    }
    let output = expand_path(&data.bullet_output);
    if is_bullet_data_path(&source) {
        copy_bullet_data(&source, &output, data.offset, data.limit);
        return;
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    let out_file = File::create(&output).unwrap_or_else(|err| {
        eprintln!("error: cannot create {}: {err}", output.display());
        process::exit(1);
    });
    let mut writer = BufWriter::new(out_file);
    let paths = data
        .source_binpack
        .split(';')
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(|path| expand_path(path).display().to_string())
        .collect::<Vec<_>>();
    let refs = paths.iter().map(String::as_str).collect::<Vec<_>>();
    let filter = Filter {
        min_ply: data.min_ply,
        max_abs_cp: data.max_abs_cp,
        quiet_only: data.quiet_only,
    };
    let loader = SfBinpackLoader::new_concat_multiple(
        &refs,
        data.buffer_mb,
        data.threads,
        move |entry| filter.keep(entry),
    );
    let start = Instant::now();
    let mut written = 0_u64;
    let mut skipped = 0_u64;
    loader.map_chunks(0, |chunk: &[ChessBoard]| {
        let mut chunk = chunk;
        if skipped < data.offset {
            let before = skipped;
            let skip = data.offset.saturating_sub(skipped).min(chunk.len() as u64) as usize;
            skipped += skip as u64;
            chunk = &chunk[skip..];
            if skipped % 5_000_000 < skipped - before {
                let secs = start.elapsed().as_secs_f32();
                eprintln!(
                    "skipped {} positions ({:.1}M/s)",
                    skipped,
                    skipped as f32 / secs.max(0.001) / 1e6
                );
            }
            if chunk.is_empty() {
                return false;
            }
        }
        let remaining = if data.limit == 0 {
            chunk.len()
        } else {
            data.limit.saturating_sub(written).min(chunk.len() as u64) as usize
        };
        if remaining == 0 {
            return true;
        }
        let chunk = &chunk[..remaining];
        write_chunk(&mut writer, chunk).unwrap_or_else(|err| {
            eprintln!("error: write failed: {err}");
            process::exit(1);
        });
        written += chunk.len() as u64;
        if written % 5_000_000 < chunk.len() as u64 {
            let secs = start.elapsed().as_secs_f32();
            eprintln!(
                "converted {} positions ({:.1}M/s)",
                written,
                written as f32 / secs.max(0.001) / 1e6
            );
        }
        data.limit != 0 && written >= data.limit
    });
    let secs = start.elapsed().as_secs_f32();
    eprintln!(
        "done: skipped {} positions, converted {} positions to {} in {:.1}s ({:.1}M/s)",
        skipped,
        written,
        output.display(),
        secs,
        written as f32 / secs.max(0.001) / 1e6
    );
}

fn set_env(key: &str, value: impl ToString) {
    unsafe {
        env::set_var(key, value.to_string());
    }
}

fn cmd_run(config: &Config) {
    validate_layout(config);
    let data = data_config(config);
    let bullet_output = expand_path(&data.bullet_output);
    if !bullet_output.exists() {
        eprintln!(
            "error: missing Bullet data: {}; run tools/bullet/train data --build build.json",
            bullet_output.display()
        );
        process::exit(1);
    }
    let output = expand_path(&out_dir(config));
    fs::create_dir_all(&output).unwrap_or_else(|err| {
        eprintln!("error: cannot create {}: {err}", output.display());
        process::exit(1);
    });

    set_env("ENYO_BULLET_DATA", bullet_output.display());
    set_env("ENYO_BULLET_LOADER", training_loader(config));
    set_env("ENYO_BULLET_OUT", output.display());
    set_env("ENYO_BULLET_NET_ID", net_id(config));
    if let Some(net) = init_net(config) {
        let init_weights = convert_init_net(config, &net);
        set_env("ENYO_BULLET_INIT_WEIGHTS", init_weights.display());
    } else if let Some(previous_run) = continue_from(config) {
        set_env("ENYO_BULLET_INIT_WEIGHTS", latest_weight_checkpoint(config, &previous_run).display());
    }
    set_env("ENYO_BULLET_MODE", string_at(&config.arch, "mode").unwrap_or("enyo"));
    set_env("ENYO_BULLET_HIDDEN", usize_at(&config.arch, "hidden", 1024));
    set_env("ENYO_BULLET_L2", usize_at(&config.arch, "l2_size", 16));
    set_env("ENYO_BULLET_BATCH_SIZE", training_batch_size(config));
    set_env("ENYO_BULLET_BATCHES", training_batches(config));
    set_env("ENYO_BULLET_SUPERBATCHES", training_superbatches(config));
    set_env("ENYO_BULLET_THREADS", training_threads(config));
    set_env("ENYO_BULLET_WDL", training_wdl(config));
    set_env("ENYO_BULLET_LR", training_lr(config));
    set_env("ENYO_BULLET_FINAL_LR", training_final_lr(config));
    set_env("ENYO_BULLET_ENYO_L0_STD", f64_at(&config.arch, "l0_std", 8.0));
    set_env("ENYO_BULLET_ENYO_L1_STD", f64_at(&config.arch, "l1_std", 1.0));
    set_env(
        "ENYO_BULLET_ENYO_L1_EXPORT_SCALE",
        f64_at(&config.arch, "l1_export_scale", 1.0),
    );
    set_env(
        "ENYO_BULLET_ENYO_INPUT_FACTORISER",
        usize::from(bool_at(&config.arch, "input_factoriser", false)),
    );
    set_env("ENYO_BULLET_ENYO_INPUT_BUCKETS", usize_at(&config.arch, "input_buckets", 1));
    set_env(
        "ENYO_BULLET_ENYO_RUNTIME_INPUT_BUCKETS",
        usize_at(&config.arch, "runtime_input_buckets", usize_at(&config.arch, "input_buckets", 1)),
    );
    set_env("ENYO_BULLET_ENYO_FEATURE_CHANNELS", usize_at(&config.arch, "feature_channels", 12));
    set_env("ENYO_BULLET_ENYO_OUTPUT_BUCKETS", usize_at(&config.arch, "output_buckets", 1));
    set_env("ENYO_BULLET_EVAL_SCALE", f64_at(&config.arch, "eval_scale", 400.0));
    set_env("ENYO_BULLET_SAVE_RATE", training_save_rate(config));
    set_env("ENYO_BULLET_EXPORT_INIT_ONLY", 0);
    set_env("ENYO_BULLET_TRAINABLE", training_trainable(config));
    set_env("ENYO_BULLET_WEIGHT_DECAY", training_weight_decay(config));
    set_env("ENYO_BULLET_SFBINPACK_BUFFER_MB", data.buffer_mb);
    set_env("ENYO_BULLET_SFBINPACK_MIN_PLY", data.min_ply);
    set_env("ENYO_BULLET_SFBINPACK_MAX_ABS_CP", data.max_abs_cp);
    set_env("ENYO_BULLET_SFBINPACK_QUIET_ONLY", usize::from(data.quiet_only));

    trainer_main::run_from_env();
    write_model(config);
}

fn enyo_network_size(
    input_buckets: usize,
    feature_channels: usize,
    output_buckets: usize,
    hidden: usize,
    l2: usize,
) -> usize {
    let features = input_buckets * feature_channels * 64;
    let l1 = 2 * hidden;
    let l3 = 32;
    features * hidden * 2
        + hidden * 2
        + l1 * l2
        + l2 * 4
        + l2 * l3 * 4
        + l3 * 4
        + output_buckets * l3 * 4
        + output_buckets * 4
}

fn trim_checkpoint(
    raw: &[u8],
    input_buckets: usize,
    feature_channels: usize,
    output_buckets: usize,
    hidden: usize,
    l2: usize,
) -> Vec<u8> {
    let expected = enyo_network_size(input_buckets, feature_channels, output_buckets, hidden, l2);
    if raw.len() < expected {
        eprintln!("error: checkpoint is {} bytes, expected at least {expected}", raw.len());
        process::exit(1);
    }
    if raw.len() == expected {
        return raw.to_vec();
    }
    let trailer = &raw[expected..];
    for (idx, byte) in trailer.iter().enumerate() {
        if *byte != b"bullet"[idx % 6] {
            eprintln!("error: checkpoint has unexpected {} byte trailer", trailer.len());
            process::exit(1);
        }
    }
    raw[..expected].to_vec()
}

fn parent_bucket(target_bucket: usize, input_buckets: usize, runtime_input_buckets: usize) -> usize {
    if input_buckets == runtime_input_buckets {
        return target_bucket;
    }
    if runtime_input_buckets == 32 {
        return ENYO_LEGACY_BUCKET_FOR_32[target_bucket] * input_buckets / 16;
    }
    target_bucket * input_buckets / runtime_input_buckets
}

fn expand_input_buckets(
    raw: &[u8],
    input_buckets: usize,
    feature_channels: usize,
    runtime_input_buckets: usize,
    output_buckets: usize,
    hidden: usize,
    l2: usize,
) -> Vec<u8> {
    let raw = trim_checkpoint(raw, input_buckets, feature_channels, output_buckets, hidden, l2);
    if input_buckets == runtime_input_buckets {
        return raw;
    }

    let feature_stride = feature_channels * 64;
    let source_features = input_buckets * feature_stride;
    let target_features = runtime_input_buckets * feature_stride;
    let feature_bytes = hidden * 2;
    let source_l0_bytes = source_features * feature_bytes;
    let target_l0_bytes = target_features * feature_bytes;
    let rest = &raw[source_l0_bytes..];
    let mut expanded = vec![0_u8; target_l0_bytes + rest.len()];

    for target_bucket in 0..runtime_input_buckets {
        let source_bucket = parent_bucket(target_bucket, input_buckets, runtime_input_buckets);
        for offset in 0..feature_stride {
            let source_feature = source_bucket * feature_stride + offset;
            let target_feature = target_bucket * feature_stride + offset;
            let source_start = source_feature * feature_bytes;
            let target_start = target_feature * feature_bytes;
            expanded[target_start..target_start + feature_bytes]
                .copy_from_slice(&raw[source_start..source_start + feature_bytes]);
        }
    }
    expanded[target_l0_bytes..].copy_from_slice(rest);
    expanded
}

fn latest_checkpoint(config: &Config) -> PathBuf {
    let output = expand_path(&out_dir(config));
    let prefix = format!("{}-", net_id(config));
    let mut checkpoints = fs::read_dir(&output)
        .unwrap_or_else(|err| {
            eprintln!("error: cannot read {}: {err}", output.display());
            process::exit(1);
        })
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(OsStr::to_str)
                .is_some_and(|name| name.starts_with(&prefix))
        })
        .map(|path| path.join("quantised.bin"))
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    checkpoints.sort_by_key(|path| {
        path.metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    checkpoints.into_iter().last().unwrap_or_else(|| {
        eprintln!("error: no quantised.bin checkpoints found under {}", output.display());
        process::exit(1);
    })
}

fn latest_weight_checkpoint(config: &Config, run: &str) -> PathBuf {
    if run.contains('/') || run.contains('\\') || run == "." || run == ".." {
        eprintln!("error: continue_from must be a previous run name, not a path");
        process::exit(2);
    }
    let output = expand_path(&format!("runs/{run}/checkpoints"));
    let prefix = format!("{}-", net_id(config));
    let entries = match fs::read_dir(&output) {
        Ok(entries) => Some(entries),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            eprintln!("error: cannot read {}: {err}", output.display());
            process::exit(1);
        }
    };
    let mut checkpoints = entries
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .map(|entry| entry.path())
        .filter_map(|path| {
            let name = path.file_name().and_then(OsStr::to_str)?;
            let superbatch = name.strip_prefix(&prefix)?.parse::<usize>().ok()?;
            let weights = path.join("optimiser_state/weights.bin");
            weights.exists().then_some((superbatch, weights))
        })
        .collect::<Vec<_>>();
    checkpoints.sort_by_key(|(superbatch, _)| *superbatch);
    if let Some((_, weights)) = checkpoints.into_iter().last() {
        return weights;
    }

    for net in [
        format!("runs/{run}/model.nn"),
        format!("~/assets/nets/{run}.nn"),
    ] {
        let net = expand_path(&net);
        if net.exists() {
            let net = net.to_string_lossy().into_owned();
            return convert_init_net(config, &net);
        }
    }

    eprintln!(
        "error: no optimiser_state/weights.bin checkpoints or exported .nn found for continue_from={run}"
    );
    eprintln!("  checked {}", output.display());
    eprintln!("  checked runs/{run}/model.nn");
    eprintln!("  checked ~/assets/nets/{run}.nn");
    process::exit(1);
}

fn write_model(config: &Config) {
    let checkpoint = latest_checkpoint(config);
    let raw = fs::read(&checkpoint).unwrap_or_else(|err| {
        eprintln!("error: cannot read {}: {err}", checkpoint.display());
        process::exit(1);
    });
    let input_buckets = usize_at(&config.arch, "input_buckets", 1);
    let runtime_input_buckets = usize_at(&config.arch, "runtime_input_buckets", input_buckets);
    let model = expand_input_buckets(
        &raw,
        input_buckets,
        usize_at(&config.arch, "feature_channels", 12),
        runtime_input_buckets,
        usize_at(&config.arch, "output_buckets", 1),
        usize_at(&config.arch, "hidden", 1024),
        usize_at(&config.arch, "l2_size", 16),
    );
    let model_path = expand_path(&format!("runs/{}/model.nn", run_name(config)));
    if let Some(parent) = model_path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    fs::write(&model_path, model).unwrap_or_else(|err| {
        eprintln!("error: cannot write {}: {err}", model_path.display());
        process::exit(1);
    });
    println!("wrote {}", model_path.display());
}

fn cmd_export(config: &Config, force: bool) {
    let model_path = expand_path(&format!("runs/{}/model.nn", run_name(config)));
    if !model_path.exists() {
        write_model(config);
    }
    let net = expand_path(&net_path(config));
    if let Some(parent) = net.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    if net.exists() && !force {
        eprintln!(
            "error: refusing to overwrite existing net: {}\npass --force or use a unique run name",
            net.display()
        );
        process::exit(1);
    }
    fs::copy(&model_path, &net).unwrap_or_else(|err| {
        eprintln!(
            "error: cannot copy {} -> {}: {err}",
            model_path.display(),
            net.display()
        );
        process::exit(1);
    });
    let size = net.metadata().map(|m| m.len()).unwrap_or(0);
    println!("wrote {} ({} bytes)", net.display(), size);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn defaults() -> Value {
        json!({
            "loader": "direct",
            "net_id": "scratch_native",
            "batches": 64,
            "batch_size": 2048,
            "superbatches": 7600,
            "threads": 16,
            "wdl": 0.3,
            "lr": 0.001,
            "final_lr": 0.000005,
            "save_rate": 7600,
            "trainable": "all",
            "weight_decay": 0.0,
            "sfbinpack": {
                "buffer_mb": 1024,
                "offset": 0,
                "min_ply": 16,
                "max_abs_cp": 10000,
                "quiet_only": true
            }
        })
    }

    fn config(build: Value) -> Config {
        Config {
            arch: json!({"lineage": "scratch-native"}),
            defaults: defaults(),
            build,
        }
    }

    #[test]
    fn build_overrides_top_level_training_defaults() {
        let config = config(json!({
            "run": "candidate",
            "lineage": "scratch-native",
            "loader": "sfbinpack",
            "net_id": "override_id",
            "batches": 32,
            "batch_size": 4096,
            "superbatches": 3800,
            "threads": 8,
            "wdl": 0.25,
            "lr": 0.0005,
            "final_lr": 0.000001,
            "save_rate": 1900,
            "trainable": "output",
            "weight_decay": 0.00001,
            "data": {"source_binpack": "data.binpack", "limit": 100, "offset": 20}
        }));

        assert!(config_contract_errors(&config).is_empty());
        assert_eq!(training_loader(&config), "sfbinpack");
        assert_eq!(net_id(&config), "override_id");
        assert_eq!(training_batches(&config), 32);
        assert_eq!(training_batch_size(&config), 4096);
        assert_eq!(training_superbatches(&config), 3800);
        assert_eq!(training_threads(&config), 8);
        assert_eq!(training_wdl(&config), 0.25);
        assert_eq!(training_lr(&config), 0.0005);
        assert_eq!(training_final_lr(&config), 0.000001);
        assert_eq!(training_save_rate(&config), 1900);
        assert_eq!(training_trainable(&config), "output");
        assert_eq!(training_weight_decay(&config), 0.00001);
    }

    #[test]
    fn build_overrides_nested_sfbinpack_defaults() {
        let config = config(json!({
            "run": "candidate",
            "lineage": "scratch-native",
            "sfbinpack": {
                "buffer_mb": 2048,
                "offset": 500,
                "min_ply": 20,
                "max_abs_cp": 2000,
                "quiet_only": false
            },
            "data": {"source_binpack": "data.binpack", "limit": 100}
        }));

        assert!(config_contract_errors(&config).is_empty());
        assert_eq!(
            training_nested_usize(&config, "sfbinpack", "buffer_mb", 0),
            2048
        );
        assert_eq!(training_nested_u64(&config, "sfbinpack", "offset", 0), 500);
        assert_eq!(
            training_nested_usize(&config, "sfbinpack", "min_ply", 0),
            20
        );
        assert_eq!(
            training_nested_usize(&config, "sfbinpack", "max_abs_cp", 0),
            2000
        );
        assert!(!training_nested_bool(
            &config,
            "sfbinpack",
            "quiet_only",
            true
        ));
    }

    #[test]
    fn data_offset_overrides_sfbinpack_offset_for_data_selection() {
        let config = config(json!({
            "run": "candidate",
            "lineage": "scratch-native",
            "sfbinpack": {"offset": 500},
            "data": {"source_binpack": "data.binpack", "limit": 100, "offset": 900, "threads": 3}
        }));

        assert!(config_contract_errors(&config).is_empty());
        let data = data_config(&config);
        assert_eq!(data.offset, 900);
        assert_eq!(data.threads, 3);
    }

    #[test]
    fn unknown_build_training_key_is_rejected() {
        let config = config(json!({
            "run": "candidate",
            "lineage": "scratch-native",
            "new_training_knob": 1,
            "data": {"source_binpack": "data.binpack"}
        }));

        let errors = config_contract_errors(&config);
        assert!(
            errors
                .iter()
                .any(|err| err.contains("build.new_training_knob"))
        );
    }

    #[test]
    fn init_net_and_continue_from_are_accepted_together() {
        let config = config(json!({
            "run": "uho-native-1.1.0",
            "lineage": "scratch-native",
            "continue_from": "uho-native-1.0.42",
            "init_net": "~/assets/nets/uho-native-1.0.42.nn",
            "reference": "uho-native-1.0.42",
            "data": {"source_binpack": "data.binpack", "limit": 100}
        }));

        assert!(config_contract_errors(&config).is_empty());
        assert_eq!(continue_from(&config).as_deref(), Some("uho-native-1.0.42"));
        assert_eq!(init_net(&config).as_deref(), Some("~/assets/nets/uho-native-1.0.42.nn"));
    }

    #[test]
    fn data_training_knobs_are_rejected() {
        let config = config(json!({
            "run": "candidate",
            "lineage": "scratch-native",
            "data": {"source_binpack": "data.binpack", "buffer_mb": 2048}
        }));

        let errors = config_contract_errors(&config);
        assert!(
            errors
                .iter()
                .any(|err| err.contains("build.data.buffer_mb"))
        );
    }

    #[test]
    fn missing_default_training_parameter_is_rejected() {
        let mut config = config(json!({
            "run": "candidate",
            "lineage": "scratch-native",
            "data": {"source_binpack": "data.binpack"}
        }));
        config.defaults.as_object_mut().unwrap().remove("wdl");

        let errors = config_contract_errors(&config);
        assert!(errors.iter().any(|err| err.contains("defaults.wdl")));
    }
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let (command, config, force) = load_config(&args);
    match command.as_str() {
        "plan" => cmd_plan(&config),
        "data" => cmd_data(&config),
        "run" => cmd_run(&config),
        "export" => cmd_export(&config, force),
        "all" => {
            cmd_data(&config);
            cmd_run(&config);
            cmd_export(&config, force);
        }
        _ => usage(),
    }
}
