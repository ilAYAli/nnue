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
        sfbinpack::{MoveType, PieceType, TrainingDataEntry},
        DataLoader, SfBinpackLoader,
    },
};
use serde_json::Value;
use sha2::{Digest, Sha256};

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

const ENYO_SUPPORTED_INPUT_BUCKETS: &[usize] = &[1, 2, 4, 8, 10, 16, 32];
const ENYO_SUPPORTED_HIDDEN: &[usize] = &[512, 768, 1024];
const ENYO_SUPPORTED_FEATURE_CHANNELS: &[usize] = &[11, 12];
const ENYO_SUPPORTED_OUTPUT_BUCKETS: &[usize] = &[1, 2, 4, 8];
const ENYO_RUNTIME_HIDDEN: usize = 1024;
const ENYO_V2_HEADER_MAGIC: &[u8; 8] = b"ENYONN2\0";
const ENYO_V3_HEADER_MAGIC: &[u8; 8] = b"ENYONN3\0";
const ENYO_V4_HEADER_MAGIC: &[u8; 8] = b"ENYONN4\0";
const ENYO_V5_HEADER_MAGIC: &[u8; 8] = b"ENYONN5\0";
const ENYO_V6_HEADER_MAGIC: &[u8; 8] = b"ENYONN6\0";
const ENYO_V7_HEADER_MAGIC: &[u8; 8] = b"ENYONN7\0";
const ENYO_V2_FORMAT_VERSION: u32 = 2;
const ENYO_V3_FORMAT_VERSION: u32 = 3;
const ENYO_V4_FORMAT_VERSION: u32 = 4;
const ENYO_V5_FORMAT_VERSION: u32 = 5;
const ENYO_V6_FORMAT_VERSION: u32 = 6;
const ENYO_V7_FORMAT_VERSION: u32 = 7;
const ENYO_NETWORK_HEADER_SIZE: usize = 64;
const ENYO_NETWORK_FLAG_FULL_THREATS: u32 = 1;
const ENYO_NETWORK_FLAG_FULL_HEADS: u32 = 2;
const ENYO_NETWORK_FLAG_MIXED_ACTIVATION: u32 = 4;
const ENYO_NETWORK_FLAG_PSQT_RESIDUAL: u32 = 8;
const ENYO_NETWORK_FLAG_PAIRWISE: u32 = 16;
const ENYO_NETWORK_FLAG_SLIDER_XRAY_THREATS: u32 = 16;
const ENYO_NETWORK_FLAG_RECKLESS_THREATS: u32 = 32;
const RECKLESS_THREAT_DIMENSIONS: usize = 66_864;
const ENYO_FULL_THREATS_DIMENSIONS: usize = 60_720;
const ENYO_LEGACY_BUCKET_FOR_32: [usize; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11, 12, 12, 13, 13, 12, 12, 13, 13, 14, 14, 15,
    15, 14, 14, 15, 15,
];
const TRAIN_PROVENANCE_SCHEMA: u64 = 1;

const TRAINING_DEFAULT_KEYS: &[&str] = &[
    "loader",
    "net_id",
    "batches",
    "batch_size",
    "superbatches",
    "lr_superbatches",
    "threads",
    "init_seed",
    "wdl",
    "lr",
    "final_lr",
    "save_rate",
    "trainable",
    "weight_decay",
    "activation_l1",
    "output_bucket_weights",
    "sfbinpack",
];

const SFBINPACK_DEFAULT_KEYS: &[&str] =
    &["buffer_mb", "offset", "min_ply", "max_abs_cp", "quiet_only"];

const BUILD_METADATA_KEYS: &[&str] = &[
    "run",
    "continue_from",
    "initialize_from",
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
    output_bucket_weights: Vec<f32>,
    eval_bucket_weights: Vec<f32>,
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
    if let Some(output) = data.get("bullet_output").and_then(Value::as_str) {
        if !path_string_has_extension(output, "bullet") {
            errors.push(format!(
                "build.data.bullet_output: must end in .bullet, got {output}"
            ));
        }
    }
    if let Some(source) = data.get("source_binpack").and_then(Value::as_str) {
        if source
            .split(';')
            .any(|path| path_string_has_extension(path.trim(), "data"))
        {
            errors.push(
                "build.data.source_binpack: .data BulletFormat inputs are deprecated; rename them to .bullet"
                    .to_string(),
            );
        }
    }
}

fn path_string_has_extension(path: &str, extension: &str) -> bool {
    Path::new(path).extension().and_then(OsStr::to_str) == Some(extension)
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

fn training_u64(config: &Config, key: &str, default: u64) -> u64 {
    resolved_training_value(config, key)
        .map(|(value, path)| value_u64(value, &path))
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

    if !matches!(string_at(&config.arch, "lineage"), Some("native" | "recklessft")) {
        eprintln!("error: unsupported architecture.json lineage");
        process::exit(2);
    }

    (command, config, force)
}

fn data_config(config: &Config) -> DataConfig {
    let data = object_at(&config.build, "data");
    let source_binpack = required_string(data, "source_binpack");
    let offset = u64_at(
        data,
        "offset",
        training_nested_u64(config, "sfbinpack", "offset", 0),
    );
    let limit = u64_at(data, "limit", 0);
    let bullet_output = string_at(data, "bullet_output")
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if source_binpack.ends_with(".bullet") && offset == 0 && limit == 0 {
                source_binpack.clone()
            } else {
                format!("data/bullet/{}.bullet", run_name(config))
            }
        });
    DataConfig {
        source_binpack,
        bullet_output,
        offset,
        limit,
        threads: usize_at(data, "threads", training_usize(config, "threads", 4)),
        buffer_mb: training_nested_usize(config, "sfbinpack", "buffer_mb", 1024),
        min_ply: training_nested_usize(config, "sfbinpack", "min_ply", 16) as u16,
        max_abs_cp: training_nested_usize(config, "sfbinpack", "max_abs_cp", 10000) as u32,
        quiet_only: training_nested_bool(config, "sfbinpack", "quiet_only", true),
        output_bucket_weights: parse_output_bucket_weights(
            &training_output_bucket_weights(config),
            usize_at(&config.arch, "output_buckets", 8),
        ),
        eval_bucket_weights: parse_eval_bucket_weights(&training_eval_bucket_weights(config)),
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
    training_string(config, "net_id", "native")
}

fn continue_from(config: &Config) -> Option<String> {
    string_at(&config.build, "continue_from").map(str::to_owned)
}

fn initialize_from(config: &Config) -> Option<String> {
    string_at(&config.build, "initialize_from").map(str::to_owned)
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

fn initialize_from_path(value: &str) -> PathBuf {
    if value.starts_with("~/")
        || value.starts_with('/')
        || value.starts_with("./")
        || value.starts_with("../")
        || value.ends_with(".nn")
        || value.ends_with(".net")
    {
        return expand_path(value);
    }

    for candidate in [
        format!("~/assets/nets/{value}.nn"),
        format!("runs/{value}/model.nn"),
    ] {
        let path = expand_path(&candidate);
        if path.exists() {
            return path;
        }
    }

    expand_path(value)
}

fn convert_initialize_from(config: &Config, initialize_from: &str) -> PathBuf {
    let input = initialize_from_path(initialize_from);
    if !input.exists() {
        eprintln!("error: missing initialize_from: {}", input.display());
        process::exit(1);
    }
    let output = init_weights_path(config);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    let mut command = Command::new(python_command());
    command
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
        .arg("--hidden")
        .arg(usize_at(&config.arch, "hidden", ENYO_RUNTIME_HIDDEN).to_string());
    if arch_full_threats(config) {
        command.arg("--full-threats");
    }
    if arch_slider_xray_threats(config) {
        command.arg("--slider-xray-threats");
    }
    if arch_full_heads(config) {
        command.arg("--full-heads");
    }
    if arch_mixed_activation(config) {
        command.arg("--mixed-activation");
    }
    if arch_psqt_residual(config) {
        command.arg("--psqt-residual");
    }
    let status = command.status().unwrap_or_else(|err| {
        eprintln!("error: cannot run initialize_from converter: {err}");
        process::exit(1);
    });
    if !status.success() {
        eprintln!("error: initialize_from conversion failed");
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

fn training_lr_superbatches(config: &Config) -> usize {
    match training_usize(config, "lr_superbatches", 0) {
        0 => training_superbatches(config),
        configured => configured,
    }
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

fn supported_trainable(value: &str) -> bool {
    matches!(
        value,
        "all"
            | "input"
            | "xray-only"
            | "dense-head"
            | "float-head"
            | "output"
            | "frozen-output"
            | "squared-branch"
            | "psqt"
    )
}

fn training_weight_decay(config: &Config) -> f64 {
    training_f64(config, "weight_decay", 0.0)
}

fn training_activation_l1(config: &Config) -> f64 {
    training_f64(config, "activation_l1", 0.0)
}

fn training_output_bucket_weights(config: &Config) -> String {
    training_string(config, "output_bucket_weights", "auto")
}

fn training_eval_bucket_weights(config: &Config) -> String {
    training_string(config, "eval_bucket_weights", "auto")
}

fn parse_eval_bucket_weights(raw: &str) -> Vec<f32> {
    let values = if raw == "auto" {
        vec![1.0; 5]
    } else {
        raw.split(',')
            .map(str::trim)
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|_| panic!("invalid eval_bucket_weights={raw}"))
    };
    if values.len() != 5 || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        panic!("eval_bucket_weights requires 5 finite positive values");
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.into_iter().map(|v| v / mean).collect()
}

fn parse_output_bucket_weights(raw: &str, buckets: usize) -> Vec<f32> {
    let values = if raw == "auto" {
        vec![1.0; buckets]
    } else {
        raw.split(',')
            .map(str::trim)
            .map(|value| value.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|_| panic!("invalid output_bucket_weights={raw}"))
    };
    if values.len() != buckets || values.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        panic!("output_bucket_weights requires {buckets} finite positive values");
    }
    let mean = values.iter().sum::<f32>() / buckets as f32;
    values.into_iter().map(|v| v / mean).collect()
}

fn arch_full_threats(config: &Config) -> bool {
    bool_at(&config.arch, "full_threats", false)
}

fn arch_slider_xray_threats(config: &Config) -> bool {
    bool_at(&config.arch, "slider_xray_threats", false)
}

fn arch_full_heads(config: &Config) -> bool {
    match string_at(&config.arch, "output_bucket_scope").unwrap_or("final") {
        "final" => false,
        "full-head" => true,
        value => {
            eprintln!("error: unsupported output_bucket_scope={value}");
            process::exit(2);
        }
    }
}

fn arch_mixed_activation(config: &Config) -> bool {
    match string_at(&config.arch, "dense_activation").unwrap_or("relu") {
        "relu" => false,
        "relu-screlu-residual" => true,
        value => {
            eprintln!("error: unsupported dense_activation={value}");
            process::exit(2);
        }
    }
}

fn arch_psqt_residual(config: &Config) -> bool {
    bool_at(&config.arch, "psqt_residual", false)
}

fn validate_layout(config: &Config) {
    let input_buckets = usize_at(&config.arch, "input_buckets", 1);
    let runtime_input_buckets = usize_at(&config.arch, "runtime_input_buckets", input_buckets);
    let feature_channels = usize_at(&config.arch, "feature_channels", 12);
    let output_buckets = usize_at(&config.arch, "output_buckets", 1);
    let hidden = usize_at(&config.arch, "hidden", ENYO_RUNTIME_HIDDEN);
    let l2 = usize_at(&config.arch, "l2_size", 16);
    let export_format = string_at(&config.arch, "export_format").unwrap_or("enyo-native-v1");
    let full_threats = arch_full_threats(config);
    let slider_xray_threats = arch_slider_xray_threats(config);
    let threat_features = full_threats || slider_xray_threats;
    let full_heads = arch_full_heads(config);
    let mixed_activation = arch_mixed_activation(config);
    let psqt_residual = arch_psqt_residual(config);
    let mode = string_at(&config.arch, "mode").unwrap_or("enyo");
    if mode == "reckless" {
        if input_buckets != 10
            || runtime_input_buckets != 10
            || feature_channels != 12
            || output_buckets != 8
            || hidden != 768
            || l2 != 16
            || !bool_at(&config.arch, "input_factoriser", false)
            || !full_heads
            || threat_features
            || mixed_activation
            || psqt_residual
            || export_format != "enyo-native-v7-reckless-threats"
        {
            eprintln!("error: current native Reckless requires 10x12-768-o8, factorised input, full heads, and enyo-native-v7-reckless-threats");
            process::exit(2);
        }
        if training_lr_superbatches(config) < training_superbatches(config) {
            eprintln!("error: lr_superbatches cannot be smaller than superbatches");
            process::exit(2);
        }
        return;
    }
    if mode != "enyo" {
        eprintln!("error: unsupported architecture mode={mode}");
        process::exit(2);
    }
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
    if feature_channels == 11
        && (!matches!(input_buckets, 10 | 16 | 32)
            || !matches!(runtime_input_buckets, 10 | 16 | 32))
    {
        eprintln!("error: 11-channel layout requires 10, 16, or 32 input/runtime buckets");
        process::exit(2);
    }
    if !ENYO_SUPPORTED_OUTPUT_BUCKETS.contains(&output_buckets) {
        eprintln!("error: unsupported Enyo output_buckets={output_buckets}");
        process::exit(2);
    }
    if !ENYO_SUPPORTED_HIDDEN.contains(&hidden) || l2 != 16 {
        eprintln!("error: Enyo layout requires hidden=512, 768, or 1024 and l2_size=16");
        process::exit(2);
    }
    if !matches!(
        export_format,
        "enyo-native-v1" | "enyo-native-v2" | "enyo-native-v3" | "enyo-native-v4" | "enyo-native-v5"
            | "enyo-native-v6"
    ) {
        eprintln!("error: unsupported export_format={export_format}");
        process::exit(2);
    }
    if export_format == "enyo-native-v1" && hidden != ENYO_RUNTIME_HIDDEN {
        eprintln!("error: non-1024 hidden widths require export_format=enyo-native-v2");
        process::exit(2);
    }
    if threat_features && export_format != "enyo-native-v2" && export_format != "enyo-native-v6" {
        eprintln!("error: threat features require export_format=enyo-native-v2 (alone) or enyo-native-v6 (with full-head)");
        process::exit(2);
    }
    if full_threats && slider_xray_threats {
        eprintln!("error: full_threats and slider_xray_threats are mutually exclusive");
        process::exit(2);
    }
    if full_heads && export_format != "enyo-native-v3" && export_format != "enyo-native-v6" {
        eprintln!("error: output_bucket_scope=full-head requires export_format=enyo-native-v3 (alone) or enyo-native-v6 (with threat features)");
        process::exit(2);
    }
    if export_format == "enyo-native-v3" && !full_heads {
        eprintln!("error: enyo-native-v3 requires output_bucket_scope=full-head");
        process::exit(2);
    }
    if export_format == "enyo-native-v6" && !(full_heads && threat_features) {
        eprintln!("error: enyo-native-v6 requires both output_bucket_scope=full-head and a threat-feature mode");
        process::exit(2);
    }
    if mixed_activation && export_format != "enyo-native-v4" {
        eprintln!("error: relu-screlu-residual requires export_format=enyo-native-v4");
        process::exit(2);
    }
    if export_format == "enyo-native-v4" && !mixed_activation {
        eprintln!("error: enyo-native-v4 requires relu-screlu-residual");
        process::exit(2);
    }
    if psqt_residual && export_format != "enyo-native-v5" {
        eprintln!("error: PSQT residual requires export_format=enyo-native-v5");
        process::exit(2);
    }
    if export_format == "enyo-native-v5" && !psqt_residual {
        eprintln!("error: enyo-native-v5 requires psqt_residual=true");
        process::exit(2);
    }
    if psqt_residual && (mixed_activation || full_heads || threat_features || output_buckets != 8) {
        eprintln!("error: PSQT residual requires the shared-head 8-bucket base architecture");
        process::exit(2);
    }
    if mixed_activation && (full_heads || threat_features || output_buckets != 8) {
        eprintln!("error: mixed activation requires the shared-head 8-bucket base architecture");
        process::exit(2);
    }
    if training_trainable(config) == "squared-branch" && !mixed_activation {
        eprintln!("error: squared-branch requires mixed activation");
        process::exit(2);
    }
    if training_trainable(config) == "psqt" && !psqt_residual {
        eprintln!("error: psqt trainable mode requires psqt_residual=true");
        process::exit(2);
    }
    if full_heads && output_buckets <= 1 {
        eprintln!("error: full-head output bucketing requires at least 2 output buckets");
        process::exit(2);
    }
    if threat_features && bool_at(&config.arch, "input_factoriser", false) {
        eprintln!("error: threat features do not support input_factoriser yet");
        process::exit(2);
    }
    if threat_features && input_buckets != runtime_input_buckets {
        eprintln!("error: threat features do not support runtime_input_buckets expansion yet");
        process::exit(2);
    }
    if training_lr_superbatches(config) < training_superbatches(config) {
        eprintln!("error: lr_superbatches cannot be smaller than superbatches");
        process::exit(2);
    }
    if !supported_trainable(&training_trainable(config)) {
        eprintln!(
            "error: unsupported trainable mode: {}",
            training_trainable(config)
        );
        process::exit(2);
    }
}

fn cmd_plan(config: &Config) {
    let data = data_config(config);
    let initialize_from = initialize_from(config);
    println!("run={}", run_name(config));
    println!("lineage={}", required_string(&config.arch, "lineage"));
    if let Some(previous_run) = continue_from(config) {
        println!("continue_from={previous_run}");
        if initialize_from.is_none() {
            println!(
                "init_weights={}",
                latest_weight_checkpoint(config, &previous_run).display()
            );
        }
    }
    if let Some(net) = initialize_from {
        println!("initialize_from={net}");
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
        "  layout={} buckets, {} channels, hidden={}, output_buckets={}, full_threats={}, slider_xray_threats={}, full_heads={}",
        usize_at(&config.arch, "input_buckets", 1),
        usize_at(&config.arch, "feature_channels", 12),
        usize_at(&config.arch, "hidden", 1024),
        usize_at(&config.arch, "output_buckets", 1),
        arch_full_threats(config),
        arch_slider_xray_threats(config),
        arch_full_heads(config),
    );
    println!(
        "  loader={}, net_id={}, threads={}, save_rate={}",
        training_loader(config),
        net_id(config),
        training_threads(config),
        training_save_rate(config),
    );
    println!(
        "  dose={} superbatches, lr_schedule={} superbatches, batch_size={}, batches={}",
        training_superbatches(config),
        training_lr_superbatches(config),
        training_batch_size(config),
        training_batches(config),
    );
    println!(
        "  wdl={}, lr={}, final_lr={}, trainable={}, weight_decay={}, activation_l1={}",
        training_wdl(config),
        training_lr(config),
        training_final_lr(config),
        training_trainable(config),
        training_weight_decay(config),
        training_activation_l1(config),
    );
    println!("output_bucket_weights={}", training_output_bucket_weights(config));
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
    let bytes =
        unsafe { std::slice::from_raw_parts(chunk.as_ptr() as *const u8, mem::size_of_val(chunk)) };
    writer.write_all(bytes)
}

fn bullet_output_tmp_path(output: &Path) -> PathBuf {
    let file_name = output
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("bullet-output");
    output.with_file_name(format!("{file_name}.tmp.{}", process::id()))
}

fn create_bullet_output(output: &Path) -> (PathBuf, BufWriter<File>) {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            eprintln!("error: cannot create {}: {err}", parent.display());
            process::exit(1);
        });
    }
    let tmp = bullet_output_tmp_path(output);
    let _ = fs::remove_file(&tmp);
    let out_file = File::create(&tmp).unwrap_or_else(|err| {
        eprintln!("error: cannot create {}: {err}", tmp.display());
        process::exit(1);
    });
    (tmp, BufWriter::new(out_file))
}

fn publish_bullet_output(tmp: PathBuf, mut writer: BufWriter<File>, output: &Path, records: u64) {
    writer.flush().unwrap_or_else(|err| {
        let _ = fs::remove_file(&tmp);
        eprintln!("error: cannot flush {}: {err}", tmp.display());
        process::exit(1);
    });
    writer.get_ref().sync_all().unwrap_or_else(|err| {
        let _ = fs::remove_file(&tmp);
        eprintln!("error: cannot sync {}: {err}", tmp.display());
        process::exit(1);
    });
    drop(writer);

    let expected = records * mem::size_of::<ChessBoard>() as u64;
    let actual = tmp
        .metadata()
        .unwrap_or_else(|err| {
            let _ = fs::remove_file(&tmp);
            eprintln!("error: cannot stat {}: {err}", tmp.display());
            process::exit(1);
        })
        .len();
    if actual != expected {
        let _ = fs::remove_file(&tmp);
        eprintln!(
            "error: wrote {actual} bytes to {}, expected {expected}",
            tmp.display()
        );
        process::exit(1);
    }

    fs::rename(&tmp, output).unwrap_or_else(|err| {
        let _ = fs::remove_file(&tmp);
        eprintln!(
            "error: cannot publish {} to {}: {err}",
            tmp.display(),
            output.display()
        );
        process::exit(1);
    });
}

fn is_bullet_data_path(path: &Path) -> bool {
    matches!(path.extension().and_then(OsStr::to_str), Some("bullet"))
}

fn is_deprecated_bullet_data_path(path: &Path) -> bool {
    matches!(path.extension().and_then(OsStr::to_str), Some("data"))
}

fn material_bucket(board: &ChessBoard, buckets: usize) -> usize {
    let divisor = 32usize.div_ceil(buckets);
    ((board.occ().count_ones().saturating_sub(2) as usize) / divisor).min(buckets - 1)
}

const RUNTIME_EVAL_CLAMP: i16 = 2045;

fn normalize_training_score(score: i16, scale: f32) -> i16 {
    let runtime_score = score.clamp(-RUNTIME_EVAL_CLAMP, RUNTIME_EVAL_CLAMP);
    (f32::from(runtime_score) / scale).round() as i16
}

fn phase_scale(board: &ChessBoard) -> f32 {
    let mut phase = 0_u32;
    let mut occupied = board.occ();
    let mut index = 0_usize;
    while occupied != 0 {
        let code = (board.pcs[index / 2] >> (4 * (index & 1))) & 0x0f;
        match code & 0x07 {
            1 | 2 => phase += 3,
            3 => phase += 5,
            4 => phase += 10,
            _ => {}
        }
        occupied &= occupied - 1;
        index += 1;
    }
    (128.0 + phase as f32) / 128.0
}

fn eval_bucket(score: i16) -> usize {
    match i32::from(score).unsigned_abs() {
        0..=50 => 0,
        51..=100 => 1,
        101..=300 => 2,
        301..=800 => 3,
        _ => 4,
    }
}

fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn keep_weighted(index: u64, weight: f32, maximum: f32) -> bool {
    if weight >= maximum {
        return true;
    }
    let threshold = (weight / maximum * 1_000_000.0) as u64;
    mix64(index) % 1_000_000 < threshold
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
    let records = if limit == 0 {
        available
    } else {
        limit.min(available)
    };
    if records == 0 {
        eprintln!("error: selected 0 Bullet data records");
        process::exit(1);
    }
    if source == output {
        if offset != 0 || records != total {
            eprintln!(
                "error: cannot slice Bullet data in place: {}",
                source.display()
            );
            process::exit(1);
        }
        eprintln!(
            "using existing Bullet data: {} records from {}",
            records,
            source.display()
        );
        return;
    }
    let mut input = BufReader::new(File::open(source).unwrap_or_else(|err| {
        eprintln!("error: cannot open {}: {err}", source.display());
        process::exit(1);
    }));
    input
        .seek(SeekFrom::Start(offset * record_size))
        .unwrap_or_else(|err| {
            eprintln!("error: cannot seek {}: {err}", source.display());
            process::exit(1);
        });
    let (tmp, mut output_file) = create_bullet_output(output);
    let bytes = records * record_size;
    let copied = std::io::copy(&mut input.take(bytes), &mut output_file).unwrap_or_else(|err| {
        let _ = fs::remove_file(&tmp);
        eprintln!("error: copy failed: {err}");
        process::exit(1);
    });
    if copied != bytes {
        let _ = fs::remove_file(&tmp);
        eprintln!("error: copied {copied} bytes, expected {bytes}");
        process::exit(1);
    }
    publish_bullet_output(tmp, output_file, output, records);
    eprintln!(
        "copied {} Bullet data records to {}",
        records,
        output.display()
    );
}

fn cmd_data(config: &Config) {
    let data = data_config(config);
    let source = expand_path(&data.source_binpack);
    let output = expand_path(&data.bullet_output);
    if !is_bullet_data_path(&output) {
        eprintln!(
            "error: build.data.bullet_output must end in .bullet: {}",
            output.display()
        );
        process::exit(2);
    }
    if is_deprecated_bullet_data_path(&source) {
        eprintln!(
            "error: deprecated BulletFormat extension .data is not accepted: {}; rename it to .bullet",
            source.display()
        );
        process::exit(2);
    }
    if !source.exists() {
        eprintln!("error: missing source binpack: {}", source.display());
        process::exit(1);
    }
    if is_bullet_data_path(&source) {
        if data.output_bucket_weights.iter().any(|weight| (*weight - 1.0).abs() > 1e-6) {
            eprintln!("error: weighted resampling requires an sfbinpack source");
            process::exit(2);
        }
        copy_bullet_data(&source, &output, data.offset, data.limit);
        return;
    }

    let (tmp, mut writer) = create_bullet_output(&output);
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
    let loader =
        SfBinpackLoader::new_concat_multiple(&refs, data.buffer_mb, data.threads, move |entry| {
            filter.keep(entry)
        });
    let start = Instant::now();
    let mut written = 0_u64;
    let mut skipped = 0_u64;
    let mut sampled = 0_u64;
    let mut bucket_seen = vec![0_u64; data.output_bucket_weights.len()];
    let mut bucket_written = vec![0_u64; data.output_bucket_weights.len()];
    let mut eval_seen = vec![0_u64; data.eval_bucket_weights.len()];
    let mut eval_written = vec![0_u64; data.eval_bucket_weights.len()];
    let maximum_weight = data.output_bucket_weights.iter().copied().fold(0.0_f32, f32::max)
        * data.eval_bucket_weights.iter().copied().fold(0.0_f32, f32::max);
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
        let mut selected = Vec::with_capacity(chunk.len());
        for board in chunk {
            if data.limit != 0 && written + selected.len() as u64 >= data.limit {
                break;
            }
            let bucket = material_bucket(board, data.output_bucket_weights.len());
            let eval = eval_bucket(board.score);
            bucket_seen[bucket] += 1;
            eval_seen[eval] += 1;
            let weight = data.output_bucket_weights[bucket] * data.eval_bucket_weights[eval];
            let keep = keep_weighted(sampled, weight, maximum_weight);
            sampled += 1;
            if keep {
                let mut normalized = *board;
                normalized.score = normalize_training_score(board.score, phase_scale(board));
                selected.push(normalized);
                bucket_written[bucket] += 1;
                eval_written[eval] += 1;
            }
        }
        if selected.is_empty() {
            return data.limit != 0 && written >= data.limit;
        }
        write_chunk(&mut writer, &selected).unwrap_or_else(|err| {
            let _ = fs::remove_file(&tmp);
            eprintln!("error: write failed: {err}");
            process::exit(1);
        });
        written += selected.len() as u64;
        if written % 5_000_000 < selected.len() as u64 {
            let secs = start.elapsed().as_secs_f32();
            eprintln!(
                "converted {} positions ({:.1}M/s)",
                written,
                written as f32 / secs.max(0.001) / 1e6
            );
        }
        data.limit != 0 && written >= data.limit
    });
    publish_bullet_output(tmp, writer, &output, written);
    let secs = start.elapsed().as_secs_f32();
    eprintln!(
        "done: skipped {} positions, converted {} positions to {} in {:.1}s ({:.1}M/s)",
        skipped,
        written,
        output.display(),
        secs,
        written as f32 / secs.max(0.001) / 1e6
    );
    eprintln!("weighted bucket input={bucket_seen:?} output={bucket_written:?}");
    eprintln!("weighted eval input={eval_seen:?} output={eval_written:?}");
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
    let target_superbatch = training_superbatches(config);
    let checkpoint = latest_current_checkpoint(config);
    ensure_resume_state(config, checkpoint.is_some());
    if let Some((superbatch, _)) = &checkpoint {
        if *superbatch > target_superbatch {
            eprintln!(
                "error: latest checkpoint superbatch {superbatch} exceeds target {target_superbatch}"
            );
            process::exit(1);
        }
        if *superbatch == target_superbatch {
            println!("checkpoint_complete={superbatch}");
            write_model(config);
            return;
        }
    }

    set_env("ENYO_BULLET_DATA", bullet_output.display());
    set_env("ENYO_BULLET_LOADER", training_loader(config));
    set_env("ENYO_BULLET_OUT", output.display());
    set_env("ENYO_BULLET_NET_ID", net_id(config));
    if let Some((superbatch, path)) = checkpoint {
        set_env("ENYO_BULLET_RESUME_CHECKPOINT", path.display());
        set_env("ENYO_BULLET_START_SUPERBATCH", superbatch + 1);
    } else if let Some(net) = initialize_from(config) {
        let init_weights = convert_initialize_from(config, &net);
        set_env("ENYO_BULLET_INIT_WEIGHTS", init_weights.display());
    } else if let Some(previous_run) = continue_from(config) {
        set_env(
            "ENYO_BULLET_INIT_WEIGHTS",
            latest_weight_checkpoint(config, &previous_run).display(),
        );
    }
    set_env(
        "ENYO_BULLET_MODE",
        string_at(&config.arch, "mode").unwrap_or("enyo"),
    );
    set_env("ENYO_BULLET_HIDDEN", usize_at(&config.arch, "hidden", 1024));
    set_env("ENYO_BULLET_L2", usize_at(&config.arch, "l2_size", 16));
    set_env("ENYO_BULLET_BATCH_SIZE", training_batch_size(config));
    set_env("ENYO_BULLET_BATCHES", training_batches(config));
    set_env("ENYO_BULLET_SUPERBATCHES", target_superbatch);
    set_env(
        "ENYO_BULLET_LR_SUPERBATCHES",
        training_lr_superbatches(config),
    );
    set_env("ENYO_BULLET_THREADS", training_threads(config));
    set_env("ENYO_BULLET_INIT_SEED", training_u64(config, "init_seed", 1));
    set_env("ENYO_BULLET_WDL", training_wdl(config));
    set_env("ENYO_BULLET_LR", training_lr(config));
    set_env("ENYO_BULLET_FINAL_LR", training_final_lr(config));
    set_env(
        "ENYO_BULLET_ENYO_L0_STD",
        f64_at(&config.arch, "l0_std", 8.0),
    );
    set_env(
        "ENYO_BULLET_ENYO_L1_STD",
        f64_at(&config.arch, "l1_std", 1.0),
    );
    set_env(
        "ENYO_BULLET_ENYO_L1_EXPORT_SCALE",
        f64_at(&config.arch, "l1_export_scale", 1.0),
    );
    set_env(
        "ENYO_BULLET_ENYO_INPUT_FACTORISER",
        usize::from(bool_at(&config.arch, "input_factoriser", false)),
    );
    set_env(
        "ENYO_BULLET_ENYO_INPUT_BUCKETS",
        usize_at(&config.arch, "input_buckets", 1),
    );
    set_env(
        "ENYO_BULLET_ENYO_RUNTIME_INPUT_BUCKETS",
        usize_at(
            &config.arch,
            "runtime_input_buckets",
            usize_at(&config.arch, "input_buckets", 1),
        ),
    );
    set_env(
        "ENYO_BULLET_ENYO_FEATURE_CHANNELS",
        usize_at(&config.arch, "feature_channels", 12),
    );
    set_env(
        "ENYO_BULLET_ENYO_OUTPUT_BUCKETS",
        usize_at(&config.arch, "output_buckets", 1),
    );
    set_env(
        "ENYO_BULLET_ENYO_FULL_THREATS",
        usize::from(arch_full_threats(config)),
    );
    set_env(
        "ENYO_BULLET_ENYO_SLIDER_XRAY_THREATS",
        usize::from(arch_slider_xray_threats(config)),
    );
    set_env(
        "ENYO_BULLET_ENYO_FULL_HEADS",
        usize::from(arch_full_heads(config)),
    );
    set_env(
        "ENYO_BULLET_ENYO_MIXED_ACTIVATION",
        usize::from(arch_mixed_activation(config)),
    );
    set_env(
        "ENYO_BULLET_ENYO_PSQT_RESIDUAL",
        usize::from(arch_psqt_residual(config)),
    );
    set_env(
        "ENYO_BULLET_EVAL_SCALE",
        f64_at(&config.arch, "eval_scale", 400.0),
    );
    set_env("ENYO_BULLET_SAVE_RATE", training_save_rate(config));
    set_env("ENYO_BULLET_EXPORT_INIT_ONLY", 0);
    set_env("ENYO_BULLET_TRAINABLE", training_trainable(config));
    set_env("ENYO_BULLET_WEIGHT_DECAY", training_weight_decay(config));
    set_env("ENYO_BULLET_ACTIVATION_L1", training_activation_l1(config));
    set_env("ENYO_BULLET_SFBINPACK_BUFFER_MB", data.buffer_mb);
    set_env("ENYO_BULLET_SFBINPACK_MIN_PLY", data.min_ply);
    set_env("ENYO_BULLET_SFBINPACK_MAX_ABS_CP", data.max_abs_cp);
    set_env(
        "ENYO_BULLET_SFBINPACK_QUIET_ONLY",
        usize::from(data.quiet_only),
    );

    trainer_main::run_from_env();
    write_model(config);
}

fn enyo_network_size(
    input_buckets: usize,
    feature_channels: usize,
    output_buckets: usize,
    hidden: usize,
    l2: usize,
    full_threats: bool,
    full_heads: bool,
    mixed_activation: bool,
    psqt_residual: bool,
) -> usize {
    let features = input_buckets * feature_channels * 64
        + if full_threats {
            ENYO_FULL_THREATS_DIMENSIONS
        } else {
            0
        };
    let l1 = 2 * hidden;
    let l3 = 32;
    let head_count = if full_heads { output_buckets } else { 1 };
    features * hidden * 2
        + hidden * 2
        + head_count * l1 * l2
        + head_count * l2 * 4
        + head_count * l2 * l3 * 4
        + head_count * l3 * 4
        + (if mixed_activation { l2 * l3 * 4 + l3 * 4 } else { 0 })
        + output_buckets * l3 * 4
        + output_buckets * 4
        + (if psqt_residual {
            features * output_buckets * 4 + output_buckets * 4
        } else {
            0
        })
}

fn trim_checkpoint(
    raw: &[u8],
    input_buckets: usize,
    feature_channels: usize,
    output_buckets: usize,
    hidden: usize,
    l2: usize,
    full_threats: bool,
    full_heads: bool,
    mixed_activation: bool,
    psqt_residual: bool,
) -> Vec<u8> {
    let expected = enyo_network_size(
        input_buckets,
        feature_channels,
        output_buckets,
        hidden,
        l2,
        full_threats,
        full_heads,
        mixed_activation,
        psqt_residual,
    );
    if raw.len() < expected {
        eprintln!(
            "error: checkpoint is {} bytes, expected at least {expected}",
            raw.len()
        );
        process::exit(1);
    }
    if raw.len() == expected {
        return raw.to_vec();
    }
    let trailer = &raw[expected..];
    for (idx, byte) in trailer.iter().enumerate() {
        if *byte != b"bullet"[idx % 6] {
            eprintln!(
                "error: checkpoint has unexpected {} byte trailer",
                trailer.len()
            );
            process::exit(1);
        }
    }
    raw[..expected].to_vec()
}

fn parent_bucket(
    target_bucket: usize,
    input_buckets: usize,
    runtime_input_buckets: usize,
) -> usize {
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
    full_threats: bool,
    full_heads: bool,
    mixed_activation: bool,
    psqt_residual: bool,
) -> Vec<u8> {
    let raw = trim_checkpoint(
        raw,
        input_buckets,
        feature_channels,
        output_buckets,
        hidden,
        l2,
        full_threats,
        full_heads,
        mixed_activation,
        psqt_residual,
    );
    if input_buckets == runtime_input_buckets {
        return raw;
    }
    if full_threats {
        eprintln!("error: full_threats cannot expand input buckets during export");
        process::exit(2);
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

fn pad_hidden_width(
    raw: &[u8],
    input_buckets: usize,
    feature_channels: usize,
    output_buckets: usize,
    hidden: usize,
    l2: usize,
    full_threats: bool,
    full_heads: bool,
    mixed_activation: bool,
    psqt_residual: bool,
) -> Vec<u8> {
    if hidden == ENYO_RUNTIME_HIDDEN {
        return raw.to_vec();
    }

    let raw = trim_checkpoint(
        raw,
        input_buckets,
        feature_channels,
        output_buckets,
        hidden,
        l2,
        full_threats,
        full_heads,
        mixed_activation,
        psqt_residual,
    );
    let features = input_buckets * feature_channels * 64
        + if full_threats {
            ENYO_FULL_THREATS_DIMENSIONS
        } else {
            0
        };
    let source_l0w = features * hidden * 2;
    let source_l0b = hidden * 2;
    let head_count = if full_heads { output_buckets } else { 1 };
    let source_l1w = head_count * 2 * hidden * l2;
    let target_l0w = features * ENYO_RUNTIME_HIDDEN * 2;
    let target_l0b = ENYO_RUNTIME_HIDDEN * 2;
    let target_l1w = head_count * 2 * ENYO_RUNTIME_HIDDEN * l2;
    let source_tail = source_l0w + source_l0b + source_l1w;
    let target_tail = target_l0w + target_l0b + target_l1w;
    let mut padded = vec![0_u8; target_tail + raw.len() - source_tail];

    let source_feature_bytes = hidden * 2;
    let target_feature_bytes = ENYO_RUNTIME_HIDDEN * 2;
    for feature in 0..features {
        let source = feature * source_feature_bytes;
        let target = feature * target_feature_bytes;
        padded[target..target + source_feature_bytes]
            .copy_from_slice(&raw[source..source + source_feature_bytes]);
    }

    padded[target_l0w..target_l0w + source_l0b]
        .copy_from_slice(&raw[source_l0w..source_l0w + source_l0b]);

    let source_l1_start = source_l0w + source_l0b;
    let target_l1_start = target_l0w + target_l0b;
    for row in 0..head_count * l2 {
        let source_row = source_l1_start + row * 2 * hidden;
        let target_row = target_l1_start + row * 2 * ENYO_RUNTIME_HIDDEN;
        padded[target_row..target_row + hidden]
            .copy_from_slice(&raw[source_row..source_row + hidden]);
        padded[target_row + ENYO_RUNTIME_HIDDEN..target_row + ENYO_RUNTIME_HIDDEN + hidden]
            .copy_from_slice(&raw[source_row + hidden..source_row + 2 * hidden]);
    }

    padded[target_tail..].copy_from_slice(&raw[source_tail..]);
    padded
}

fn write_u32_le(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn enyo_container(
    payload: &[u8],
    input_buckets: usize,
    feature_channels: usize,
    hidden: usize,
    l2: usize,
    output_buckets: usize,
    full_threats: bool,
    slider_xray_threats: bool,
    full_heads: bool,
    mixed_activation: bool,
    psqt_residual: bool,
    pairwise: bool,
    format_version: u32,
) -> Vec<u8> {
    let payload_size = u32::try_from(payload.len()).unwrap_or_else(|_| {
        eprintln!("error: Enyo network payload is too large");
        process::exit(1);
    });
    let mut output = vec![0_u8; ENYO_NETWORK_HEADER_SIZE + payload.len()];
    let magic = if format_version == ENYO_V7_FORMAT_VERSION {
        ENYO_V7_HEADER_MAGIC
    } else if format_version == ENYO_V6_FORMAT_VERSION {
        ENYO_V6_HEADER_MAGIC
    } else if format_version == ENYO_V5_FORMAT_VERSION {
        ENYO_V5_HEADER_MAGIC
    } else if format_version == ENYO_V4_FORMAT_VERSION {
        ENYO_V4_HEADER_MAGIC
    } else if format_version == ENYO_V3_FORMAT_VERSION {
        ENYO_V3_HEADER_MAGIC
    } else {
        ENYO_V2_HEADER_MAGIC
    };
    output[..magic.len()].copy_from_slice(magic);
    write_u32_le(&mut output, 8, format_version);
    write_u32_le(&mut output, 12, ENYO_NETWORK_HEADER_SIZE as u32);
    write_u32_le(&mut output, 16, input_buckets as u32);
    write_u32_le(&mut output, 20, feature_channels as u32);
    write_u32_le(&mut output, 24, hidden as u32);
    write_u32_le(&mut output, 28, ENYO_RUNTIME_HIDDEN as u32);
    write_u32_le(&mut output, 32, l2 as u32);
    write_u32_le(&mut output, 36, 32);
    write_u32_le(&mut output, 40, output_buckets as u32);
    write_u32_le(&mut output, 44, 0);
    write_u32_le(
        &mut output,
        48,
        (if full_threats {
            ENYO_NETWORK_FLAG_FULL_THREATS
        } else {
            0
        }) | (if slider_xray_threats {
            ENYO_NETWORK_FLAG_SLIDER_XRAY_THREATS
        } else {
            0
        }) | (if full_heads {
            ENYO_NETWORK_FLAG_FULL_HEADS
        } else {
            0
        }) | (if mixed_activation {
            ENYO_NETWORK_FLAG_MIXED_ACTIVATION
        } else {
            0
        }) | (if psqt_residual {
            ENYO_NETWORK_FLAG_PSQT_RESIDUAL
        } else {
            0
        }) | (if pairwise {
            ENYO_NETWORK_FLAG_PAIRWISE
        } else {
            0
        }) | (if format_version == ENYO_V7_FORMAT_VERSION {
            ENYO_NETWORK_FLAG_RECKLESS_THREATS
        } else {
            0
        }),
    );
    write_u32_le(&mut output, 52, payload_size);
    output[ENYO_NETWORK_HEADER_SIZE..].copy_from_slice(payload);
    output
}

fn reckless_payload(raw: &[u8], hidden: usize, l2: usize, output_buckets: usize) -> Vec<u8> {
    let features = 10 * 12 * 64;
    let source_l0w = features * hidden * 2;
    let threat_l0w = RECKLESS_THREAT_DIMENSIONS * hidden;
    let source_l0b = hidden * 2;
    let l1w = output_buckets * hidden * l2;
    let l1b = output_buckets * l2 * 4;
    let l2w = output_buckets * l2 * 32 * 4;
    let l2b = output_buckets * 32 * 4;
    let l3w = output_buckets * 32 * 4;
    let l3b = output_buckets * 4;
    let expected = source_l0w + threat_l0w + source_l0b + l1w + l1b + l2w + l2b + l3w + l3b;
    if raw.len() < expected {
        eprintln!("error: Reckless checkpoint is {} bytes, expected at least {expected}", raw.len());
        process::exit(1);
    }
    if raw[expected..]
        .iter()
        .enumerate()
        .any(|(idx, byte)| *byte != b"bullet"[idx % 6])
    {
        eprintln!("error: Reckless checkpoint has unexpected {} byte trailer", raw.len() - expected);
        process::exit(1);
    }

    let target_l0w = features * ENYO_RUNTIME_HIDDEN * 2;
    let target_l0b = ENYO_RUNTIME_HIDDEN * 2;
    let source_bias = source_l0w + threat_l0w;
    let source_tail = source_bias + source_l0b;
    let target_bias = target_l0w + threat_l0w;
    let target_tail = target_bias + target_l0b;
    let mut payload = vec![0_u8; target_tail + expected - source_tail];
    for feature in 0..features {
        let source = feature * hidden * 2;
        let target = feature * ENYO_RUNTIME_HIDDEN * 2;
        payload[target..target + hidden * 2]
            .copy_from_slice(&raw[source..source + hidden * 2]);
    }
    payload[target_l0w..target_bias].copy_from_slice(&raw[source_l0w..source_bias]);
    payload[target_bias..target_bias + source_l0b]
        .copy_from_slice(&raw[source_bias..source_tail]);
    payload[target_tail..].copy_from_slice(&raw[source_tail..expected]);
    payload
}

fn checkpoint_dirs(output: &Path, net_id: &str) -> Vec<(usize, PathBuf)> {
    let prefix = format!("{net_id}-");
    let entries = match fs::read_dir(output) {
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
            Some((superbatch, path))
        })
        .collect::<Vec<_>>();
    checkpoints.sort_by_key(|(superbatch, _)| *superbatch);
    checkpoints
}

fn latest_current_checkpoint(config: &Config) -> Option<(usize, PathBuf)> {
    let output = expand_path(&out_dir(config));
    checkpoint_dirs(&output, &net_id(config))
        .into_iter()
        .filter(|(_, path)| {
            path.join("optimiser_state/weights.bin").is_file()
                && path.join("optimiser_state/momentum.bin").is_file()
                && path.join("optimiser_state/velocity.bin").is_file()
                && path.join("quantised.bin").is_file()
        })
        .last()
}

fn latest_checkpoint(config: &Config) -> PathBuf {
    let output = expand_path(&out_dir(config));
    let checkpoint = checkpoint_dirs(&output, &net_id(config))
        .into_iter()
        .filter(|(_, path)| path.join("quantised.bin").is_file())
        .last();
    checkpoint
        .map(|(_, path)| path.join("quantised.bin"))
        .unwrap_or_else(|| {
            eprintln!(
                "error: no quantised.bin checkpoints found under {}",
                output.display()
            );
            process::exit(1);
        })
}

fn latest_weight_checkpoint(config: &Config, run: &str) -> PathBuf {
    if run.contains('/') || run.contains('\\') || run == "." || run == ".." {
        eprintln!("error: continue_from must be a previous run name, not a path");
        process::exit(2);
    }
    let output = expand_path(&format!("runs/{run}/checkpoints"));
    if let Some((_, path)) = checkpoint_dirs(&output, &net_id(config))
        .into_iter()
        .filter(|(_, path)| path.join("optimiser_state/weights.bin").is_file())
        .last()
    {
        return path.join("optimiser_state/weights.bin");
    }

    for net in [
        format!("runs/{run}/model.nn"),
        format!("~/assets/nets/{run}.nn"),
    ] {
        let net = expand_path(&net);
        if net.exists() {
            let net = net.to_string_lossy().into_owned();
            return convert_initialize_from(config, &net);
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

fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|err| format!("cannot open {}: {err}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|err| format!("cannot read {}: {err}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn config_sha256(config: &Config) -> Result<String, String> {
    let resolved = serde_json::json!({
        "architecture": &config.arch,
        "defaults": &config.defaults,
        "build": &config.build,
    });
    serde_json::to_vec(&resolved)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|err| format!("cannot serialize resolved training config: {err}"))
}

fn resume_config_sha256(config: &Config) -> Result<String, String> {
    let mut build = config.build.clone();
    if let Some(build) = build.as_object_mut() {
        build.remove("superbatches");
    }
    let resolved = serde_json::json!({
        "architecture": &config.arch,
        "defaults": &config.defaults,
        "build": build,
    });
    serde_json::to_vec(&resolved)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|err| format!("cannot serialize resumable training config: {err}"))
}

fn resume_state_path(config: &Config) -> PathBuf {
    expand_path(&format!("runs/{}/training.resume.json", run_name(config)))
}

fn resume_state(config: &Config) -> Result<Value, String> {
    Ok(serde_json::json!({
        "schema": 1,
        "run": run_name(config),
        "resume_config_sha256": resume_config_sha256(config)?,
        "lr_superbatches": training_lr_superbatches(config),
    }))
}

fn ensure_resume_state(config: &Config, has_checkpoint: bool) {
    let path = resume_state_path(config);
    let expected = resume_state(config).unwrap_or_else(|err| {
        eprintln!("error: cannot resolve resume state: {err}");
        process::exit(1);
    });
    if path.exists() {
        let actual = read_provenance(&path).unwrap_or_else(|| {
            eprintln!("error: invalid resume state: {}", path.display());
            process::exit(1);
        });
        if actual != expected {
            if !has_checkpoint {
                write_json_atomic(&path, &expected).unwrap_or_else(|err| {
                    eprintln!("error: cannot replace resume state: {err}");
                    process::exit(1);
                });
                return;
            }
            eprintln!("error: checkpoint config does not match current resumable training config");
            eprintln!("  state={}", path.display());
            process::exit(1);
        }
        return;
    }
    if has_checkpoint {
        eprintln!(
            "error: refusing to resume checkpoints without {}",
            path.display()
        );
        process::exit(1);
    }
    write_json_atomic(&path, &expected).unwrap_or_else(|err| {
        eprintln!("error: cannot write resume state: {err}");
        process::exit(1);
    });
}

fn resolved_init_weights(config: &Config) -> Option<PathBuf> {
    if initialize_from(config).is_some() {
        Some(init_weights_path(config))
    } else {
        continue_from(config).map(|run| latest_weight_checkpoint(config, &run))
    }
}

fn training_inputs(
    config: &Config,
    init_weights: Option<&Path>,
    trainer: &Path,
) -> Result<Value, String> {
    let init_weights_sha256 = init_weights.map(sha256_file).transpose()?;
    Ok(serde_json::json!({
        "config_sha256": config_sha256(config)?,
        "init_weights": init_weights.map(|path| path.display().to_string()),
        "init_weights_sha256": init_weights_sha256,
        "trainer_sha256": sha256_file(trainer)?,
    }))
}

fn model_path(config: &Config) -> PathBuf {
    expand_path(&format!("runs/{}/model.nn", run_name(config)))
}

fn provenance_path(config: &Config) -> PathBuf {
    expand_path(&format!("runs/{}/train.provenance.json", run_name(config)))
}

fn read_provenance(path: &Path) -> Option<Value> {
    fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    }
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("provenance");
    let tmp = path.with_file_name(format!("{name}.tmp.{}", process::id()));
    let mut file =
        File::create(&tmp).map_err(|err| format!("cannot create {}: {err}", tmp.display()))?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|err| format!("cannot write {}: {err}", tmp.display()))?;
    file.write_all(b"\n")
        .map_err(|err| format!("cannot write {}: {err}", tmp.display()))?;
    file.sync_all()
        .map_err(|err| format!("cannot sync {}: {err}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        format!("cannot publish {}: {err}", path.display())
    })
}

fn provenance_inputs_match(provenance: &Value, run: &str, inputs: &Value) -> bool {
    provenance.get("schema").and_then(Value::as_u64) == Some(TRAIN_PROVENANCE_SCHEMA)
        && provenance.get("run").and_then(Value::as_str) == Some(run)
        && provenance.get("inputs") == Some(inputs)
}

fn provenance_artifact_matches(provenance: &Value, key: &str, path: &Path) -> bool {
    path.is_file()
        && sha256_file(path)
            .is_ok_and(|hash| provenance.get(key).and_then(Value::as_str) == Some(hash.as_str()))
}

fn training_provenance(
    config: &Config,
    init_weights: Option<&Path>,
    trainer: &Path,
    model: &Path,
    candidate: &Path,
) -> Result<Value, String> {
    Ok(serde_json::json!({
        "schema": TRAIN_PROVENANCE_SCHEMA,
        "run": run_name(config),
        "inputs": training_inputs(config, init_weights, trainer)?,
        "model_sha256": sha256_file(model)?,
        "candidate_sha256": sha256_file(candidate)?,
    }))
}

fn write_training_provenance(config: &Config) -> Result<(), String> {
    let model = model_path(config);
    let candidate = expand_path(&net_path(config));
    let init_weights = resolved_init_weights(config);
    let trainer =
        env::current_exe().map_err(|err| format!("cannot resolve trainer executable: {err}"))?;
    let provenance = training_provenance(
        config,
        init_weights.as_deref(),
        &trainer,
        &model,
        &candidate,
    )?;
    write_json_atomic(&provenance_path(config), &provenance)
}

fn stale_training_artifacts(config: &Config, detail: &str) -> ! {
    eprintln!(
        "error: stale train/export artifacts for {}; remove them or pass --force",
        run_name(config)
    );
    if !detail.is_empty() {
        eprintln!("  {detail}");
    }
    process::exit(1);
}

fn copy_file_atomic(source: &Path, destination: &Path) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
    }
    let name = destination
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("net");
    let tmp = destination.with_file_name(format!("{name}.tmp.{}", process::id()));
    fs::copy(source, &tmp).map_err(|err| {
        format!(
            "cannot copy {} to {}: {err}",
            source.display(),
            tmp.display()
        )
    })?;
    File::open(&tmp)
        .and_then(|file| file.sync_all())
        .map_err(|err| format!("cannot sync {}: {err}", tmp.display()))?;
    fs::rename(&tmp, destination).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        format!("cannot publish {}: {err}", destination.display())
    })
}

fn ensure_training_data(config: &Config) {
    let output = expand_path(&data_config(config).bullet_output);
    if !output.is_file() {
        eprintln!("rebuilding missing Bullet data: {}", output.display());
        cmd_data(config);
    }
}

fn resume_training(config: &Config, force: bool) -> bool {
    if force {
        return false;
    }

    let model = model_path(config);
    let candidate = expand_path(&net_path(config));
    if !model.is_file() && !candidate.is_file() {
        return false;
    }
    if latest_current_checkpoint(config)
        .is_some_and(|(superbatch, _)| superbatch < training_superbatches(config))
    {
        ensure_resume_state(config, true);
        return false;
    }

    let path = provenance_path(config);
    let provenance = read_provenance(&path)
        .unwrap_or_else(|| stale_training_artifacts(config, "missing or invalid provenance"));
    let init_weights = resolved_init_weights(config);
    let trainer =
        env::current_exe().unwrap_or_else(|err| stale_training_artifacts(config, &err.to_string()));
    let inputs = training_inputs(config, init_weights.as_deref(), &trainer)
        .unwrap_or_else(|err| stale_training_artifacts(config, &err));
    if !provenance_inputs_match(&provenance, &run_name(config), &inputs) {
        stale_training_artifacts(config, "training inputs changed");
    }

    if provenance_artifact_matches(&provenance, "candidate_sha256", &candidate) {
        ensure_training_data(config);
        println!("training_artifacts=reused");
        return true;
    }
    if provenance_artifact_matches(&provenance, "model_sha256", &model) {
        copy_file_atomic(&model, &candidate)
            .unwrap_or_else(|err| stale_training_artifacts(config, &err));
        write_training_provenance(config)
            .unwrap_or_else(|err| stale_training_artifacts(config, &err));
        ensure_training_data(config);
        println!("training_artifacts=restored");
        return true;
    }

    stale_training_artifacts(config, "artifact checksum mismatch");
}

fn cmd_all(config: &Config, force: bool) {
    let extending = latest_current_checkpoint(config)
        .is_some_and(|(superbatch, _)| superbatch < training_superbatches(config));
    if resume_training(config, force) {
        return;
    }
    if extending {
        ensure_training_data(config);
    } else {
        cmd_data(config);
    }
    cmd_run(config);
    cmd_export(config, force || extending);
    write_training_provenance(config).unwrap_or_else(|err| {
        eprintln!("error: cannot write training provenance: {err}");
        process::exit(1);
    });
    println!("training_artifacts=created");
}

fn write_model(config: &Config) {
    let checkpoint = latest_checkpoint(config);
    let raw = fs::read(&checkpoint).unwrap_or_else(|err| {
        eprintln!("error: cannot read {}: {err}", checkpoint.display());
        process::exit(1);
    });
    if string_at(&config.arch, "mode") == Some("reckless") {
        let hidden = usize_at(&config.arch, "hidden", 768);
        let l2 = usize_at(&config.arch, "l2_size", 16);
        let output_buckets = usize_at(&config.arch, "output_buckets", 8);
        let payload = reckless_payload(&raw, hidden, l2, output_buckets);
        let model = enyo_container(
            &payload,
            10,
            12,
            hidden,
            l2,
            output_buckets,
            false,
            false,
            true,
            false,
            false,
            true,
            ENYO_V7_FORMAT_VERSION,
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
        return;
    }
    let input_buckets = usize_at(&config.arch, "input_buckets", 1);
    let runtime_input_buckets = usize_at(&config.arch, "runtime_input_buckets", input_buckets);
    let feature_channels = usize_at(&config.arch, "feature_channels", 12);
    let output_buckets = usize_at(&config.arch, "output_buckets", 1);
    let hidden = usize_at(&config.arch, "hidden", ENYO_RUNTIME_HIDDEN);
    let l2 = usize_at(&config.arch, "l2_size", 16);
    let full_threats = arch_full_threats(config);
    let slider_xray_threats = arch_slider_xray_threats(config);
    let threat_features = full_threats || slider_xray_threats;
    let full_heads = arch_full_heads(config);
    let psqt_residual = arch_psqt_residual(config);
    let mixed_activation = arch_mixed_activation(config);
    let model = expand_input_buckets(
        &raw,
        input_buckets,
        feature_channels,
        runtime_input_buckets,
        output_buckets,
        hidden,
        l2,
        threat_features,
        full_heads,
        mixed_activation,
        psqt_residual,
    );
    let model = pad_hidden_width(
        &model,
        runtime_input_buckets,
        feature_channels,
        output_buckets,
        hidden,
        l2,
        threat_features,
        full_heads,
        mixed_activation,
        psqt_residual,
    );
    let model = match string_at(&config.arch, "export_format") {
        Some("enyo-native-v2") => enyo_container(
            &model,
            runtime_input_buckets,
            feature_channels,
            hidden,
            l2,
            output_buckets,
            full_threats,
            slider_xray_threats,
            false,
            false,
            false,
            false,
            ENYO_V2_FORMAT_VERSION,
        ),
        Some("enyo-native-v3") => enyo_container(
            &model,
            runtime_input_buckets,
            feature_channels,
            hidden,
            l2,
            output_buckets,
            false,
            false,
            full_heads,
            false,
            false,
            false,
            ENYO_V3_FORMAT_VERSION,
        ),
        Some("enyo-native-v4") => enyo_container(
            &model,
            runtime_input_buckets,
            feature_channels,
            hidden,
            l2,
            output_buckets,
            false,
            false,
            false,
            mixed_activation,
            false,
            false,
            ENYO_V4_FORMAT_VERSION,
        ),
        Some("enyo-native-v5") => enyo_container(
            &model,
            runtime_input_buckets,
            feature_channels,
            hidden,
            l2,
            output_buckets,
            false,
            false,
            false,
            false,
            psqt_residual,
            false,
            ENYO_V5_FORMAT_VERSION,
        ),
        Some("enyo-native-v6") => enyo_container(
            &model,
            runtime_input_buckets,
            feature_channels,
            hidden,
            l2,
            output_buckets,
            full_threats,
            slider_xray_threats,
            full_heads,
            false,
            false,
            false,
            ENYO_V6_FORMAT_VERSION,
        ),
        _ => model,
    };
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
            "net_id": "native",
            "batches": 64,
            "batch_size": 2048,
            "superbatches": 7600,
            "lr_superbatches": 0,
            "threads": 16,
            "init_seed": 1,
            "wdl": 0.3,
            "lr": 0.001,
            "final_lr": 0.000005,
            "save_rate": 7600,
            "trainable": "all",
            "weight_decay": 0.0,
            "activation_l1": 0.0,
            "output_bucket_weights": "auto",
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
            arch: json!({"lineage": "native"}),
            defaults: defaults(),
            build,
        }
    }

    fn test_dir(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = env::temp_dir().join(format!("enyo-train-{name}-{}-{nonce}", process::id()));
        fs::create_dir_all(&path).expect("create test directory");
        path
    }

    #[test]
    fn build_overrides_top_level_training_defaults() {
        let config = config(json!({
            "run": "candidate",
            "loader": "sfbinpack",
            "net_id": "override_id",
            "batches": 32,
            "batch_size": 4096,
            "superbatches": 3800,
            "lr_superbatches": 12000,
            "threads": 8,
            "wdl": 0.25,
            "lr": 0.0005,
            "final_lr": 0.000001,
            "save_rate": 1900,
            "trainable": "input",
            "weight_decay": 0.00001,
            "activation_l1": 0.00002,
            "data": {"source_binpack": "data.binpack", "limit": 100, "offset": 20}
        }));

        assert!(config_contract_errors(&config).is_empty());
        assert_eq!(training_loader(&config), "sfbinpack");
        assert_eq!(net_id(&config), "override_id");
        assert_eq!(training_batches(&config), 32);
        assert_eq!(training_batch_size(&config), 4096);
        assert_eq!(training_superbatches(&config), 3800);
        assert_eq!(training_lr_superbatches(&config), 12000);
        assert_eq!(training_threads(&config), 8);
        assert_eq!(training_wdl(&config), 0.25);
        assert_eq!(training_lr(&config), 0.0005);
        assert_eq!(training_final_lr(&config), 0.000001);
        assert_eq!(training_save_rate(&config), 1900);
        assert_eq!(training_trainable(&config), "input");
        assert_eq!(training_weight_decay(&config), 0.00001);
        assert_eq!(training_activation_l1(&config), 0.00002);
    }

    #[test]
    fn zero_lr_schedule_uses_training_stop() {
        let config = config(json!({
            "run": "candidate",
            "superbatches": 16384,
            "data": {"source_binpack": "data.binpack", "limit": 100}
        }));

        assert_eq!(training_lr_superbatches(&config), 16384);
    }

    #[test]
    fn dense_head_is_supported_and_unknown_mode_is_rejected() {
        assert!(supported_trainable("dense-head"));
        assert!(supported_trainable("xray-only"));
        assert!(supported_trainable("squared-branch"));
        assert!(supported_trainable("frozen-output"));
        assert!(!supported_trainable("typo"));
    }

    #[test]
    fn build_overrides_nested_sfbinpack_defaults() {
        let config = config(json!({
            "run": "candidate",
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
            "sfbinpack": {"offset": 500},
            "data": {"source_binpack": "data.binpack", "limit": 100, "offset": 900, "threads": 3}
        }));

        assert!(config_contract_errors(&config).is_empty());
        let data = data_config(&config);
        assert_eq!(data.offset, 900);
        assert_eq!(data.threads, 3);
    }

    #[test]
    fn unsliced_bullet_source_is_used_in_place() {
        let config = config(json!({
            "run": "candidate",
            "data": {"source_binpack": "data/shared.bullet"}
        }));

        let data = data_config(&config);
        assert_eq!(data.bullet_output, "data/shared.bullet");
    }

    #[test]
    fn sliced_bullet_source_keeps_run_specific_output() {
        let config = config(json!({
            "run": "candidate",
            "data": {"source_binpack": "data/shared.bullet", "limit": 100}
        }));

        let data = data_config(&config);
        assert_eq!(data.bullet_output, "data/bullet/candidate.bullet");
    }

    #[test]
    fn unknown_build_training_key_is_rejected() {
        let config = config(json!({
            "run": "candidate",
            "new_training_knob": 1,
            "data": {"source_binpack": "data.binpack"}
        }));

        let errors = config_contract_errors(&config);
        assert!(errors
            .iter()
            .any(|err| err.contains("build.new_training_knob")));
    }

    #[test]
    fn initialize_from_and_continue_from_are_accepted_together() {
        let config = config(json!({
            "run": "uho-native-1.1.0",
            "continue_from": "uho-native-1.0.42",
            "initialize_from": "~/assets/nets/uho-native-1.0.42.nn",
            "reference": "uho-native-1.0.42",
            "data": {"source_binpack": "data.binpack", "limit": 100}
        }));

        assert!(config_contract_errors(&config).is_empty());
        assert_eq!(continue_from(&config).as_deref(), Some("uho-native-1.0.42"));
        assert_eq!(
            initialize_from(&config).as_deref(),
            Some("~/assets/nets/uho-native-1.0.42.nn")
        );
    }

    #[test]
    fn initialize_from_bare_name_resolves_to_exported_net() {
        let home = env::var("HOME").expect("HOME");
        let net = Path::new(&home).join("assets/nets/native-test-init.nn");
        fs::create_dir_all(net.parent().expect("net parent")).expect("create net parent");
        fs::write(&net, b"net").expect("write net");

        assert_eq!(initialize_from_path("native-test-init"), net);

        let _ = fs::remove_file(net);
    }

    #[test]
    fn data_training_knobs_are_rejected() {
        let config = config(json!({
            "run": "candidate",
            "data": {"source_binpack": "data.binpack", "buffer_mb": 2048}
        }));

        let errors = config_contract_errors(&config);
        assert!(errors
            .iter()
            .any(|err| err.contains("build.data.buffer_mb")));
    }

    #[test]
    fn deprecated_data_extension_is_rejected() {
        let config = config(json!({
            "run": "candidate",
            "data": {
                "source_binpack": "rows.data",
                "bullet_output": "data/bullet/candidate.data"
            }
        }));

        let errors = config_contract_errors(&config);
        assert!(errors
            .iter()
            .any(|err| err.contains("build.data.source_binpack")));
        assert!(errors
            .iter()
            .any(|err| err.contains("build.data.bullet_output")));
    }

    #[test]
    fn missing_default_training_parameter_is_rejected() {
        let mut config = config(json!({
            "run": "candidate",
            "data": {"source_binpack": "data.binpack"}
        }));
        config.defaults.as_object_mut().unwrap().remove("wdl");

        let errors = config_contract_errors(&config);
        assert!(errors.iter().any(|err| err.contains("defaults.wdl")));
    }

    #[test]
    fn provenance_validates_inputs_and_artifacts() {
        let dir = test_dir("provenance");
        let init = dir.join("init.bin");
        let trainer = dir.join("train");
        let model = dir.join("model.nn");
        let candidate = dir.join("candidate.nn");
        let manifest = dir.join("train.provenance.json");
        fs::write(&init, b"initial weights").expect("write init");
        fs::write(&trainer, b"trainer").expect("write trainer");
        fs::write(&model, b"model").expect("write model");
        fs::write(&candidate, b"candidate").expect("write candidate");

        let config = config(json!({
            "run": "candidate",
            "continue_from": "parent",
            "data": {"source_binpack": "data.binpack", "limit": 100}
        }));
        let provenance = training_provenance(&config, Some(&init), &trainer, &model, &candidate)
            .expect("create provenance");
        write_json_atomic(&manifest, &provenance).expect("write provenance");
        let loaded = read_provenance(&manifest).expect("read provenance");
        let inputs = training_inputs(&config, Some(&init), &trainer).expect("hash inputs");

        assert!(provenance_inputs_match(&loaded, "candidate", &inputs));
        assert!(provenance_artifact_matches(&loaded, "model_sha256", &model));
        assert!(provenance_artifact_matches(
            &loaded,
            "candidate_sha256",
            &candidate
        ));

        fs::write(&candidate, b"changed").expect("change candidate");
        assert!(!provenance_artifact_matches(
            &loaded,
            "candidate_sha256",
            &candidate
        ));
        fs::write(&init, b"changed init").expect("change init");
        let changed_inputs =
            training_inputs(&config, Some(&init), &trainer).expect("rehash inputs");
        assert!(!provenance_inputs_match(
            &loaded,
            "candidate",
            &changed_inputs
        ));

        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn phase_normalized_training_score_roundtrips_runtime_units() {
        let scale = 1.375_f32;
        let normalized = normalize_training_score(640, scale);
        assert_eq!(normalized, 465);
        assert!((f32::from(normalized) * scale - 640.0).abs() < 1.0);

        let capped = normalize_training_score(10_000, scale);
        assert!((f32::from(capped) * scale - f32::from(RUNTIME_EVAL_CLAMP)).abs() < 1.0);
    }

    #[test]
    fn resume_hash_ignores_only_training_stop() {
        let first = config(json!({
            "run": "candidate",
            "superbatches": 16384,
            "lr_superbatches": 65536,
            "data": {"source_binpack": "data.bullet", "limit": 0}
        }));
        let mut extended = first.clone();
        extended.build["superbatches"] = json!(65536);
        assert_eq!(
            resume_config_sha256(&first).expect("first hash"),
            resume_config_sha256(&extended).expect("extended hash")
        );

        extended.build["lr"] = json!(0.0005);
        assert_ne!(
            resume_config_sha256(&first).expect("first hash"),
            resume_config_sha256(&extended).expect("changed hash")
        );
    }

    #[test]
    fn narrow_hidden_padding_preserves_l1_perspectives() {
        let input_buckets = 1;
        let feature_channels = 1;
        let output_buckets = 1;
        let hidden = 2;
        let l2 = 1;
        let size = enyo_network_size(
            input_buckets,
            feature_channels,
            output_buckets,
            hidden,
            l2,
            false,
            false,
            false,
            false,
        );
        let mut raw = vec![0_u8; size];
        let source_l0w = input_buckets * feature_channels * 64 * hidden * 2;
        let source_l0b = hidden * 2;
        let source_l1 = source_l0w + source_l0b;
        raw[source_l1..source_l1 + 2 * hidden].copy_from_slice(&[1, 2, 3, 4]);

        let padded = pad_hidden_width(
            &raw,
            input_buckets,
            feature_channels,
            output_buckets,
            hidden,
            l2,
            false,
            false,
            false,
            false,
        );
        let target_l0w = input_buckets * feature_channels * 64 * ENYO_RUNTIME_HIDDEN * 2;
        let target_l0b = ENYO_RUNTIME_HIDDEN * 2;
        let target_l1 = target_l0w + target_l0b;
        assert_eq!(&padded[target_l1..target_l1 + hidden], &[1, 2]);
        assert_eq!(
            &padded[target_l1 + ENYO_RUNTIME_HIDDEN..target_l1 + ENYO_RUNTIME_HIDDEN + hidden],
            &[3, 4]
        );
    }

    #[test]
    fn psqt_residual_size_includes_bucketed_piece_square_tail() {
        let input_buckets = 16;
        let feature_channels = 12;
        let output_buckets = 8;
        let hidden = 1024;
        let l2 = 16;
        let base = enyo_network_size(
            input_buckets,
            feature_channels,
            output_buckets,
            hidden,
            l2,
            false,
            false,
            false,
            false,
        );
        let with_psqt = enyo_network_size(
            input_buckets,
            feature_channels,
            output_buckets,
            hidden,
            l2,
            false,
            false,
            false,
            true,
        );
        let features = input_buckets * feature_channels * 64;
        assert_eq!(
            with_psqt - base,
            features * output_buckets * 4 + output_buckets * 4,
        );

        let payload = vec![0_u8; with_psqt];
        assert_eq!(
            trim_checkpoint(
                &payload,
                input_buckets,
                feature_channels,
                output_buckets,
                hidden,
                l2,
                false,
                false,
                false,
                true,
            )
            .len(),
            with_psqt,
        );
    }

    #[test]
    fn versioned_header_records_trained_architecture() {
        let payload = vec![7_u8; 32];
        let container = enyo_container(
            &payload,
            10,
            11,
            768,
            16,
            8,
            false,
            false,
            true,
            false,
            false,
            false,
            ENYO_V3_FORMAT_VERSION,
        );
        assert_eq!(&container[..8], ENYO_V3_HEADER_MAGIC);
        assert_eq!(u32::from_le_bytes(container[8..12].try_into().unwrap()), 3);
        assert_eq!(
            u32::from_le_bytes(container[16..20].try_into().unwrap()),
            10
        );
        assert_eq!(
            u32::from_le_bytes(container[20..24].try_into().unwrap()),
            11
        );
        assert_eq!(
            u32::from_le_bytes(container[24..28].try_into().unwrap()),
            768
        );
        assert_eq!(u32::from_le_bytes(container[40..44].try_into().unwrap()), 8);
        assert_eq!(
            u32::from_le_bytes(container[48..52].try_into().unwrap()),
            ENYO_NETWORK_FLAG_FULL_HEADS
        );
        assert_eq!(
            u32::from_le_bytes(container[52..56].try_into().unwrap()),
            32
        );
        assert_eq!(&container[ENYO_NETWORK_HEADER_SIZE..], payload);
    }

    #[test]
    fn reckless_payload_pads_only_accumulator_width_and_preserves_dense_tail() {
        let hidden = 768;
        let l2 = 16;
        let output_buckets = 8;
        let features = 10 * 12 * 64;
        let source_l0w = features * hidden * 2;
        let threat_l0w = RECKLESS_THREAT_DIMENSIONS * hidden;
        let source_l0b = hidden * 2;
        let dense_tail = output_buckets * hidden * l2
            + output_buckets * l2 * 4
            + output_buckets * l2 * 32 * 4
            + output_buckets * 32 * 4
            + output_buckets * 32 * 4
            + output_buckets * 4;
        let mut raw = vec![0_u8; source_l0w + threat_l0w + source_l0b + dense_tail];
        raw[0..4].copy_from_slice(&[1, 2, 3, 4]);
        raw[(hidden * 2)..(hidden * 2 + 4)].copy_from_slice(&[5, 6, 7, 8]);
        raw[source_l0w..source_l0w + 4].copy_from_slice(&[21, 22, 23, 24]);
        raw[source_l0w + threat_l0w..source_l0w + threat_l0w + 4]
            .copy_from_slice(&[9, 10, 11, 12]);
        raw[source_l0w + threat_l0w + source_l0b..].fill(13);
        raw.extend_from_slice(b"bulletbullet");

        let payload = reckless_payload(&raw, hidden, l2, output_buckets);
        let target_l0w = features * ENYO_RUNTIME_HIDDEN * 2;
        let target_threat = RECKLESS_THREAT_DIMENSIONS * hidden;
        let target_l0b = ENYO_RUNTIME_HIDDEN * 2;
        assert_eq!(&payload[0..4], &[1, 2, 3, 4]);
        assert!(payload[hidden * 2..ENYO_RUNTIME_HIDDEN * 2]
            .iter()
            .all(|byte| *byte == 0));
        assert_eq!(
            &payload[ENYO_RUNTIME_HIDDEN * 2..ENYO_RUNTIME_HIDDEN * 2 + 4],
            &[5, 6, 7, 8]
        );
        assert_eq!(&payload[target_l0w..target_l0w + 4], &[21, 22, 23, 24]);
        assert_eq!(&payload[target_l0w + target_threat..target_l0w + target_threat + 4], &[9, 10, 11, 12]);
        assert!(payload[target_l0w + target_threat + source_l0b..target_l0w + target_threat + target_l0b]
            .iter()
            .all(|byte| *byte == 0));
        assert!(payload[target_l0w + target_threat + target_l0b..]
            .iter()
            .all(|byte| *byte == 13));
        assert_eq!(payload.len(), target_l0w + target_threat + target_l0b + dense_tail);
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
        "all" => cmd_all(&config, force),
        _ => usage(),
    }
}
