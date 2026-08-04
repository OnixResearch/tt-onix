use rwkv_layer_harness::{
    FullDeviceReferenceStep, PromptRequest, TokenizerAuthorityInputs, parse_prompt_arguments,
    run_full_device_reference,
};
use serde_json::{Value, json};
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const MODEL_PATH: Option<&str> = option_env!("RWKV_LAYER_MODEL");
const EXPECTED_MODEL_BLAKE3: Option<&str> = option_env!("RWKV_LAYER_MODEL_BLAKE3");
const VOCABULARY_PATH: Option<&str> = option_env!("RWKV_TOKENIZER_VOCABULARY");
const TOKENIZER_CONFIG_PATH: Option<&str> = option_env!("RWKV_TOKENIZER_CONFIG");
const ADDED_TOKENS_PATH: Option<&str> = option_env!("RWKV_TOKENIZER_ADDED_TOKENS");
const TOKENIZER_IMPLEMENTATION_PATH: Option<&str> = option_env!("RWKV_TOKENIZER_IMPLEMENTATION");
const SPECIAL_TOKENS_MAP_PATH: Option<&str> = option_env!("RWKV_SPECIAL_TOKENS_MAP");
const MODEL_CONFIG_PATH: Option<&str> = option_env!("RWKV_MODEL_CONFIG");
const GENERATION_CONFIG_PATH: Option<&str> = option_env!("RWKV_GENERATION_CONFIG");
const PROGRAM_ARGUMENT_COUNT: usize = 1;
const PROMPT_ARGUMENT_COUNT: usize = 6;
const OUTPUT_ARGUMENT_COUNT: usize = 2;
const EXPECTED_ARGUMENT_COUNT: usize = PROMPT_ARGUMENT_COUNT + OUTPUT_ARGUMENT_COUNT;
const OUTPUT_OPTION_INDEX: usize = PROMPT_ARGUMENT_COUNT;
const OUTPUT_VALUE_INDEX: usize = OUTPUT_OPTION_INDEX + 1;
const OUTPUT_OPTION: &str = "--output-root";
const SCHEMA_VERSION: u32 = 1;
#[cfg(test)]
const TEST_MAX_PROMPT_TOKENS_TEXT: &str = "32";
#[cfg(test)]
const TEST_MAX_PROMPT_TOKENS: usize = 32;
#[cfg(test)]
const TEST_MAX_NEW_TOKENS_TEXT: &str = "3";
#[cfg(test)]
const TEST_MAX_NEW_TOKENS: usize = 3;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rwkv-full-device-reference: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = std::env::args_os()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .map(require_utf8_argument)
        .collect::<Result<Vec<_>, _>>()?;
    let (request, output_root) = parse_arguments(&arguments).map_err(invalid_input)?;
    prepare_output_root(&output_root)?;

    let checkpoint = read_embedded_store_file(MODEL_PATH, "RWKV_LAYER_MODEL")?;
    let expected_model_blake3 = EXPECTED_MODEL_BLAKE3.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "RWKV_LAYER_MODEL_BLAKE3 was not embedded at build time",
        )
    })?;
    let vocabulary = read_embedded_store_file(VOCABULARY_PATH, "RWKV_TOKENIZER_VOCABULARY")?;
    let tokenizer_config =
        read_embedded_store_file(TOKENIZER_CONFIG_PATH, "RWKV_TOKENIZER_CONFIG")?;
    let added_tokens = read_embedded_store_file(ADDED_TOKENS_PATH, "RWKV_TOKENIZER_ADDED_TOKENS")?;
    let tokenizer_implementation = read_embedded_store_file(
        TOKENIZER_IMPLEMENTATION_PATH,
        "RWKV_TOKENIZER_IMPLEMENTATION",
    )?;
    let special_tokens_map =
        read_embedded_store_file(SPECIAL_TOKENS_MAP_PATH, "RWKV_SPECIAL_TOKENS_MAP")?;
    let model_config = read_embedded_store_file(MODEL_CONFIG_PATH, "RWKV_MODEL_CONFIG")?;
    let generation_config =
        read_embedded_store_file(GENERATION_CONFIG_PATH, "RWKV_GENERATION_CONFIG")?;
    let authority = TokenizerAuthorityInputs {
        vocabulary: &vocabulary,
        tokenizer_config: &tokenizer_config,
        added_tokens: &added_tokens,
        tokenizer_implementation: &tokenizer_implementation,
        special_tokens_map: &special_tokens_map,
        model_config: &model_config,
        generation_config: &generation_config,
    };
    let trace = run_full_device_reference(
        &checkpoint,
        expected_model_blake3,
        authority,
        request.clone(),
    )
    .map_err(invalid_data)?;

    let steps = trace
        .steps
        .iter()
        .enumerate()
        .map(|(index, step)| write_step(&output_root, &format!("reference-step-{index}"), step))
        .collect::<Result<Vec<_>, _>>()?;
    let reset_control = write_step(
        &output_root,
        "reference-control-reset",
        &trace.reset_control,
    )?;
    let transposed_control = write_step(
        &output_root,
        "reference-control-transposed",
        &trace.transposed_control,
    )?;
    let manifest = json!({
        "schema_version": SCHEMA_VERSION,
        "arithmetic_precision": "CPU FP32 from BF16 checkpoint values",
        "request": {
            "user_message": request.user_message,
            "max_prompt_tokens": request.max_prompt_tokens,
            "max_new_tokens": request.max_new_tokens,
        },
        "prompt_token_ids": trace.prompt_token_ids,
        "generated_token_ids": trace.generated_token_ids,
        "steps": steps,
        "controls": {
            "reset": reset_control,
            "transposed": transposed_control,
        },
    });
    write_json(&output_root.join("reference-manifest.json"), &manifest)?;
    println!("{}", output_root.join("reference-manifest.json").display());
    Ok(())
}

fn parse_arguments(arguments: &[String]) -> Result<(PromptRequest, PathBuf), String> {
    if arguments.len() != EXPECTED_ARGUMENT_COUNT {
        return Err(format!(
            "requires six bounded prompt arguments followed by {OUTPUT_OPTION} PATH"
        ));
    }
    if arguments[OUTPUT_OPTION_INDEX] != OUTPUT_OPTION {
        return Err(format!(
            "argument {OUTPUT_OPTION_INDEX} must be {OUTPUT_OPTION}"
        ));
    }
    let request = parse_prompt_arguments(&arguments[..PROMPT_ARGUMENT_COUNT])?;
    let output_root = PathBuf::from(&arguments[OUTPUT_VALUE_INDEX]);
    if !output_root.is_absolute() {
        return Err("output root must be absolute".to_owned());
    }
    if output_root.starts_with("/nix/store/") {
        return Err("output root must be outside the Nix store".to_owned());
    }
    Ok((request, output_root))
}

fn prepare_output_root(root: &Path) -> Result<(), io::Error> {
    let canonical = fs::canonicalize(root)?;
    if canonical != root {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "output root must already be canonical",
        ));
    }
    if !fs::metadata(&canonical)?.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "output root must be a directory",
        ));
    }
    if fs::read_dir(&canonical)?.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "output root must be empty",
        ));
    }
    Ok(())
}

fn write_step(root: &Path, name: &str, step: &FullDeviceReferenceStep) -> Result<Value, io::Error> {
    Ok(json!({
        "input_token_id": step.input_token_id,
        "selected_token_id": step.selected_token_id,
        "final_hidden": write_f32_artifact(root, &format!("{name}-final-hidden.f32"), &step.final_hidden)?,
        "logits": write_f32_artifact(root, &format!("{name}-logits.f32"), &step.logits)?,
        "complete_recurrent_state": write_f32_artifact(
            root,
            &format!("{name}-state.f32"),
            &step.complete_recurrent_state,
        )?,
    }))
}

fn write_f32_artifact(root: &Path, name: &str, values: &[f32]) -> Result<Value, io::Error> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "reference tensor must be non-empty and finite",
        ));
    }
    let mut bytes = Vec::with_capacity(values.len() * size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let path = root.join(name);
    let mut output = BufWriter::new(File::create(&path)?);
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(json!({
        "path": name,
        "element_count": values.len(),
        "byte_count": bytes.len(),
        "blake3": blake3::hash(&bytes).to_hex().to_string(),
    }))
}

fn write_json(path: &Path, value: &Value) -> Result<(), io::Error> {
    let mut output = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()
}

fn require_utf8_argument(argument: OsString) -> Result<String, io::Error> {
    argument.into_string().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "rwkv-full-device-reference arguments must be valid UTF-8",
        )
    })
}

fn read_embedded_store_file(
    configured_path: Option<&str>,
    variable_name: &str,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let configured_path = configured_path.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("{variable_name} was not embedded at build time"),
        )
    })?;
    let canonical = fs::canonicalize(configured_path)?;
    if !canonical.starts_with("/nix/store/") || !fs::metadata(&canonical)?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "embedded authority must be a Nix store file: {}",
                canonical.display()
            ),
        )
        .into());
    }
    fs::read(&canonical).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("failed to read {}: {error}", canonical.display()),
        )
        .into()
    })
}

fn invalid_input(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn invalid_data(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::{
        OUTPUT_VALUE_INDEX, TEST_MAX_NEW_TOKENS, TEST_MAX_NEW_TOKENS_TEXT, TEST_MAX_PROMPT_TOKENS,
        TEST_MAX_PROMPT_TOKENS_TEXT, parse_arguments,
    };

    fn valid_arguments() -> Vec<String> {
        [
            "--message",
            "Hello",
            "--max-prompt-tokens",
            TEST_MAX_PROMPT_TOKENS_TEXT,
            "--max-new-tokens",
            TEST_MAX_NEW_TOKENS_TEXT,
            "--output-root",
            "/tmp/reference",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    #[test]
    fn parses_complete_bounded_request() {
        let (request, root) =
            parse_arguments(&valid_arguments()).expect("valid request must parse");
        assert_eq!(request.user_message, "Hello");
        assert_eq!(request.max_prompt_tokens, TEST_MAX_PROMPT_TOKENS);
        assert_eq!(request.max_new_tokens, TEST_MAX_NEW_TOKENS);
        assert_eq!(root, std::path::Path::new("/tmp/reference"));
    }

    #[test]
    fn rejects_relative_or_incomplete_output_root() {
        let mut relative = valid_arguments();
        relative[OUTPUT_VALUE_INDEX] = "relative".to_owned();
        assert!(
            parse_arguments(&relative)
                .expect_err("relative root must fail")
                .contains("must be absolute")
        );
        let mut incomplete = valid_arguments();
        incomplete.pop();
        assert!(
            parse_arguments(&incomplete)
                .expect_err("incomplete request must fail")
                .contains("requires six")
        );
    }
}
