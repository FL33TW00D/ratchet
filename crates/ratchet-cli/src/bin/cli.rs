use clap::{value_parser, Arg, ArgMatches, Command};
use hf_hub::api::sync::Api;
use ratchet::{Device, DeviceRequest};
use ratchet_loader::ggml::GGMLCompatible;
use ratchet_models::model::Whisper;
use ratchet_models::options::DecodingOptionsBuilder;
use ratchet_models::registry::{AvailableModels, Quantization, Whisper as RegistryWhisper};
use ratchet_models::transcribe::transcribe;
use std::path::Path;
use std::process::Command as TermCommand;

fn ffmpeg_preproc<P: AsRef<Path>>(path: P) -> Vec<f32> {
    let path = path.as_ref();
    let output = TermCommand::new("ffmpeg")
        .args([
            "-nostdin",
            "-threads",
            "0",
            "-i",
            path.to_str().unwrap(),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-loglevel",
            "error",
            "-ar",
            "16000",
            "-",
        ])
        .output()
        .expect("Failed to execute ffmpeg command");

    if !output.status.success() {
        panic!(
            "ffmpeg command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let audio_data = output.stdout;
    let mut samples = Vec::new();

    for chunk in audio_data.chunks(2) {
        let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
        samples.push(sample);
    }

    samples
}

pub fn start_logger() {
    let logger = fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level_for("tokenizers", log::LevelFilter::Off)
        .level(log::LevelFilter::Warn)
        .apply();
    match logger {
        Ok(_) => log::info!("Logging initialized."),
        Err(error) => eprintln!("Error initializing logging: {:?}", error),
    }
}

fn handle_whisper(matches: ArgMatches, api: Api) {
    if let Some(matches) = matches.subcommand_matches("whisper") {
        let mut whisper = if let Some(variant) = matches.get_one::<RegistryWhisper>("variant") {
            let model = AvailableModels::Whisper(variant.clone());
            let repo = api.model(model.repo_id());
            let model_path = repo.get(&model.model_id(Quantization::Q8)).unwrap();

            let mut reader = std::io::BufReader::new(std::fs::File::open(model_path).unwrap());
            let gg_disk = Whisper::load_ggml(&mut reader).unwrap();

            let device = Device::request_device(DeviceRequest::GPU).unwrap();
            Whisper::load(&gg_disk, &mut reader, device).unwrap()
        } else {
            panic!("Model not found");
        };

        if let Some(input) = matches.get_one::<String>("input") {
            let options = DecodingOptionsBuilder::new().build();
            let samples = ffmpeg_preproc(input);
            let transcript =
                transcribe(&mut whisper, samples, options, Some(|s| println!("{}", s))).unwrap();
            log::info!("Processing time: {:?}", transcript.processing_time);
        } else {
            panic!("Input file not found");
        };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let matches = Command::new("ratchet")
        .about("LLM & VLM CLI")
        .version("0.1.0")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("whisper")
                .long_about(
                    "Cross-platform, GPU accelerated implementation of OpenAI's Whisper Model.",
                )
                .arg(
                    Arg::new("variant")
                        .short('v')
                        .long("variant")
                        .default_value("small")
                        .help("Whisper model variant to use.")
                        .value_parser(value_parser!(RegistryWhisper)),
                )
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .required(true)
                        .help("Path to the input file"),
                ),
        )
        .get_matches();

    let api = Api::new().unwrap();
    handle_whisper(matches, api);

    Ok(())
}
