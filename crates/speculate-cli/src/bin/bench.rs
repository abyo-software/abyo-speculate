//! `abyo-speculate-bench` — measure tok/s for each supported method on a given model.
//!
//! Phase 1: scaffolding. Real measurement lands once `engine.generate()` runs a
//! model. The CLI surface is stable so external scripts can pin against it.

use abyo_speculate::{presets, Method, SpeculateEngine};
use anyhow::{Context, Result};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "abyo-speculate-bench",
    version,
    about = "Benchmark abyo-speculate against a baseline autoregressive run"
)]
struct Args {
    /// Preset name (see `--list-presets`) or a Hugging Face id for the target model.
    #[arg(long)]
    model: Option<String>,

    /// Show available presets and exit.
    #[arg(long)]
    list_presets: bool,

    /// SD method: autoregressive, vanilla, medusa, eagle2, eagle3.
    #[arg(long, default_value = "autoregressive")]
    method: String,

    /// Draft model id (only required for vanilla / eagle*).
    #[arg(long)]
    draft: Option<String>,

    /// Tokens to generate per prompt.
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,

    /// Number of warm-up runs (excluded from measurement).
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Number of measured runs.
    #[arg(long, default_value_t = 3)]
    runs: usize,
}

fn parse_method(s: &str) -> Result<Method> {
    Ok(match s.to_lowercase().as_str() {
        "autoregressive" | "ar" | "baseline" => Method::Autoregressive,
        "vanilla" | "vanilla-sd" | "leviathan" => Method::Vanilla,
        "medusa" => Method::Medusa,
        "eagle2" | "eagle-2" => Method::Eagle2,
        "eagle3" | "eagle-3" => Method::Eagle3,
        other => anyhow::bail!("unknown method: {other}"),
    })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    if args.list_presets {
        println!("Available presets:");
        for n in presets::known_names() {
            let p = presets::lookup(n).expect("known preset");
            println!(
                "  {:<20}  target={:<45}  method={}",
                n,
                p.target,
                p.method.name()
            );
        }
        return Ok(());
    }

    let model = args
        .model
        .as_deref()
        .context("--model is required (or use --list-presets)")?;
    let method = parse_method(&args.method)?;

    let mut builder = SpeculateEngine::builder()
        .target_model(model)
        .method(method);
    if let Some(d) = args.draft.as_deref() {
        builder = builder.draft_model(d);
    }
    let engine = builder.build()?;

    println!(
        "engine ready: method={} target={:?}",
        engine.config().method.name(),
        engine.config().target
    );
    println!(
        "(phase-1 stub; running {} warm-up + {} measured pseudo-runs at max_tokens={})",
        args.warmup, args.runs, args.max_tokens
    );

    let prompt = "The quick brown fox jumps over the lazy dog.";
    for i in 0..args.warmup {
        let _ = engine.generate(prompt, args.max_tokens)?;
        println!("warm-up {i}: ok");
    }
    for i in 0..args.runs {
        let out = engine.generate(prompt, args.max_tokens)?;
        println!("run {i}: {out}");
    }
    Ok(())
}
