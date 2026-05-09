//! `abyo-speculate-demo` — minimal "hello world" runner that prints the
//! resolved engine config and emits a single completion.

use abyo_speculate::{Method, SpeculateEngine};
use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "abyo-speculate-demo", version)]
struct Args {
    /// Hugging Face id or local path for the target model.
    #[arg(long, default_value = "meta-llama/Llama-3.1-8B-Instruct")]
    model: String,

    /// Prompt to feed.
    #[arg(long, default_value = "Hello, abyo-speculate!")]
    prompt: String,

    /// Tokens to generate.
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let engine = SpeculateEngine::builder()
        .target_model(&args.model)
        .method(Method::Autoregressive)
        .build()?;
    let out = engine.generate(&args.prompt, args.max_tokens)?;
    println!("{out}");
    Ok(())
}
