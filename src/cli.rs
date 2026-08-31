use clap::Parser;

/// tinyminigo — a minimal inference engine for MiniCPM5 (Llama-style decoder).
#[derive(Parser, Debug)]
#[command(name = "tinyminigo", version, about)]
pub struct Args {
    /// Path to the model weights (safetensors file)
    #[arg(short = 'm', long = "model", value_name = "FILE")]
    pub model: String,

    /// Path to the Jinja chat template file
    #[arg(long = "template", value_name = "FILE")]
    pub template: String,

    /// Path to the tokenizer (defaults to tokenizer.json next to the model)
    #[arg(long = "tokenizer", value_name = "FILE")]
    pub tokenizer: Option<String>,

    /// Maximum number of new tokens to generate
    #[arg(long = "max-tokens", value_name = "N", default_value_t = 256)]
    pub max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long = "temperature", value_name = "T", default_value_t = 0.0)]
    pub temperature: f32,

    /// Top-k sampling (0 = disabled)
    #[arg(long = "top-k", value_name = "K", default_value_t = 0)]
    pub top_k: usize,

    /// Top-p (nucleus) sampling threshold (1.0 = disabled)
    #[arg(long = "top-p", value_name = "P", default_value_t = 1.0)]
    pub top_p: f32,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long = "repetition-penalty", value_name = "R", default_value_t = 1.0)]
    pub repetition_penalty: f32,

    /// Random seed for sampling
    #[arg(long = "seed", value_name = "SEED", default_value_t = 42)]
    pub seed: u64,

    /// The user prompt
    pub prompt: String,

    /// Debug: dump first-step logits to a JSON file (for cross-checking)
    #[arg(long = "dump-first-logits", value_name = "FILE", hide = true)]
    pub dump_first_logits: Option<String>,
}
