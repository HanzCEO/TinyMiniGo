mod cli;
mod config;
mod model;
mod safetensors_loader;
mod template;
mod tensor;
mod tmb;
mod tokenizer;

use anyhow::Result;
use clap::Parser;
use std::io::Write;
use std::path::Path;

use cli::Args;
use model::{GenParams, Model};
use tokenizer::Tokenizer;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    if let Some(dst) = &args.repack {
        let src_path = Path::new(&args.model);
        let cfg_path = config::ModelConfig::config_path_for_model(src_path)
            .ok_or_else(|| anyhow::anyhow!("config.json not found next to {}", src_path.display()))?;
        let cfg = config::ModelConfig::load(&cfg_path)?;
        tmb::write_tmb(src_path, Path::new(dst), &cfg)?;
        return Ok(());
    }

    if let Ok(t) = std::env::var("TMG_THREADS") {
        if let Ok(n) = t.parse::<usize>() {
            tensor::set_num_threads(n);
        }
    }

    let model_path = Path::new(&args.model);
    if !model_path.exists() {
        anyhow::bail!("model file not found: {}", model_path.display());
    }

    eprintln!("loading model {} ...", model_path.display());
    let load_start = std::time::Instant::now();
    let mut m = Model::load(model_path)?;
    let load_secs = load_start.elapsed().as_secs_f32();
    eprintln!(
        "loaded in {:.2}s: layers={} hidden={} heads={} kv_heads={} head_dim={} vocab={}",
        load_secs,
        m.w.config.num_hidden_layers,
        m.w.config.hidden_size,
        m.w.config.num_attention_heads,
        m.w.config.num_key_value_heads,
        m.w.config.head_dim,
        m.w.config.vocab_size
    );

    let tok_path = tokenizer::resolve_tokenizer_path(args.tokenizer.as_deref(), model_path)?;
    if !tok_path.exists() {
        anyhow::bail!(
            "tokenizer not found at {} (pass --tokenizer to specify)",
            tok_path.display()
        );
    }
    let tok = Tokenizer::from_file(&tok_path)?;

    // BOS token from tokenizer config if available; default "<s>".
    let bos = "<s>";
    let prompt_rendered = template::render_chat_template(Path::new(&args.template), &args.prompt, bos)?;
    eprintln!("--- rendered prompt ---\n{prompt_rendered}\n-----------------------");

    let ids = tok.encode(&prompt_rendered, false)?;
    eprintln!("prompt tokens: {} {:?}", ids.len(), ids);

    let params = GenParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    let t_start = std::time::Instant::now();
    let result = model::generate(&mut m, &ids, &params, |t| {
        // stream detokenization of each new token
        if let Ok(s) = tok.decode(&[t], true) {
            print!("{s}");
            let _ = out.flush();
        }
    })?;
    if let Some(path) = &args.dump_first_logits {
        let top: Vec<(usize, f32)> = {
            let mut idx: Vec<usize> = (0..result.logits_first_step.len()).collect();
            idx.sort_by(|&a, &b| result.logits_first_step[b]
                .partial_cmp(&result.logits_first_step[a])
                .unwrap());
            idx.truncate(5);
            idx.iter().map(|&i| (i, result.logits_first_step[i])).collect()
        };
        let dump = serde_json::json!({
            "top5_ids": top.iter().map(|(i, _)| i).collect::<Vec<_>>(),
            "top5_vals": top.iter().map(|(_, v)| v).collect::<Vec<_>>(),
            "first8": &result.logits_first_step[..8],
        });
        std::fs::write(path, serde_json::to_string_pretty(&dump)?)?;
    }

    println!();
    let elapsed = t_start.elapsed().as_secs_f32();
    let n_gen = result.tokens.len();
    let decode_tokens = n_gen.saturating_sub(1); // last gen step is part of prefill+1 forward
    eprintln!(
        "\n[{} tokens generated in {:.2}s | prefill: {} tok in {:.2}s ({:.1} tok/s) | decode: ~{} tok in {:.2}s ({:.2} tok/s)]",
        n_gen,
        elapsed,
        ids.len(),
        result.prefill_secs,
        ids.len() as f32 / result.prefill_secs.max(1e-9),
        decode_tokens,
        elapsed - result.prefill_secs,
        decode_tokens as f32 / (elapsed - result.prefill_secs).max(1e-9),
    );
    Ok(())
}
