use anyhow::{Context, Result};

use std::path::Path;

/// Render a chat template for generation from a user prompt (no system, no history).
pub fn render_chat_template(template_path: &Path, user_prompt: &str, bos_token: &str) -> Result<String> {
    let src = std::fs::read_to_string(template_path)
        .with_context(|| format!("reading chat template {}", template_path.display()))?;
    let mut env = minijinja::Environment::new();
    // HF chat templates call Python string methods (startswith, lstrip, ...).
    // minijinja-contrib's pycompat module emulates them.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", &src)
        .with_context(|| format!("parsing chat template {}", template_path.display()))?;
    let tmpl = env.get_template("chat")?;

    let messages = vec![minijinja::Value::from_serialize(serde_json::json!({
        "role": "user",
        "content": user_prompt,
    }))];
    let ctx = minijinja::Value::from_serialize(serde_json::json!({
        "messages": messages,
        "add_generation_prompt": true,
        "bos_token": bos_token,
        "eos_token": "</s>",
        "pad_token": "</s>",
    }));
    Ok(tmpl.render(ctx)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_simple_template() {
        let dir = std::env::temp_dir().join("tinyminigo-test-template");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.jinja");
        std::fs::write(&path, "{{ bos_token }}{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}").unwrap();
        let out = render_chat_template(&path, "Hi", "<s>").unwrap();
        assert_eq!(out, "<s><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }
}
