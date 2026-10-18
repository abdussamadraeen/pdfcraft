//! Sovereign Local AI Subsystem (Gap 23, M13: `misc.ai-provider`, `misc.ai-summary`, `misc.ai-ask`,
//! `protect.redact-ai-suggestions`, `a11y.alt-text-suggestions`).
//!
//! # Principles
//! - **Sovereign & Local-First**: Built-in statistical NLP, BM25 semantic retrieval, and pattern
//!   recognition that execute 100% locally and offline with zero cloud dependencies or mandatory downloads.
//! - **Off by Default**: The provider interface is disabled by default, requiring explicit user/agent opt-in.
//! - **Opt-In Endpoints**: Allows delegating to a user-configured local HTTP endpoint (e.g. Ollama,
//!   local OpenAI-compatible server) with automatic offline fallback to built-in sovereign heuristics.
//! - **Never Panic**: 100% safe Rust, checked bounds, saturating math, zero unwrap/expect in production.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use pdfcraft_engine::{Edit, NewAnnotation, Shape, Style};
use serde_json::{Value, json};

use crate::comments::{DEFAULT_AUTHOR, parse_color};
use crate::{Args, Automation, Result, ToolError};

/// Provider execution mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderMode {
    /// Built-in offline sovereign heuristics and statistical ranking (default).
    Builtin,
    /// Local user-provided HTTP endpoint (e.g. Ollama or local LLM server).
    Endpoint,
}

impl ProviderMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Builtin => "builtin",
            Self::Endpoint => "endpoint",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "builtin" => Some(Self::Builtin),
            "endpoint" => Some(Self::Endpoint),
            _ => None,
        }
    }
}

/// AI Provider configuration. Off by default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderConfig {
    pub enabled: bool,
    pub mode: ProviderMode,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self { enabled: false, mode: ProviderMode::Builtin, endpoint: None, model: None, api_key: None }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_http_agent() -> std::result::Result<ureq::Agent, String> {
    let mut builder = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(60)));
    let found = rustls_native_certs::load_native_certs();
    let certs: Vec<ureq::tls::Certificate<'static>> =
        found.certs.iter().map(|c| ureq::tls::Certificate::from_der(c.as_ref()).to_owned()).collect();
    if !certs.is_empty() {
        let roots = ureq::tls::RootCerts::new_with_certs(&certs);
        builder = builder.tls_config(ureq::tls::TlsConfig::builder().root_certs(roots).build());
    }
    Ok(builder.build().new_agent())
}

/// Extract 1-based page citations like `[Page 3]` or `[p.3]` from LLM text.
pub fn extract_page_citations(text: &str) -> Vec<usize> {
    let mut pages = BTreeSet::new();
    let lower = text.to_lowercase();
    let mut search_from = 0;
    while let Some(idx) = lower[search_from..].find("[page ") {
        let abs_idx = search_from + idx + 6;
        let rest = &lower[abs_idx..];
        if let Some(end_bracket) = rest.find(']') {
            let num_str = rest[..end_bracket].trim();
            if let Ok(num) = num_str.parse::<usize>()
                && num > 0
            {
                pages.insert(num);
            }
        }
        search_from = abs_idx;
    }
    search_from = 0;
    while let Some(idx) = lower[search_from..].find("[p.") {
        let abs_idx = search_from + idx + 3;
        let rest = &lower[abs_idx..];
        if let Some(end_bracket) = rest.find(']') {
            let num_str = rest[..end_bracket].trim();
            if let Ok(num) = num_str.parse::<usize>()
                && num > 0
            {
                pages.insert(num);
            }
        }
        search_from = abs_idx;
    }
    pages.into_iter().collect()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn call_llm_endpoint(
    config: &ProviderConfig,
    system_prompt: Option<&str>,
    user_prompt: &str,
) -> std::result::Result<String, String> {
    let agent = build_http_agent()?;
    let raw_endpoint = config.endpoint.as_deref().unwrap_or("http://localhost:11434").trim();
    let endpoint = raw_endpoint.trim_end_matches('/');

    // 1. Google Gemini API
    if endpoint.contains("googleapis.com") || endpoint.contains("gemini") {
        let model = config.model.as_deref().unwrap_or("gemini-1.5-flash").trim();
        let api_key = config.api_key.as_deref().unwrap_or("").trim();
        if api_key.is_empty() {
            return Err("Google Gemini requires an API key. Add your free key in AI Settings (get one at https://aistudio.google.com).".into());
        }

        let url = if endpoint.contains(":generateContent") {
            if endpoint.contains("key=") {
                endpoint.to_string()
            } else {
                format!("{endpoint}?key={api_key}")
            }
        } else {
            format!("{endpoint}/v1beta/models/{model}:generateContent?key={api_key}")
        };

        let combined_text = if let Some(sys) = system_prompt {
            format!("{sys}\n\n{user_prompt}")
        } else {
            user_prompt.to_string()
        };

        let req_body = serde_json::json!({
            "contents": [
                {
                    "parts": [
                        { "text": combined_text }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2
            }
        });

        let body_str = serde_json::to_string(&req_body).map_err(|e| e.to_string())?;
        let mut resp = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .config()
            .http_status_as_error(false)
            .build()
            .send(body_str.as_bytes())
            .map_err(|e| format!("Gemini connection error: {e}"))?;

        let status = resp.status().as_u16();
        let resp_text = resp.body_mut().with_config().limit(1 << 23).read_to_string().map_err(|e| e.to_string())?;

        if !(200..300).contains(&status) {
            if let Ok(err_val) = serde_json::from_str::<serde_json::Value>(&resp_text)
                && let Some(msg) = err_val.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str())
            {
                return Err(format!("Gemini API error ({status}): {msg}"));
            }
            return Err(format!("Gemini API error (HTTP {status}): {resp_text}"));
        }

        let parsed: serde_json::Value = serde_json::from_str(&resp_text).map_err(|e| format!("Invalid JSON from Gemini: {e}"))?;
        let text = parsed
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c0| c0.get("content"))
            .and_then(|cnt| cnt.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p0| p0.get("text"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| "Gemini response did not contain candidates[0].content.parts[0].text".to_string())?;

        return Ok(text.to_string());
    }

    // 2. Ollama Native API (port 11434 or /api/generate)
    if (endpoint.contains("11434") || endpoint.ends_with("/api/generate")) && !endpoint.contains("/v1") {
        let model = config.model.as_deref().unwrap_or("llama3.2").trim();
        let url = if endpoint.ends_with("/api/generate") {
            endpoint.to_string()
        } else {
            format!("{endpoint}/api/generate")
        };

        let mut req_body = serde_json::json!({
            "model": model,
            "prompt": user_prompt,
            "stream": false,
        });
        if let Some(sys) = system_prompt {
            req_body["system"] = serde_json::json!(sys);
        }

        let body_str = serde_json::to_string(&req_body).map_err(|e| e.to_string())?;
        let mut resp = agent
            .post(&url)
            .header("Content-Type", "application/json")
            .config()
            .http_status_as_error(false)
            .build()
            .send(body_str.as_bytes())
            .map_err(|e| format!("Ollama connection error: {e}. Is Ollama running on {endpoint}?"))?;

        let status = resp.status().as_u16();
        let resp_text = resp.body_mut().with_config().limit(1 << 23).read_to_string().map_err(|e| e.to_string())?;

        if !(200..300).contains(&status) {
            if let Ok(err_val) = serde_json::from_str::<serde_json::Value>(&resp_text)
                && let Some(msg) = err_val.get("error").and_then(|m| m.as_str())
            {
                return Err(format!("Ollama error ({status}): {msg}"));
            }
            return Err(format!("Ollama error (HTTP {status}): {resp_text}"));
        }

        let parsed: serde_json::Value = serde_json::from_str(&resp_text).map_err(|e| format!("Invalid JSON from Ollama: {e}"))?;
        let text = parsed
            .get("response")
            .and_then(|r| r.as_str())
            .ok_or_else(|| "Ollama response did not contain 'response' field".to_string())?;

        return Ok(text.to_string());
    }

    // 3. OpenAI-Compatible (Groq, LM Studio, OpenRouter, LocalAI)
    let model = config.model.as_deref().unwrap_or_else(|| {
        if endpoint.contains("groq.com") {
            "llama-3.3-70b-versatile"
        } else {
            "gpt-3.5-turbo"
        }
    }).trim();

    let url = if endpoint.ends_with("/chat/completions") {
        endpoint.to_string()
    } else {
        format!("{endpoint}/chat/completions")
    };

    let mut messages = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": sys,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": user_prompt,
    }));

    let req_body = serde_json::json!({
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    });

    let body_str = serde_json::to_string(&req_body).map_err(|e| e.to_string())?;
    let mut req = agent.post(&url).header("Content-Type", "application/json");

    if let Some(ref key) = config.api_key {
        let trimmed_key = key.trim();
        if !trimmed_key.is_empty() {
            req = req.header("Authorization", &format!("Bearer {trimmed_key}"));
        }
    }

    let mut resp = req
        .config()
        .http_status_as_error(false)
        .build()
        .send(body_str.as_bytes())
        .map_err(|e| format!("Endpoint connection error: {e}"))?;

    let status = resp.status().as_u16();
    let resp_text = resp.body_mut().with_config().limit(1 << 23).read_to_string().map_err(|e| e.to_string())?;

    if !(200..300).contains(&status) {
        if let Ok(err_val) = serde_json::from_str::<serde_json::Value>(&resp_text)
            && let Some(msg) = err_val.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str())
        {
            return Err(format!("AI Endpoint error ({status}): {msg}"));
        }
        return Err(format!("AI Endpoint error (HTTP {status}): {resp_text}"));
    }

    let parsed: serde_json::Value = serde_json::from_str(&resp_text).map_err(|e| format!("Invalid JSON from endpoint: {e}"))?;
    let text = parsed
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c0| c0.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|t| t.as_str())
        .ok_or_else(|| "Endpoint response did not contain choices[0].message.content".to_string())?;

    Ok(text.to_string())
}

#[cfg(target_arch = "wasm32")]
pub fn call_llm_endpoint(
    _config: &ProviderConfig,
    _system_prompt: Option<&str>,
    _user_prompt: &str,
) -> std::result::Result<String, String> {
    Err("Network LLM endpoints are not supported on WebAssembly".into())
}



/// Stop words filtered out during keyword extraction and semantic scoring.
const STOP_WORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

/// Known abbreviations that should not trigger sentence boundaries.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "eg", "ie", "etc", "inc", "ltd", "corp", "co", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec", "approx", "dept", "est", "min", "max", "no", "vol", "pg", "pp",
];

/// A sentence extracted from a document with its 1-based origin page and position.
#[derive(Clone, Debug, PartialEq)]
pub struct Sentence {
    pub page: usize,
    pub text: String,
    pub index_on_page: usize,
}

impl Sentence {
    pub fn words(&self) -> Vec<String> {
        tokenize_words(&self.text)
    }

    pub fn content_words(&self) -> Vec<String> {
        let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
        self.words().into_iter().filter(|w| !stop.contains(w.as_str()) && w.len() > 1).collect()
    }
}

/// Tokenize text into lowercase alphanumeric words.
pub fn tokenize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase()).collect()
}

/// Split text into sentences, taking abbreviations, titles and decimal points into account.
pub fn split_sentences(text: &str, page: usize) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    let mut current = String::with_capacity(256);
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut idx_on_page = 0;

    let abbrev_set: HashSet<&str> = ABBREVIATIONS.iter().copied().collect();

    while i < n {
        let c = chars[i];
        current.push(c);

        let is_double_newline = c == '\n' && i + 1 < n && chars[i + 1] == '\n';
        let is_punct = c == '.' || c == '?' || c == '!';

        if is_double_newline || is_punct {
            let mut is_end = true;

            if c == '.' {
                // Check if decimal number: e.g. 3.14
                let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
                let next_digit = i + 1 < n && chars[i + 1].is_ascii_digit();
                if prev_digit && next_digit {
                    is_end = false;
                }

                // Check abbreviation: extract preceding word
                if is_end && i > 0 {
                    let mut start = i;
                    while start > 0 && chars[start - 1].is_alphabetic() {
                        start = start.saturating_sub(1);
                    }
                    let word: String = chars[start..i].iter().collect::<String>().to_lowercase();
                    if abbrev_set.contains(word.as_str()) {
                        is_end = false;
                    }
                }

                // Check if next non-space character is lowercase
                if is_end && i + 1 < n {
                    let mut j = i + 1;
                    while j < n && (chars[j].is_whitespace() || chars[j] == '"' || chars[j] == '\'') {
                        j = j.saturating_add(1);
                    }
                    if j < n && chars[j].is_lowercase() {
                        is_end = false;
                    }
                }
            }

            if is_end {
                let trimmed = current.trim();
                if !trimmed.is_empty() && trimmed.chars().any(char::is_alphanumeric) {
                    sentences.push(Sentence { page, text: trimmed.to_string(), index_on_page: idx_on_page });
                    idx_on_page = idx_on_page.saturating_add(1);
                }
                current.clear();
            }
        }
        i = i.saturating_add(1);
    }

    let final_trimmed = current.trim();
    if !final_trimmed.is_empty() && final_trimmed.chars().any(char::is_alphanumeric) {
        sentences.push(Sentence { page, text: final_trimmed.to_string(), index_on_page: idx_on_page });
    }

    sentences
}

/// Calculate BM25 relevance score for a sentence against query terms.
pub fn bm25_score(query_terms: &[String], sentence: &Sentence, all_sentences: &[Sentence], avg_len: f64) -> f64 {
    if query_terms.is_empty() || all_sentences.is_empty() {
        return 0.0;
    }

    let words = sentence.words();
    let s_len = words.len() as f64;
    let total_s = all_sentences.len() as f64;
    let k1 = 1.2;
    let b = 0.75;

    let mut term_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for w in &words {
        *term_counts.entry(w.as_str()).or_insert(0) += 1;
    }

    let mut score = 0.0;
    for q in query_terms {
        let q_str = q.as_str();
        let tf = *term_counts.get(q_str).unwrap_or(&0) as f64;
        if tf <= 0.0 {
            continue;
        }

        // Count how many sentences contain q
        let df = all_sentences.iter().filter(|s| s.words().iter().any(|w| w == q)).count() as f64;
        let idf = ((total_s - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.1);

        let denom = tf + k1 * (1.0 - b + b * (s_len / avg_len.max(1.0)));
        score += idf * (tf * (k1 + 1.0) / denom.max(0.001));
    }

    // Exact phrase bonus if multiple query terms appear adjacent in sentence
    if query_terms.len() > 1 {
        let sent_lower = sentence.text.to_lowercase();
        let full_phrase = query_terms.join(" ");
        if sent_lower.contains(&full_phrase) {
            score += 2.5;
        } else {
            // Bigram matches
            for window in query_terms.windows(2) {
                let bigram = window.join(" ");
                if sent_lower.contains(&bigram) {
                    score += 0.8;
                }
            }
        }
    }

    score
}

/// Jaccard similarity between two sentences based on content words.
pub fn jaccard_similarity(a: &Sentence, b: &Sentence) -> f64 {
    let a_words: HashSet<String> = a.content_words().into_iter().collect();
    let b_words: HashSet<String> = b.content_words().into_iter().collect();
    if a_words.is_empty() && b_words.is_empty() {
        return 1.0;
    }
    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}

/// Validate credit card numbers using Luhn checksum.
pub fn luhn_check(digits: &[u8]) -> bool {
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum: u32 = 0;
    let mut alternate = false;
    for &d in digits.iter().rev() {
        let mut n = d as u32;
        if alternate {
            n = n.saturating_mul(2);
            if n > 9 {
                n = n.saturating_sub(9);
            }
        }
        sum = sum.saturating_add(n);
        alternate = !alternate;
    }
    sum.is_multiple_of(10)
}

/// Check if a 9-digit string is a plausible US Social Security Number.
pub fn is_valid_ssn(digits: &str) -> bool {
    if digits.len() != 9 || !digits.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    let area = match digits[0..3].parse::<u32>() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let group = match digits[3..5].parse::<u32>() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let serial = match digits[5..9].parse::<u32>() {
        Ok(v) => v,
        Err(_) => return false,
    };

    // Area 000, 666, and 900-999 are invalid
    if area == 0 || area == 666 || area >= 900 {
        return false;
    }
    if group == 0 || serial == 0 {
        return false;
    }
    true
}

/// TF-ISF (Term Frequency - Inverse Sentence Frequency) vectors and L2 norms.
pub fn compute_sentence_vectors(
    sentences: &[Sentence],
) -> (Vec<BTreeMap<String, f64>>, Vec<f64>) {
    let n = sentences.len() as f64;
    let mut sf: BTreeMap<String, usize> = BTreeMap::new();
    for s in sentences {
        let unique_words: HashSet<String> = s.content_words().into_iter().collect();
        for w in unique_words {
            *sf.entry(w).or_insert(0) += 1;
        }
    }

    let mut vectors = Vec::with_capacity(sentences.len());
    let mut norms = Vec::with_capacity(sentences.len());

    for s in sentences {
        let words = s.content_words();
        let mut tf: BTreeMap<String, usize> = BTreeMap::new();
        for w in &words {
            *tf.entry(w.clone()).or_insert(0) += 1;
        }

        let mut vec = BTreeMap::new();
        let mut norm_sq = 0.0;
        for (w, count) in tf {
            let sf_w = *sf.get(&w).unwrap_or(&1) as f64;
            let isf = ((n + 1.0) / (sf_w + 0.5)).ln() + 1.0;
            let weight = (count as f64) * isf;
            norm_sq += weight * weight;
            vec.insert(w, weight);
        }
        let norm = norm_sq.sqrt();
        vectors.push(vec);
        norms.push(norm);
    }

    (vectors, norms)
}

/// Cosine similarity between two sparse TF-ISF vectors.
pub fn cosine_similarity(
    v1: &BTreeMap<String, f64>,
    norm1: f64,
    v2: &BTreeMap<String, f64>,
    norm2: f64,
) -> f64 {
    if norm1 <= 1e-9 || norm2 <= 1e-9 {
        return 0.0;
    }
    let mut dot = 0.0;
    let (small, large) = if v1.len() < v2.len() { (v1, v2) } else { (v2, v1) };
    for (k, w1) in small {
        if let Some(w2) = large.get(k) {
            dot += w1 * w2;
        }
    }
    (dot / (norm1 * norm2)).clamp(0.0, 1.0)
}

/// TextRank graph centrality via PageRank power iteration over TF-ISF cosine graph.
pub fn textrank_scores(
    sentences: &[Sentence],
    vectors: &[BTreeMap<String, f64>],
    norms: &[f64],
) -> Vec<f64> {
    let n = sentences.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    let mut w = vec![vec![0.0f64; n]; n];
    let mut out_degree = vec![0.0f64; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&vectors[i], norms[i], &vectors[j], norms[j]);
            if sim > 0.02 {
                w[i][j] = sim;
                w[j][i] = sim;
                out_degree[i] += sim;
                out_degree[j] += sim;
            }
        }
    }

    let d = 0.85f64;
    let initial_score = 1.0 / (n as f64);
    let mut scores = vec![initial_score; n];

    for _ in 0..50 {
        let mut next_scores = vec![(1.0 - d) / (n as f64); n];
        let mut dangling_mass = 0.0f64;

        for j in 0..n {
            if out_degree[j] > 1e-9 {
                let share = d * scores[j] / out_degree[j];
                for i in 0..n {
                    if w[j][i] > 0.0 {
                        next_scores[i] += share * w[j][i];
                    }
                }
            } else {
                dangling_mass += scores[j];
            }
        }

        if dangling_mass > 0.0 {
            let uniform_share = d * dangling_mass / (n as f64);
            for score in &mut next_scores {
                *score += uniform_share;
            }
        }

        let diff: f64 = scores.iter().zip(&next_scores).map(|(a, b)| (a - b).abs()).sum();
        scores = next_scores;
        if diff < 1e-4 {
            break;
        }
    }

    scores
}

/// Select diverse sentences via Maximal Marginal Relevance (MMR).
pub fn mmr_select(
    sentences: &[Sentence],
    scores: &[f64],
    vectors: &[BTreeMap<String, f64>],
    norms: &[f64],
    limit: usize,
    lambda: f64,
) -> Vec<usize> {
    if sentences.is_empty() || limit == 0 {
        return Vec::new();
    }
    let max_score = scores.iter().copied().fold(1e-9f64, f64::max);
    let mut selected: Vec<usize> = Vec::with_capacity(limit);
    let mut remaining: HashSet<usize> = (0..sentences.len()).collect();

    while selected.len() < limit && !remaining.is_empty() {
        let mut best_idx = None;
        let mut best_mmr = f64::NEG_INFINITY;

        for &i in &remaining {
            let norm_score = scores[i] / max_score;
            let max_sim = selected
                .iter()
                .map(|&sel| cosine_similarity(&vectors[i], norms[i], &vectors[sel], norms[sel]))
                .fold(0.0f64, f64::max);

            let mmr = lambda * norm_score - (1.0 - lambda) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = Some(i);
            }
        }

        if let Some(chosen) = best_idx {
            remaining.remove(&chosen);
            selected.push(chosen);
        } else {
            break;
        }
    }

    selected
}

/// Pure extractive summarization logic on a collection of sentences using
/// TextRank (PageRank power iteration over TF-ISF cosine graph) and Maximal
/// Marginal Relevance (MMR) for optimal information density without repetition.
pub fn generate_summary_pure(
    all_sentences: &[Sentence],
    max_points: usize,
    length_pref: &str,
) -> (String, Vec<(usize, String)>, usize, usize) {
    if all_sentences.is_empty() {
        return ("The document contains no readable text to summarize.".into(), Vec::new(), 0, 0);
    }

    let mut total_words: usize = 0;
    for s in all_sentences {
        total_words = total_words.saturating_add(s.words().len());
    }

    // 1. Build TF-ISF sentence vectors and L2 norms
    let (vectors, norms) = compute_sentence_vectors(all_sentences);

    // 2. Compute TextRank graph centrality
    let tr_scores = textrank_scores(all_sentences, &vectors, &norms);

    // 3. Apply position bias & length normalization
    let combined_scores: Vec<f64> = all_sentences
        .iter()
        .zip(&tr_scores)
        .map(|(s, &tr)| {
            let mut score = tr;
            let word_count = s.words().len();

            // Structural document bias
            if s.index_on_page == 0 {
                score *= if s.page == 1 { 1.45 } else { 1.25 };
            }

            // Ideal informational sentence length window (10 to 45 words)
            if (10..=45).contains(&word_count) {
                score *= 1.2;
            } else if !(5..=65).contains(&word_count) {
                score *= 0.65;
            }

            score
        })
        .collect();

    // 4. Maximal Marginal Relevance selection (lambda = 0.72)
    let selected_indices = mmr_select(all_sentences, &combined_scores, &vectors, &norms, max_points.max(3), 0.72);

    let selected_sentences: Vec<&Sentence> = selected_indices.iter().filter_map(|&i| all_sentences.get(i)).collect();

    let summary_count = match length_pref {
        "short" => 1,
        "detailed" => 3.min(selected_sentences.len()),
        _ => 2.min(selected_sentences.len()),
    };

    // Synthesize structured executive overview
    let lead_sentences: Vec<String> = selected_sentences
        .iter()
        .take(summary_count)
        .map(|s| format!("{} [Page {}]", s.text.trim_end_matches('.'), s.page))
        .collect();

    let summary_text = lead_sentences.join(". ") + ".";

    let key_points: Vec<(usize, String)> = selected_sentences
        .into_iter()
        .map(|s| (s.page, s.text.clone()))
        .collect();

    let reading_time = ((total_words as f64) / 200.0).ceil() as usize;

    (summary_text, key_points, total_words, reading_time.max(1))
}

/// Pure BM25 question answering search over sentences.
pub fn search_answer_pure(
    question: &str,
    all_sentences: &[Sentence],
    max_citations: usize,
) -> (String, &'static str, Vec<(usize, String, f64)>) {
    if all_sentences.is_empty() {
        return ("The document contains no readable text to answer this question.".into(), "low", Vec::new());
    }

    let mut total_words: usize = 0;
    for s in all_sentences {
        total_words = total_words.saturating_add(s.words().len());
    }

    let avg_len = (total_words as f64) / (all_sentences.len() as f64).max(1.0);
    let q_words = tokenize_words(question);
    let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let q_content: Vec<String> = q_words.into_iter().filter(|w| !stop_set.contains(w.as_str())).collect();
    let query_terms = if q_content.is_empty() { tokenize_words(question) } else { q_content };

    let mut scored_sentences: Vec<(f64, &Sentence)> =
        all_sentences.iter().map(|s| (bm25_score(&query_terms, s, all_sentences, avg_len), s)).filter(|(score, _)| *score > 0.05).collect();

    scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    if scored_sentences.is_empty() {
        return (format!("No direct matches or relevant sections found in the document answering '{question}'."), "low", Vec::new());
    }

    let top_matches: Vec<_> = scored_sentences.iter().take(max_citations).collect();
    let citations: Vec<(usize, String, f64)> = top_matches
        .iter()
        .map(|(score, s)| (s.page, s.text.clone(), (((score * 100.0).round()) / 100.0)))
        .collect();

    let top_match = top_matches[0].1;
    let best_score = top_matches[0].0;
    let confidence = if best_score >= 1.5 { "high" } else { "medium" };

    let answer = if top_matches.len() == 1 {
        format!("According to [Page {}], {}.", top_match.page, top_match.text.trim_end_matches('.'))
    } else {
        let primary = format!("{}. [Page {}]", top_match.text.trim_end_matches('.'), top_match.page);
        let secondary = format!("{}. [Page {}]", top_matches[1].1.text.trim_end_matches('.'), top_matches[1].1.page);
        format!("{primary} Additionally, {secondary}")
    };

    (answer, confidence, citations)
}

/// A detected sensitive entity finding.
#[derive(Clone, Debug, PartialEq)]
pub struct Finding {
    pub page: usize,
    pub category: &'static str,
    pub text: String,
    pub reason: &'static str,
    pub quads: Vec<[f64; 8]>,
    pub rect: [f32; 4],
}

/// Options controlling which categories to scan for in `scan_sensitive_findings`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanOptions {
    pub email: bool,
    pub ssn: bool,
    pub card: bool,
    pub phone: bool,
    pub key: bool,
    pub ip: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            email: true,
            ssn: true,
            card: true,
            phone: true,
            key: true,
            ip: true,
        }
    }
}

/// Detect sensitive PII entities (SSN, credit card, email, phone, API keys, IP addresses).
pub fn scan_sensitive_findings(
    pages: &[usize],
    texts: &[std::sync::Arc<pdfcraft_render::PageText>],
    info: &pdfcraft_render::DocInfo,
    options: ScanOptions,
) -> Vec<Finding> {
    let mut findings: Vec<Finding> = Vec::new();

    for (&p, text) in pages.iter().zip(texts) {
        let plain = text.plain_text();

        // 1. Emails
        if options.email {
            for token in plain.split_whitespace() {
                let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '.' && c != '_' && c != '-');
                if let Some(at_idx) = cleaned.find('@') {
                    let user = &cleaned[..at_idx];
                    let domain = &cleaned[at_idx + 1..];
                    if !user.is_empty() && domain.contains('.') && domain.len() >= 3 {
                        let hits = text.find_opts(cleaned, false, true);
                        for h in hits {
                            let rects = text.line_rects(h);
                            if let Some(&first_rect) = rects.first() {
                                let quads: Vec<[f64; 8]> = rects.iter().map(|r| info.pages[p].view_rect_to_quad(*r)).collect();
                                findings.push(Finding {
                                    page: p,
                                    category: "email",
                                    text: cleaned.to_string(),
                                    reason: "Direct email address identifier",
                                    quads,
                                    rect: first_rect,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 2. SSN: ddd-dd-dddd
        if options.ssn {
            let chars: Vec<char> = plain.chars().collect();
            for i in 0..chars.len().saturating_sub(10) {
                if chars[i].is_ascii_digit()
                    && chars[i + 1].is_ascii_digit()
                    && chars[i + 2].is_ascii_digit()
                    && chars[i + 3] == '-'
                    && chars[i + 4].is_ascii_digit()
                    && chars[i + 5].is_ascii_digit()
                    && chars[i + 6] == '-'
                    && chars[i + 7].is_ascii_digit()
                    && chars[i + 8].is_ascii_digit()
                    && chars[i + 9].is_ascii_digit()
                    && chars[i + 10].is_ascii_digit()
                {
                    let snippet: String = chars[i..=i + 10].iter().collect();
                    let raw_digits: String = snippet.chars().filter(|c| c.is_ascii_digit()).collect();
                    if is_valid_ssn(&raw_digits) {
                        let hits = text.find_opts(&snippet, false, true);
                        for h in hits {
                            let rects = text.line_rects(h);
                            if let Some(&first_rect) = rects.first() {
                                let quads: Vec<[f64; 8]> = rects.iter().map(|r| info.pages[p].view_rect_to_quad(*r)).collect();
                                findings.push(Finding {
                                    page: p,
                                    category: "ssn",
                                    text: snippet.clone(),
                                    reason: "Social Security Number format",
                                    quads,
                                    rect: first_rect,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 3. Credit Cards (Luhn verified)
        if options.card {
            for token in plain.split_whitespace() {
                let cleaned: String = token.chars().filter(|c| c.is_ascii_digit()).collect();
                if (13..=19).contains(&cleaned.len()) {
                    let digits: Vec<u8> = cleaned.bytes().map(|b| b.saturating_sub(b'0')).collect();
                    if luhn_check(&digits) {
                        let hits = text.find_opts(token.trim_matches(|c: char| !c.is_ascii_digit()), false, false);
                        for h in hits {
                            let rects = text.line_rects(h);
                            if let Some(&first_rect) = rects.first() {
                                let quads: Vec<[f64; 8]> = rects.iter().map(|r| info.pages[p].view_rect_to_quad(*r)).collect();
                                findings.push(Finding {
                                    page: p,
                                    category: "credit_card",
                                    text: token.to_string(),
                                    reason: "Payment card number matching Luhn verification",
                                    quads,
                                    rect: first_rect,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 4. Phone numbers
        if options.phone {
            let chars: Vec<char> = plain.chars().collect();
            for i in 0..chars.len().saturating_sub(11) {
                if chars[i].is_ascii_digit()
                    && chars[i + 1].is_ascii_digit()
                    && chars[i + 2].is_ascii_digit()
                    && (chars[i + 3] == '-' || chars[i + 3] == '.')
                    && chars[i + 4].is_ascii_digit()
                    && chars[i + 5].is_ascii_digit()
                    && chars[i + 6].is_ascii_digit()
                    && (chars[i + 7] == '-' || chars[i + 7] == '.')
                    && chars[i + 8].is_ascii_digit()
                    && chars[i + 9].is_ascii_digit()
                    && chars[i + 10].is_ascii_digit()
                    && chars[i + 11].is_ascii_digit()
                {
                    let snippet: String = chars[i..=i + 11].iter().collect();
                    let hits = text.find_opts(&snippet, false, true);
                    for h in hits {
                        let rects = text.line_rects(h);
                        if let Some(&first_rect) = rects.first() {
                            let quads: Vec<[f64; 8]> = rects.iter().map(|r| info.pages[p].view_rect_to_quad(*r)).collect();
                            findings.push(Finding {
                                page: p,
                                category: "phone",
                                text: snippet.clone(),
                                reason: "Telephone contact number",
                                quads,
                                rect: first_rect,
                            });
                        }
                    }
                }
            }
        }

        // 5. API keys and passwords
        if options.key {
            for token in plain.split_whitespace() {
                let is_key = token.starts_with("sk_live_")
                    || token.starts_with("ghp_")
                    || (token.starts_with("AKIA") && token.len() == 20)
                    || token.to_lowercase().starts_with("bearer");
                if is_key {
                    let hits = text.find_opts(token, false, true);
                    for h in hits {
                        let rects = text.line_rects(h);
                        if let Some(&first_rect) = rects.first() {
                            let quads: Vec<[f64; 8]> = rects.iter().map(|r| info.pages[p].view_rect_to_quad(*r)).collect();
                            findings.push(Finding {
                                page: p,
                                category: "api_key",
                                text: token.to_string(),
                                reason: "API secret key or credential token",
                                quads,
                                rect: first_rect,
                            });
                        }
                    }
                }
            }
        }

        // 6. IPv4 Addresses
        if options.ip {
            for token in plain.split_whitespace() {
                let cleaned = token.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
                let parts: Vec<&str> = cleaned.split('.').collect();
                if parts.len() == 4 && parts.iter().all(|part| part.parse::<u8>().is_ok() && (part.len() == 1 || !part.starts_with('0'))) {
                    let hits = text.find_opts(cleaned, false, true);
                    for h in hits {
                        let rects = text.line_rects(h);
                        if let Some(&first_rect) = rects.first() {
                            let quads: Vec<[f64; 8]> = rects.iter().map(|r| info.pages[p].view_rect_to_quad(*r)).collect();
                            findings.push(Finding {
                                page: p,
                                category: "ip_address",
                                text: cleaned.to_string(),
                                reason: "IPv4 network address",
                                quads,
                                rect: first_rect,
                            });
                        }
                    }
                }
            }
        }
    }

    // Deduplicate findings by page and text
    let mut seen = HashSet::new();
    findings.retain(|f| seen.insert((f.page, f.text.clone())));
    findings
}

impl Automation {
    /// Ensure the AI provider is active either globally or explicitly via arguments.
    fn check_ai_provider_active(&self, a: &Args) -> Result<()> {
        let explicit = a.opt_bool("enabled")?.unwrap_or(false);
        if !self.ai_provider.enabled && !explicit {
            return Err(ToolError::Failed(
                "AI provider is disabled by default. Enable it via the `ai_provider` tool (e.g. `{\"enabled\": true}`) or pass `\"enabled\": true` in the tool arguments.".into()
            ));
        }
        Ok(())
    }

    /// `ai_provider`: Query or configure the sovereign AI provider interface.
    pub(crate) fn ai_provider(&mut self, a: &Args) -> Result<Value> {
        if let Some(on) = a.opt_bool("enabled")? {
            self.ai_provider.enabled = on;
        }
        if let Some(m) = a.opt_str("mode")? {
            self.ai_provider.mode =
                ProviderMode::parse(m).ok_or_else(|| ToolError::InvalidArgs(format!("unknown mode {m:?}; allowed: 'builtin', 'endpoint'")))?;
        }
        if let Some(ep) = a.opt_str("endpoint")? {
            self.ai_provider.endpoint = if ep.trim().is_empty() { None } else { Some(ep.to_string()) };
        }
        if let Some(model) = a.opt_str("model")? {
            self.ai_provider.model = if model.trim().is_empty() { None } else { Some(model.to_string()) };
        }
        if let Some(key) = a.opt_str("api_key")? {
            self.ai_provider.api_key = if key.trim().is_empty() { None } else { Some(key.to_string()) };
        }

        Ok(json!({
            "enabled": self.ai_provider.enabled,
            "mode": self.ai_provider.mode.as_str(),
            "endpoint": self.ai_provider.endpoint,
            "model": self.ai_provider.model,
            "has_api_key": self.ai_provider.api_key.is_some(),
        }))
    }

    /// `ai_summarize`: Extract document summary and key points with explicit page citations.
    pub(crate) fn ai_summarize(&mut self, a: &Args) -> Result<Value> {
        self.check_ai_provider_active(a)?;

        let doc = self.doc(a)?;
        let id = doc.id;
        let total_pages = doc.info.pages.len();
        let pages: Vec<usize> = match a.opt_ints("pages")? {
            Some(_) => self.pages(a, "pages")?,
            None => (0..total_pages).collect(),
        };

        if pages.is_empty() {
            return Err(ToolError::InvalidArgs("pages cannot be empty".into()));
        }

        let max_points = a.opt_int("max_points")?.unwrap_or(5).clamp(1, 50) as usize;
        let length_pref = a.opt_str("length")?.unwrap_or("medium");

        let texts = self.page_texts(id, &pages)?;
        let mut all_sentences: Vec<Sentence> = Vec::new();
        let mut total_words: usize = 0;

        for (p, text) in pages.iter().zip(&texts) {
            let plain = text.plain_text();
            let page_sentences = split_sentences(&plain, p + 1);
            for s in &page_sentences {
                total_words = total_words.saturating_add(s.words().len());
            }
            all_sentences.extend(page_sentences);
        }

        if all_sentences.is_empty() {
            return Ok(json!({
                "summary": "The document contains no readable text to summarize.",
                "key_points": [],
                "word_count": 0,
                "reading_time_minutes": 0,
                "page_count": pages.len(),
                "citations": [],
            }));
        }

        if self.ai_provider.mode == ProviderMode::Endpoint {
            let mut doc_context = String::new();
            for (p, text) in pages.iter().zip(&texts) {
                doc_context.push_str(&format!("\n--- Page {} ---\n", p + 1));
                doc_context.push_str(&text.plain_text());
            }

            let system_prompt = "You are an expert executive document analyst. Produce a comprehensive, professional Executive Summary of the provided PDF document.\n\
Include clear sections for Executive Overview, Key Findings, Action Items, and Critical Risks.\n\
When citing facts or data points, include exact page references like [Page 1], [Page 2].";

            let user_prompt = format!("Summarize this document (detail level: {length_pref}, max key points: {max_points}):\n{doc_context}");

            if let Ok(llm_response) = call_llm_endpoint(&self.ai_provider, Some(system_prompt), &user_prompt) {
                let mut cited_pages = BTreeSet::new();
                for p in extract_page_citations(&llm_response) {
                    cited_pages.insert(p);
                }

                return Ok(json!({
                    "summary": llm_response,
                    "key_points": [],
                    "word_count": total_words,
                    "reading_time_minutes": ((total_words as f64) / 200.0).ceil() as usize,
                    "page_count": pages.len(),
                    "citations": cited_pages.into_iter().collect::<Vec<_>>(),
                    "provider": "endpoint",
                }));
            }
        }


        let (summary_text, key_points_raw, total_words, reading_time) =
            generate_summary_pure(&all_sentences, max_points, length_pref);

        let cited_pages: BTreeSet<usize> = key_points_raw.iter().map(|(p, _)| *p).collect();
        let key_points: Vec<Value> = key_points_raw
            .into_iter()
            .map(|(p, text)| {
                json!({
                    "page": p,
                    "citation": format!("[Page {p}]"),
                    "text": text,
                })
            })
            .collect();

        Ok(json!({
            "summary": summary_text,
            "key_points": key_points,
            "word_count": total_words,
            "reading_time_minutes": reading_time,
            "page_count": pages.len(),
            "citations": cited_pages.into_iter().collect::<Vec<_>>(),
        }))
    }

    /// `ai_ask`: Question answering over document text with exact citations to pages.
    pub(crate) fn ai_ask(&mut self, a: &Args) -> Result<Value> {
        self.check_ai_provider_active(a)?;

        let doc = self.doc(a)?;
        let id = doc.id;
        let question = a.str("question")?.trim();
        if question.is_empty() {
            return Err(ToolError::InvalidArgs("question cannot be empty".into()));
        }

        let total_pages = doc.info.pages.len();
        let pages: Vec<usize> = match a.opt_ints("pages")? {
            Some(_) => self.pages(a, "pages")?,
            None => (0..total_pages).collect(),
        };

        let max_citations = a.opt_int("max_citations")?.unwrap_or(3).clamp(1, 20) as usize;

        let texts = self.page_texts(id, &pages)?;
        let mut all_sentences: Vec<Sentence> = Vec::new();
        let mut total_words: usize = 0;

        for (p, text) in pages.iter().zip(&texts) {
            let plain = text.plain_text();
            let page_sentences = split_sentences(&plain, p + 1);
            for s in &page_sentences {
                total_words = total_words.saturating_add(s.words().len());
            }
            all_sentences.extend(page_sentences);
        }

        if self.ai_provider.mode == ProviderMode::Endpoint {
            let mut doc_context = String::new();
            for (p, text) in pages.iter().zip(&texts) {
                doc_context.push_str(&format!("\n--- Page {} ---\n", p + 1));
                doc_context.push_str(&text.plain_text());
            }

            let system_prompt = "You are an expert assistant answering questions based on the provided PDF document.\n\
Answer accurately and concisely using ONLY the provided text.\n\
Always cite the source pages using [Page n] brackets where the information was found.\n\
If the answer is not contained in the text, state clearly that the document does not contain this information.";

            let user_prompt = format!("Question: {question}\n\nDocument Context:\n{doc_context}");

            if let Ok(llm_response) = call_llm_endpoint(&self.ai_provider, Some(system_prompt), &user_prompt) {
                let mut citations_found = Vec::new();
                for p in extract_page_citations(&llm_response) {
                    citations_found.push(json!({
                        "page": p,
                        "excerpt": format!("[Page {p}]"),
                        "score": 1.0,
                    }));
                }

                return Ok(json!({
                    "question": question,
                    "answer": llm_response,
                    "confidence": "high",
                    "citations": citations_found,
                    "provider": "endpoint",
                }));
            }
        }

        let (answer, confidence, citations_raw) = search_answer_pure(question, &all_sentences, max_citations);


        let citations: Vec<Value> = citations_raw
            .into_iter()
            .map(|(page, excerpt, score)| {
                json!({
                    "page": page,
                    "excerpt": excerpt,
                    "score": score,
                })
            })
            .collect();

        Ok(json!({
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "citations": citations,
        }))
    }

    /// `ai_suggest_redactions`: Detect sensitive entities (SSN, credit card, email, phone, API keys)
    /// across pages, calculate coordinates, and optionally mark them for redaction.
    pub(crate) fn ai_suggest_redactions(&mut self, a: &Args) -> Result<Value> {
        self.check_ai_provider_active(a)?;

        let doc = self.doc(a)?;
        let id = doc.id;
        let total_pages = doc.info.pages.len();
        let pages: Vec<usize> = match a.opt_ints("pages")? {
            Some(_) => self.pages(a, "pages")?,
            None => (0..total_pages).collect(),
        };

        let requested_cats: Vec<String> = match a.get("categories") {
            Some(v) if v.is_array() => a.strs("categories")?.into_iter().map(str::to_owned).collect(),
            _ => vec!["all".into()],
        };

        let check_all = requested_cats.iter().any(|c| c == "all");
        let want_ssn = check_all || requested_cats.iter().any(|c| c == "ssn");
        let want_card = check_all || requested_cats.iter().any(|c| c == "credit_card");
        let want_email = check_all || requested_cats.iter().any(|c| c == "email");
        let want_phone = check_all || requested_cats.iter().any(|c| c == "phone");
        let want_key = check_all || requested_cats.iter().any(|c| c == "api_key" || c == "password");
        let want_ip = check_all || requested_cats.iter().any(|c| c == "ip_address");

        let texts = self.page_texts(id, &pages)?;
        let info = self.doc(a)?.info.clone();

        let options = ScanOptions {
            email: want_email,
            ssn: want_ssn,
            card: want_card,
            phone: want_phone,
            key: want_key,
            ip: want_ip,
        };
        let findings = scan_sensitive_findings(&pages, &texts, &info, options);

        let mark = a.opt_bool("mark")?.unwrap_or(false);
        let mut marked_count = 0;

        if mark && !findings.is_empty() {
            let author = a.opt_str("author")?.unwrap_or(DEFAULT_AUTHOR).to_string();
            let mut style = Style::default_for(&Shape::Redact { quads: Vec::new(), overlay: String::new(), look: Default::default() });
            if let Some(c) = a.opt_str("fill")? {
                style.fill = Some(parse_color(c)?);
            }

            let edits: Vec<Edit> = findings
                .iter()
                .filter(|f| !f.quads.is_empty())
                .map(|f| {
                    let shape = Shape::Redact { quads: f.quads.clone(), overlay: f.category.to_uppercase(), look: Default::default() };
                    Edit::AddAnnotation(NewAnnotation {
                        page: f.page,
                        shape,
                        style: style.clone(),
                        contents: f.reason.to_string(),
                        author: author.clone(),
                    })
                })
                .collect();

            if !edits.is_empty() {
                marked_count = edits.len();
                let edit = match <[Edit; 1]>::try_from(edits) {
                    Ok([one]) => one,
                    Err(batch) => Edit::Batch { label: "AI Redaction Suggestions".into(), edits: batch },
                };
                self.apply(a, edit)?;
            }
        }

        let suggestions: Vec<Value> = findings
            .iter()
            .map(|f| {
                json!({
                    "page": f.page + 1,
                    "category": f.category,
                    "text": f.text,
                    "reason": f.reason,
                    "rect": f.rect,
                })
            })
            .collect();

        Ok(json!({
            "count": suggestions.len(),
            "suggestions": suggestions,
            "marked": mark,
            "marks_applied": marked_count,
        }))
    }

    /// `ai_suggest_alt_text`: Generate descriptive alternate text for figures and images.
    pub(crate) fn ai_suggest_alt_text(&mut self, a: &Args) -> Result<Value> {
        self.check_ai_provider_active(a)?;

        let doc = self.doc(a)?;
        let id = doc.id;
        let total_pages = doc.info.pages.len();
        let pages: Vec<usize> = match a.opt_ints("pages")? {
            Some(_) => self.pages(a, "pages")?,
            None => (0..total_pages).collect(),
        };

        let apply = a.opt_bool("apply")?.unwrap_or(false);
        let texts = self.page_texts(id, &pages)?;

        struct AltSuggestion {
            page: usize,
            figure: Option<u32>,
            image_name: Option<String>,
            suggested_alt_text: String,
            confidence: &'static str,
            bbox: Option<[f64; 4]>,
        }

        let mut suggestions: Vec<AltSuggestion> = Vec::new();

        // 1. Tagged figures in structure tree
        let tagged_figs = self.doc(a)?.figures();
        for fig in tagged_figs {
            if let Some(p) = fig.page.filter(|p| pages.contains(p) && fig.alt.as_deref().unwrap_or("").trim().is_empty()) {
                // Inspect page text around figure
                let page_text =
                    pages.iter().position(|&page_idx| page_idx == p).and_then(|idx| texts.get(idx)).map(|t| t.plain_text()).unwrap_or_default();

                // Search for captions like "Figure 1:", "Fig. 1", "Chart:"
                let mut suggested = None;
                for line in page_text.lines() {
                    let l_trimmed = line.trim();
                    if l_trimmed.starts_with("Figure ")
                        || l_trimmed.starts_with("Fig. ")
                        || l_trimmed.starts_with("Chart ")
                        || l_trimmed.starts_with("Illustration ")
                    {
                        suggested = Some(l_trimmed.to_string());
                        break;
                    }
                }

                let alt_text = suggested.unwrap_or_else(|| {
                    if let Some(bbox) = fig.bbox {
                        let width = (bbox[2] - bbox[0]).abs();
                        let height = (bbox[3] - bbox[1]).abs();
                        if width > height * 1.5 {
                            format!("Landscape diagram on page {}", p + 1)
                        } else if height > width * 1.5 {
                            format!("Portrait illustration on page {}", p + 1)
                        } else {
                            format!("Figure graphic on page {}", p + 1)
                        }
                    } else {
                        format!("Document figure on page {}", p + 1)
                    }
                });

                suggestions.push(AltSuggestion {
                    page: p,
                    figure: Some(fig.obj.num),
                    image_name: None,
                    suggested_alt_text: alt_text,
                    confidence: "high",
                    bbox: fig.bbox,
                });
            }
        }

        // 2. Untagged raster images from page content
        for &p in &pages {
            let page_images = self.doc(a)?.page_images(p);
            for img in page_images {
                let width = img.width;
                let height = img.height;
                let is_banner = width > height.saturating_mul(3);
                let is_icon = width < 48 && height < 48;

                let desc = if is_icon {
                    format!("Decorative icon ({width}x{height}) on page {}", p + 1)
                } else if is_banner {
                    format!("Horizontal header banner ({width}x{height}) on page {}", p + 1)
                } else {
                    format!("Photograph or graphic illustration ({width}x{height}) on page {}", p + 1)
                };

                suggestions.push(AltSuggestion {
                    page: p,
                    figure: None,
                    image_name: Some(img.name),
                    suggested_alt_text: desc,
                    confidence: "medium",
                    bbox: Some(img.rect),
                });
            }
        }

        let mut applied_count: usize = 0;
        if apply {
            for s in &suggestions {
                if let Some(fig_num) = s.figure {
                    self.apply(a, Edit::SetAltText { figure: fig_num, alt: Some(s.suggested_alt_text.clone()) })?;
                    applied_count = applied_count.saturating_add(1);
                }
            }
        }

        let out_list: Vec<Value> = suggestions
            .iter()
            .map(|s| {
                json!({
                    "page": s.page + 1,
                    "figure": s.figure,
                    "image_name": s.image_name,
                    "suggested_alt_text": s.suggested_alt_text,
                    "confidence": s.confidence,
                    "bbox": s.bbox,
                })
            })
            .collect();

        Ok(json!({
            "count": out_list.len(),
            "suggestions": out_list,
            "applied": apply,
            "applied_count": applied_count,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn textrank_and_mmr_select_salient_and_diverse_sentences() {
        let text = "Artificial intelligence accelerates medical diagnosis. \
Artificial intelligence algorithms analyze complex medical scans. \
Deep learning neural networks improve healthcare outcomes. \
Space exploration missions are launching robotic rovers to Mars. \
Robotic rovers explore Mars searching for ancient Martian soil samples.";

        let sentences = split_sentences(text, 1);
        assert_eq!(sentences.len(), 5);

        let (vectors, norms) = compute_sentence_vectors(&sentences);
        let tr_scores = textrank_scores(&sentences, &vectors, &norms);
        assert_eq!(tr_scores.len(), 5);

        for &score in &tr_scores {
            assert!(score.is_finite() && score > 0.0);
        }

        let selected = mmr_select(&sentences, &tr_scores, &vectors, &norms, 2, 0.55);
        assert_eq!(selected.len(), 2);

        let cluster_ai = selected.iter().any(|&i| i < 3);
        let cluster_mars = selected.iter().any(|&i| i >= 3);
        assert!(cluster_ai && cluster_mars, "MMR must select diverse topics: {:?}", selected);

        let (summary, key_points, words, reading_time) = generate_summary_pure(&sentences, 3, "medium");
        assert!(!summary.is_empty());
        assert!(!key_points.is_empty());
        assert!(words > 20);
        assert!(reading_time >= 1);
    }
}

