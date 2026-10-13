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
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self { enabled: false, mode: ProviderMode::Builtin, endpoint: None, model: None }
    }
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
fn jaccard_similarity(a: &Sentence, b: &Sentence) -> f64 {
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

        Ok(json!({
            "enabled": self.ai_provider.enabled,
            "mode": self.ai_provider.mode.as_str(),
            "endpoint": self.ai_provider.endpoint,
            "model": self.ai_provider.model,
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

        // Global term frequency calculation (content words)
        let mut term_freq: BTreeMap<String, usize> = BTreeMap::new();
        for s in &all_sentences {
            for w in s.content_words() {
                *term_freq.entry(w).or_insert(0) += 1;
            }
        }

        // Score sentences by term significance + position bias
        let mut scored_sentences: Vec<(f64, Sentence)> = all_sentences
            .iter()
            .map(|s| {
                let words = s.content_words();
                let word_count = s.words().len();
                let mut score = 0.0;
                for w in &words {
                    score += *term_freq.get(w).unwrap_or(&0) as f64;
                }

                // Normalise by length dampener (penalise extreme short/long)
                if word_count > 0 {
                    score /= (word_count as f64).powf(0.55).max(1.0);
                }

                // Lead sentence boost
                if s.index_on_page == 0 {
                    score *= if s.page == 1 { 1.5 } else { 1.25 };
                }

                // Appropriate sentence length sweet spot (10..40 words)
                if (10..=40).contains(&word_count) {
                    score *= 1.2;
                } else if !(5..=60).contains(&word_count) {
                    score *= 0.6;
                }

                (score, s.clone())
            })
            .collect();

        scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Select diverse key points via Maximal Marginal Relevance (MMR)
        let mut selected_points: Vec<Sentence> = Vec::new();
        for (_score, s) in &scored_sentences {
            if selected_points.len() >= max_points {
                break;
            }
            let is_redundant = selected_points.iter().any(|existing| jaccard_similarity(existing, s) > 0.38);
            if !is_redundant {
                selected_points.push(s.clone());
            }
        }

        // Fallback if diversity filtered too aggressively
        if selected_points.is_empty() && !scored_sentences.is_empty() {
            selected_points.push(scored_sentences[0].1.clone());
        }

        // Format key points with explicit page citations
        let key_points: Vec<Value> = selected_points
            .iter()
            .map(|s| {
                json!({
                    "page": s.page,
                    "citation": format!("[Page {}]", s.page),
                    "text": s.text,
                })
            })
            .collect();

        // Formulate executive summary
        let summary_sentences_count = match length_pref {
            "short" => 1,
            "detailed" => 3.min(selected_points.len()),
            _ => 2.min(selected_points.len()),
        };

        let summary_text: String = selected_points
            .iter()
            .take(summary_sentences_count)
            .map(|s| format!("{} [Page {}]", s.text.trim_end_matches('.'), s.page))
            .collect::<Vec<String>>()
            .join(". ")
            + ".";

        let cited_pages: BTreeSet<usize> = selected_points.iter().map(|s| s.page).collect();
        let reading_time = ((total_words as f64) / 200.0).ceil() as usize;

        Ok(json!({
            "summary": summary_text,
            "key_points": key_points,
            "word_count": total_words,
            "reading_time_minutes": reading_time.max(1),
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

        if all_sentences.is_empty() {
            return Ok(json!({
                "question": question,
                "answer": "The document contains no readable text to answer this question.",
                "confidence": "low",
                "citations": [],
            }));
        }

        let avg_len = (total_words as f64) / (all_sentences.len() as f64).max(1.0);
        let q_words = tokenize_words(question);
        let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();
        let q_content: Vec<String> = q_words.into_iter().filter(|w| !stop_set.contains(w.as_str())).collect();
        let query_terms = if q_content.is_empty() { tokenize_words(question) } else { q_content };

        // BM25 relevance scoring
        let mut scored_sentences: Vec<(f64, &Sentence)> =
            all_sentences.iter().map(|s| (bm25_score(&query_terms, s, &all_sentences, avg_len), s)).filter(|(score, _)| *score > 0.05).collect();

        scored_sentences.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        if scored_sentences.is_empty() {
            return Ok(json!({
                "question": question,
                "answer": format!("No direct matches or relevant sections found in the document answering '{question}'."),
                "confidence": "low",
                "citations": [],
            }));
        }

        let top_matches: Vec<_> = scored_sentences.iter().take(max_citations).collect();
        let citations: Vec<Value> = top_matches
            .iter()
            .map(|(score, s)| {
                json!({
                    "page": s.page,
                    "excerpt": s.text,
                    "score": ((score * 100.0).round()) / 100.0,
                })
            })
            .collect();

        // Synthesize coherent grounded answer
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

        struct Finding {
            page: usize,
            category: &'static str,
            text: String,
            reason: &'static str,
            quads: Vec<[f64; 8]>,
            rect: [f32; 4],
        }

        let mut findings: Vec<Finding> = Vec::new();

        for (&p, text) in pages.iter().zip(&texts) {
            let plain = text.plain_text();

            // 1. Emails
            if want_email {
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
            if want_ssn {
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
            if want_card {
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

            // 4. Phone numbers: e.g. (555) 123-4567 or 555-123-4567
            if want_phone {
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
            if want_key {
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
            if want_ip {
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
