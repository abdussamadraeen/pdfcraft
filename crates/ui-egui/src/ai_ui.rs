//! Sovereign Local AI Assistant Panel (M13: `misc.ai-provider`, `misc.ai-summary`, `misc.ai-ask`,
//! `protect.redact-ai-suggestions`, `a11y.alt-text-suggestions`).
//!
//! # Architecture & Principles
//! - **100% Sovereign & Offline First**: Executes entirely on the local machine using built-in
//!   statistical NLP and BM25 semantic retrieval with zero cloud dependencies or telemetry leaks.
//! - **Adaptive Theme Excellence**: Strictly adheres to design `Tokens` (`t.panel`, `t.card`,
//!   `t.border`, `t.text`, `t.accent`, `t.field`), ensuring gorgeous contrast and visual harmony
//!   in both Light and Dark themes.
//! - **Interactive Grounded Citations**: Direct page citation badges that navigate and flash
//!   the document viewport to the exact referenced location.
//! - **Actionable Redactions**: Direct 1-click generation of ISO 32000-2 `Shape::Redact` marks
//!   for detected sensitive entities (SSN, payment cards, emails, phone numbers, API keys).
//! - **Clean-Room Safe Rust**: Never panics, checked bounds, saturating calculations.

use std::sync::Arc;

use egui::{Align, Color32, CornerRadius, Layout, Rect, Stroke, vec2};
use pdfcraft_automation::ai::{
    ProviderMode, ScanOptions, Sentence, generate_summary_pure, scan_sensitive_findings, search_answer_pure, split_sentences,
};
use pdfcraft_render::PageText;

use crate::canvas::DocView;
use crate::theme::{self, Tokens};
use crate::{icons, widgets};

/// Role of an entity in the conversation stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AiRole {
    User,
    Assistant,
}

/// An interactive citation pointing to a 1-based document page.
#[derive(Clone, Debug, PartialEq)]
pub struct CitationUi {
    pub page: usize,
    pub score: Option<f64>,
}

/// A detected sensitive PII finding with page coordinate information.
#[derive(Clone, Debug, PartialEq)]
pub struct FindingUi {
    pub page: usize,
    pub category: &'static str,
    pub text: String,
    pub reason: &'static str,
    pub quads: Vec<[f64; 8]>,
    pub rect: [f32; 4],
}

/// A suggested accessible alternate text caption for a figure or image.
#[derive(Clone, Debug, PartialEq)]
pub struct AltSuggestionUi {
    pub page: usize,
    pub figure: Option<u32>,
    pub suggested_alt_text: String,
    pub bbox: Option<[f64; 4]>,
}

/// A message in the conversation stream.
#[derive(Clone, Debug, PartialEq)]
pub struct AiMessage {
    pub role: AiRole,
    pub content: String,
    pub citations: Vec<CitationUi>,
    pub findings: Vec<FindingUi>,
    pub alt_suggestions: Vec<AltSuggestionUi>,
    pub key_points: Vec<(usize, String)>,
}

/// High-level executive document metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct DocTelemetry {
    pub total_pages: usize,
    pub total_words: usize,
    pub reading_time_minutes: usize,
}

/// Length preference for extractive summarization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SummaryLength {
    Short,
    Balanced,
    Detailed,
}

impl SummaryLength {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Short => "short",
            Self::Balanced => "medium",
            Self::Detailed => "detailed",
        }
    }
}

/// The state of the AI assistant panel.
pub struct AiState {
    pub history: Vec<AiMessage>,
    pub input: String,
    pub is_busy: bool,
    pub provider_mode: ProviderMode,
    pub endpoint_url: String,
    pub model_name: String,
    pub show_settings: bool,
    pub telemetry: Option<DocTelemetry>,
    pub last_doc_id: Option<pdfcraft_engine::DocId>,
    pub summary_length: SummaryLength,
}

impl Default for AiState {
    fn default() -> Self {
        Self {
            history: Vec::new(),
            input: String::new(),
            is_busy: false,
            provider_mode: ProviderMode::Builtin,
            endpoint_url: "http://localhost:11434".to_string(),
            model_name: "llama3.2".to_string(),
            show_settings: false,
            telemetry: None,
            last_doc_id: None,
            summary_length: SummaryLength::Balanced,
        }
    }
}

/// Actions dispatched from the AI panel to the parent application.
pub enum AiPanelAction {
    /// Navigate viewport to 0-based page index.
    GoToPage(usize),
    /// Flash bounding box rectangle on 0-based page.
    FlashRect(usize, [f32; 4]),
    /// Apply a single redaction mark on the page.
    ApplyRedaction { page: usize, quads: Vec<[f64; 8]>, overlay: String, reason: String },
    /// Apply all detected redaction marks in one batch undo step.
    ApplyAllRedactions(Vec<(usize, Vec<[f64; 8]>, String, String)>),
    /// Set alternate text property for a figure object.
    ApplyAltText { figure: u32, alt: String },
}

/// Extract or retrieve cached text for all pages of a document.
fn extract_all_page_texts(doc: &pdfcraft_engine::Document, view: &DocView) -> Vec<(usize, Arc<PageText>)> {
    let pages: Vec<usize> = (0..doc.info.pages.len()).collect();
    let mut missing = Vec::new();
    let mut result: Vec<(usize, Arc<PageText>)> = Vec::with_capacity(pages.len());

    for &p in &pages {
        if let Some(t) = view.texts.get(&p) {
            result.push((p, t.clone()));
        } else {
            missing.push(p);
        }
    }

    if !missing.is_empty() {
        let extracted = pdfcraft_automation::extract_parallel(&doc.bytes, None, &missing);
        for (p, res) in extracted {
            if let Ok(t) = res {
                result.push((p, t));
            }
        }
    }

    result.sort_by_key(|(p, _)| *p);
    result
}

/// Extract all sentences from page texts.
fn collect_all_sentences(texts: &[(usize, Arc<PageText>)]) -> Vec<Sentence> {
    let mut all_sentences = Vec::new();
    for (page_idx, text) in texts {
        let page_num = page_idx.saturating_add(1);
        let plain = text.plain_text();
        let page_sentences = split_sentences(&plain, page_num);
        all_sentences.extend(page_sentences);
    }
    all_sentences
}

impl AiState {
    /// Invalidate telemetry if active document changed.
    pub fn sync_doc(&mut self, doc: &pdfcraft_engine::Document, view: &DocView) {
        if self.last_doc_id != Some(doc.id) {
            self.last_doc_id = Some(doc.id);
            self.telemetry = None;
            self.compute_telemetry(doc, view);
        }
    }

    /// Compute document telemetry (pages, word count, reading time).
    pub fn compute_telemetry(&mut self, doc: &pdfcraft_engine::Document, view: &DocView) {
        let texts = extract_all_page_texts(doc, view);
        let mut total_words: usize = 0;
        for (_, text) in &texts {
            let plain = text.plain_text();
            total_words = total_words.saturating_add(plain.split_whitespace().count());
        }
        let reading_time = ((total_words as f64) / 200.0).ceil() as usize;
        self.telemetry = Some(DocTelemetry {
            total_pages: doc.info.pages.len(),
            total_words,
            reading_time_minutes: reading_time.max(1),
        });
    }

    /// Generate executive document summary.
    pub fn trigger_summary(&mut self, doc: &pdfcraft_engine::Document, view: &DocView) {
        self.history.push(AiMessage {
            role: AiRole::User,
            content: "Summarize this document".to_string(),
            citations: Vec::new(),
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: Vec::new(),
        });

        let texts = extract_all_page_texts(doc, view);
        let sentences = collect_all_sentences(&texts);
        let (summary_text, key_points_raw, total_words, reading_time) =
            generate_summary_pure(&sentences, 5, self.summary_length.as_str());

        self.telemetry = Some(DocTelemetry {
            total_pages: doc.info.pages.len(),
            total_words,
            reading_time_minutes: reading_time,
        });

        let mut citations_set = std::collections::BTreeSet::new();
        for (page, _) in &key_points_raw {
            citations_set.insert(*page);
        }
        let citations: Vec<CitationUi> = citations_set.into_iter().map(|p| CitationUi { page: p, score: None }).collect();

        self.history.push(AiMessage {
            role: AiRole::Assistant,
            content: summary_text,
            citations,
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: key_points_raw,
        });
    }

    /// Generate key takeaways.
    pub fn trigger_key_insights(&mut self, doc: &pdfcraft_engine::Document, view: &DocView) {
        self.history.push(AiMessage {
            role: AiRole::User,
            content: "Extract key insights and core takeaways".to_string(),
            citations: Vec::new(),
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: Vec::new(),
        });

        let texts = extract_all_page_texts(doc, view);
        let sentences = collect_all_sentences(&texts);
        let (_, key_points_raw, _, _) = generate_summary_pure(&sentences, 6, "detailed");

        let mut citations_set = std::collections::BTreeSet::new();
        for (page, _) in &key_points_raw {
            citations_set.insert(*page);
        }
        let citations: Vec<CitationUi> = citations_set.into_iter().map(|p| CitationUi { page: p, score: None }).collect();

        self.history.push(AiMessage {
            role: AiRole::Assistant,
            content: "Here are the primary insights and core takeaways extracted from the document:".to_string(),
            citations,
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: key_points_raw,
        });
    }

    /// Scan for sensitive PII and compliance findings.
    pub fn trigger_pii_scan(&mut self, doc: &pdfcraft_engine::Document, view: &DocView) {
        self.history.push(AiMessage {
            role: AiRole::User,
            content: "Audit document for sensitive PII and confidential credentials".to_string(),
            citations: Vec::new(),
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: Vec::new(),
        });

        let texts_pair = extract_all_page_texts(doc, view);
        let pages: Vec<usize> = texts_pair.iter().map(|(p, _)| *p).collect();
        let texts: Vec<Arc<PageText>> = texts_pair.into_iter().map(|(_, t)| t).collect();

        let raw_findings = scan_sensitive_findings(&pages, &texts, &doc.info, ScanOptions::default());

        if raw_findings.is_empty() {
            self.history.push(AiMessage {
                role: AiRole::Assistant,
                content: "Privacy audit complete. Zero unmasked sensitive entities (SSNs, payment cards, emails, phone numbers, API keys, or IP addresses) detected across any page.".to_string(),
                citations: Vec::new(),
                findings: Vec::new(),
                alt_suggestions: Vec::new(),
                key_points: Vec::new(),
            });
        } else {
            let count = raw_findings.len();
            let findings_ui: Vec<FindingUi> = raw_findings
                .into_iter()
                .map(|f| FindingUi {
                    page: f.page,
                    category: f.category,
                    text: f.text,
                    reason: f.reason,
                    quads: f.quads,
                    rect: f.rect,
                })
                .collect();

            let header_msg = if count == 1 {
                "Detected 1 sensitive entity requiring redaction verification:".to_string()
            } else {
                format!("Detected {count} sensitive entities requiring redaction verification:")
            };

            self.history.push(AiMessage {
                role: AiRole::Assistant,
                content: header_msg,
                citations: Vec::new(),
                findings: findings_ui,
                alt_suggestions: Vec::new(),
                key_points: Vec::new(),
            });
        }
    }

    /// Scan and generate WCAG 2.2 accessible alt-text suggestions.
    pub fn trigger_alt_text_scan(&mut self, doc: &pdfcraft_engine::Document, view: &DocView) {
        self.history.push(AiMessage {
            role: AiRole::User,
            content: "Audit accessibility: suggest alt-text for figures and illustrations".to_string(),
            citations: Vec::new(),
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: Vec::new(),
        });

        let texts_pair = extract_all_page_texts(doc, view);
        let mut suggestions = Vec::new();

        let tagged_figs = doc.figures();
        for fig in tagged_figs {
            if let Some(p) = fig.page.filter(|_| fig.alt.as_deref().unwrap_or("").trim().is_empty()) {
                let page_text = texts_pair
                    .iter()
                    .find(|(idx, _)| *idx == p)
                    .map(|(_, t)| t.plain_text())
                    .unwrap_or_default();

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

                suggestions.push(AltSuggestionUi {
                    page: p,
                    figure: Some(fig.obj.num),
                    suggested_alt_text: alt_text,
                    bbox: fig.bbox,
                });
            }
        }

        if suggestions.is_empty() {
            self.history.push(AiMessage {
                role: AiRole::Assistant,
                content: "Accessibility audit complete. All figures and graphics in this document already have accessible alternative text.".to_string(),
                citations: Vec::new(),
                findings: Vec::new(),
                alt_suggestions: Vec::new(),
                key_points: Vec::new(),
            });
        } else {
            let count = suggestions.len();
            self.history.push(AiMessage {
                role: AiRole::Assistant,
                content: format!("Identified {count} figures missing accessible descriptions:"),
                citations: Vec::new(),
                findings: Vec::new(),
                alt_suggestions: suggestions,
                key_points: Vec::new(),
            });
        }
    }

    /// Ask user question using BM25 semantic retrieval over sentences.
    pub fn trigger_question(&mut self, question: &str, doc: &pdfcraft_engine::Document, view: &DocView) {
        let q_trimmed = question.trim();
        if q_trimmed.is_empty() {
            return;
        }

        self.history.push(AiMessage {
            role: AiRole::User,
            content: q_trimmed.to_string(),
            citations: Vec::new(),
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: Vec::new(),
        });

        let texts = extract_all_page_texts(doc, view);
        let sentences = collect_all_sentences(&texts);
        let (answer, confidence, citations_raw) = search_answer_pure(q_trimmed, &sentences, 3);

        let citations: Vec<CitationUi> = citations_raw
            .into_iter()
            .map(|(page, _, score)| CitationUi { page, score: Some(score) })
            .collect();

        let formatted_answer = match confidence {
            "high" => answer,
            _ => format!("{answer} (Note: Moderate match confidence based on offline document index)"),
        };

        self.history.push(AiMessage {
            role: AiRole::Assistant,
            content: formatted_answer,
            citations,
            findings: Vec::new(),
            alt_suggestions: Vec::new(),
            key_points: Vec::new(),
        });
    }
}

/// Paint an interactive executive telemetry badge card.
fn render_telemetry_card(ui: &mut egui::Ui, t: &Tokens, telemetry: &DocTelemetry) {
    let frame = egui::Frame::new()
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(10, 8));

    frame.show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = vec2(8.0, 4.0);

            // Pages badge
            ui.horizontal(|ui| {
                icons::paint(ui, Rect::from_min_size(ui.cursor().min, vec2(14.0, 14.0)), "files", 13.0, t.accent);
                ui.add_space(16.0);
                ui.label(egui::RichText::new(format!("{} pages", telemetry.total_pages)).font(theme::medium(11.5)).color(t.text));
            });

            ui.label(egui::RichText::new("·").color(t.divider));

            // Words badge
            ui.horizontal(|ui| {
                icons::paint(ui, Rect::from_min_size(ui.cursor().min, vec2(14.0, 14.0)), "file-text", 13.0, t.icon);
                ui.add_space(16.0);
                ui.label(egui::RichText::new(format!("{} words", telemetry.total_words)).font(theme::medium(11.5)).color(t.text));
            });

            ui.label(egui::RichText::new("·").color(t.divider));

            // Reading time badge
            ui.horizontal(|ui| {
                icons::paint(ui, Rect::from_min_size(ui.cursor().min, vec2(14.0, 14.0)), "clock", 13.0, t.icon);
                ui.add_space(16.0);
                ui.label(
                    egui::RichText::new(format!("~{} min read", telemetry.reading_time_minutes)).font(theme::medium(11.5)).color(t.text),
                );
            });

            ui.label(egui::RichText::new("·").color(t.divider));

            // Sovereign Shield badge
            ui.horizontal(|ui| {
                let shield_color = Color32::from_rgb(0x10, 0xB9, 0x81);
                icons::paint(ui, Rect::from_min_size(ui.cursor().min, vec2(14.0, 14.0)), "shield-check", 13.0, shield_color);
                ui.add_space(16.0);
                ui.label(egui::RichText::new("Local & Sovereign").font(theme::semibold(11.5)).color(shield_color))
                    .on_hover_text("100% Offline: No document content is ever sent to external cloud servers.");
            });
        });
    });
}

/// Paint quick prompt pill buttons.
fn render_quick_prompts(
    ui: &mut egui::Ui,
    t: &Tokens,
    state: &mut AiState,
    doc: &pdfcraft_engine::Document,
    view: &DocView,
) {
    widgets::section_title(ui, "Quick Actions");

    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(6.0, 6.0);

        let quick_btn = |ui: &mut egui::Ui, label: &str, tip: &str| {
            ui.add(
                egui::Button::new(egui::RichText::new(label).font(theme::medium(12.0)).color(t.text))
                    .fill(t.card)
                    .stroke(Stroke::new(1.0, t.border))
                    .corner_radius(CornerRadius::same(12)),
            )
            .on_hover_text(tip)
            .clicked()
        };

        if quick_btn(ui, "✨ Summarize", "Generate an executive extractive summary of this document") {
            state.trigger_summary(doc, view);
        }

        if quick_btn(ui, "🎯 Key Insights", "Extract high-impact key takeaway bullet points") {
            state.trigger_key_insights(doc, view);
        }

        if quick_btn(ui, "🛡️ Scan Sensitive PII", "Audit for SSNs, payment cards, emails, phone numbers, and API credentials") {
            state.trigger_pii_scan(doc, view);
        }

        if quick_btn(ui, "🏷️ Alt-Text Suggestions", "Audit figures and generate accessible alt-text descriptions") {
            state.trigger_alt_text_scan(doc, view);
        }
    });

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Summary Detail:").font(theme::medium(11.0)).color(t.text_faint));
        let lengths = [
            (SummaryLength::Short, "Short"),
            (SummaryLength::Balanced, "Balanced"),
            (SummaryLength::Detailed, "Detailed"),
        ];
        for (len_variant, label) in lengths {
            let selected = state.summary_length == len_variant;
            let btn = ui.add(
                egui::Button::new(
                    egui::RichText::new(label)
                        .font(theme::medium(10.5))
                        .color(if selected { t.accent_text } else { t.text_muted }),
                )
                .fill(if selected { t.accent_soft } else { Color32::TRANSPARENT })
                .stroke(if selected { Stroke::new(1.0, t.accent.gamma_multiply(0.4)) } else { Stroke::NONE })
                .corner_radius(CornerRadius::same(6)),
            );
            if btn.clicked() {
                state.summary_length = len_variant;
            }
        }
    });
}

/// Paint a single message card with formatted text, citations, and actions.
fn render_message_card(
    ui: &mut egui::Ui,
    t: &Tokens,
    msg: &AiMessage,
    action_out: &mut Option<AiPanelAction>,
) {
    match msg.role {
        AiRole::User => {
            ui.horizontal(|ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let frame = egui::Frame::new()
                        .fill(t.selected)
                        .stroke(Stroke::new(1.0, t.accent.gamma_multiply(0.4)))
                        .corner_radius(CornerRadius::same(12))
                        .inner_margin(egui::Margin::symmetric(12, 8));

                    frame.show(ui, |ui| {
                        ui.label(egui::RichText::new(&msg.content).font(theme::medium(12.5)).color(t.text));
                    });
                });
            });
        }
        AiRole::Assistant => {
            let frame = egui::Frame::new()
                .fill(t.card)
                .stroke(Stroke::new(1.0, t.border))
                .corner_radius(CornerRadius::same(8))
                .inner_margin(egui::Margin::symmetric(12, 10));

            frame.show(ui, |ui| {
                // Header row
                ui.horizontal(|ui| {
                    icons::paint(ui, Rect::from_min_size(ui.cursor().min, vec2(16.0, 16.0)), "wand-sparkles", 14.0, t.accent);
                    ui.add_space(18.0);
                    ui.label(egui::RichText::new("PdfCraft Assistant").font(theme::semibold(12.0)).color(t.accent_text));

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if icons::button(ui, "copy", 22.0, false, "Copy full response to clipboard").clicked() {
                            let mut full_copy = msg.content.clone();
                            if !msg.key_points.is_empty() {
                                full_copy.push_str("\n\nKey Insights:\n");
                                for (page, pt) in &msg.key_points {
                                    full_copy.push_str(&format!("• {pt} [Page {page}]\n"));
                                }
                            }
                            ui.ctx().copy_text(full_copy);
                        }
                    });
                });

                ui.add_space(4.0);

                // Body content
                widgets::wrapped_text(ui, &msg.content, theme::regular(13.0), t.text, false);

                // Render key points if present
                if !msg.key_points.is_empty() {
                    ui.add_space(6.0);
                    for (page, point_text) in &msg.key_points {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("•").font(theme::semibold(14.0)).color(t.accent));
                            widgets::wrapped_text(ui, point_text, theme::regular(12.5), t.text, false);
                            let citation_btn = ui.add(
                                egui::Button::new(
                                    egui::RichText::new(format!("[p.{page}]"))
                                        .font(theme::medium(11.0))
                                        .color(t.accent_text),
                                )
                                .fill(t.accent_soft)
                                .stroke(Stroke::new(1.0, t.accent.gamma_multiply(0.25)))
                                .corner_radius(CornerRadius::same(4)),
                            );
                            if citation_btn.on_hover_text(format!("Jump to page {page}")).clicked() {
                                *action_out = Some(AiPanelAction::GoToPage(page.saturating_sub(1)));
                            }
                        });
                        ui.add_space(2.0);
                    }
                }

                // Render citation badges
                if !msg.citations.is_empty() {
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(egui::RichText::new("Citations:").font(theme::medium(11.5)).color(t.text_muted));
                        for c in &msg.citations {
                            let label = if let Some(score) = c.score {
                                format!("Page {} ({:.0}%)", c.page, score * 100.0)
                            } else {
                                format!("Page {}", c.page)
                            };

                            let btn = ui.add(
                                egui::Button::new(egui::RichText::new(label).font(theme::medium(11.0)).color(t.accent_text))
                                    .fill(t.accent_soft)
                                    .stroke(Stroke::new(1.0, t.accent.gamma_multiply(0.3)))
                                    .corner_radius(CornerRadius::same(10)),
                            );

                            if btn.on_hover_text(format!("Jump to page {}", c.page)).clicked() {
                                *action_out = Some(AiPanelAction::GoToPage(c.page.saturating_sub(1)));
                            }
                        }
                    });
                }

                // Render PII findings
                if !msg.findings.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);

                    // Batch redact button at the top
                    let all_quads: Vec<(usize, Vec<[f64; 8]>, String, String)> = msg
                        .findings
                        .iter()
                        .filter(|f| !f.quads.is_empty())
                        .map(|f| (f.page, f.quads.clone(), f.category.to_uppercase(), f.reason.to_string()))
                        .collect();

                    if !all_quads.is_empty() {
                        let btn_label = format!("🛡️ Redact All ({}) Matches", all_quads.len());
                        if ui.button(egui::RichText::new(btn_label).font(theme::semibold(12.0)).color(Color32::WHITE))
                            .on_hover_text("Apply black redaction marks over every detected sensitive entity across the document")
                            .clicked()
                        {
                            *action_out = Some(AiPanelAction::ApplyAllRedactions(all_quads));
                        }
                        ui.add_space(6.0);
                    }

                    for f in &msg.findings {
                        let row_frame = egui::Frame::new()
                            .fill(t.field)
                            .stroke(Stroke::new(1.0, t.border))
                            .corner_radius(CornerRadius::same(6))
                            .inner_margin(egui::Margin::symmetric(8, 6));

                        row_frame.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                // Category badge
                                let cat_bg = Color32::from_rgb(0xFE, 0xF2, 0xF2);
                                let cat_fg = Color32::from_rgb(0xDC, 0x26, 0x26);
                                ui.painter().rect_filled(
                                    Rect::from_min_size(ui.cursor().min, vec2(56.0, 18.0)),
                                    CornerRadius::same(3),
                                    if t.dark() { Color32::from_rgb(0x45, 0x1A, 0x1A) } else { cat_bg },
                                );
                                ui.allocate_ui_with_layout(vec2(56.0, 18.0), Layout::centered_and_justified(egui::Direction::LeftToRight), |ui| {
                                    ui.label(egui::RichText::new(f.category.to_uppercase()).font(theme::semibold(9.5)).color(cat_fg));
                                });

                                // Snippet text
                                ui.label(egui::RichText::new(&f.text).font(theme::medium(12.0)).color(t.text));

                                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                    // 1-Click Redact mark
                                    let redact_btn = ui.add(
                                        egui::Button::new(egui::RichText::new("Redact").font(theme::semibold(10.5)).color(cat_fg))
                                            .fill(if t.dark() { Color32::from_rgb(0x45, 0x1A, 0x1A) } else { cat_bg })
                                            .stroke(Stroke::new(1.0, cat_fg.gamma_multiply(0.4)))
                                            .corner_radius(CornerRadius::same(4)),
                                    );
                                    if redact_btn.on_hover_text("Place a permanent redaction mark over this text").clicked() {
                                        *action_out = Some(AiPanelAction::ApplyRedaction {
                                            page: f.page,
                                            quads: f.quads.clone(),
                                            overlay: f.category.to_uppercase(),
                                            reason: f.reason.to_string(),
                                        });
                                    }

                                    // View on page button
                                    let page_label = format!("p.{}", f.page + 1);
                                    let view_btn = ui.add(
                                        egui::Button::new(egui::RichText::new(page_label).font(theme::medium(10.5)).color(t.text_muted))
                                            .fill(t.hover)
                                            .corner_radius(CornerRadius::same(4)),
                                    );
                                    if view_btn.on_hover_text("Jump to and highlight this occurrence on page").clicked() {
                                        *action_out = Some(AiPanelAction::FlashRect(f.page, f.rect));
                                    }
                                });
                            });
                        });
                        ui.add_space(3.0);
                    }
                }

                // Render Alt-Text suggestions
                if !msg.alt_suggestions.is_empty() {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(4.0);

                    for s in &msg.alt_suggestions {
                        let row_frame = egui::Frame::new()
                            .fill(t.field)
                            .stroke(Stroke::new(1.0, t.border))
                            .corner_radius(CornerRadius::same(6))
                            .inner_margin(egui::Margin::symmetric(8, 6));

                        row_frame.show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(
                                        egui::RichText::new(format!("Figure on page {}", s.page + 1))
                                            .font(theme::medium(11.5))
                                            .color(t.text_faint),
                                    );
                                    widgets::wrapped_text(ui, &s.suggested_alt_text, theme::regular(12.5), t.text, false);
                                });

                                if let Some(fig_num) = s.figure {
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        if ui.button(egui::RichText::new("Apply Alt-Text").font(theme::medium(11.0))).on_hover_text("Save this caption into the PDF structure tree").clicked() {
                                            *action_out = Some(AiPanelAction::ApplyAltText {
                                                figure: fig_num,
                                                alt: s.suggested_alt_text.clone(),
                                            });
                                        }
                                    });
                                }
                            });
                        });
                        ui.add_space(3.0);
                    }
                }
            });
        }
    }
}

/// Paint the settings drawer for endpoint configuration.
fn render_settings_drawer(ui: &mut egui::Ui, t: &Tokens, state: &mut AiState) {
    let frame = egui::Frame::new()
        .fill(t.field)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(egui::Margin::symmetric(10, 8));

    frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Execution Mode:").font(theme::semibold(12.0)).color(t.text));
            if ui.selectable_label(state.provider_mode == ProviderMode::Builtin, "Offline Sovereign").clicked() {
                state.provider_mode = ProviderMode::Builtin;
            }
            if ui.selectable_label(state.provider_mode == ProviderMode::Endpoint, "Local Endpoint").clicked() {
                state.provider_mode = ProviderMode::Endpoint;
            }
        });

        if state.provider_mode == ProviderMode::Endpoint {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("URL:").font(theme::medium(11.5)).color(t.text_muted));
                ui.add(egui::TextEdit::singleline(&mut state.endpoint_url).hint_text("http://localhost:11434").desired_width(160.0));
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Model:").font(theme::medium(11.5)).color(t.text_muted));
                ui.add(egui::TextEdit::singleline(&mut state.model_name).hint_text("llama3.2").desired_width(160.0));
            });
        }
    });
    ui.add_space(6.0);
}

/// The main content of the AI assistant panel.
pub fn panel_content(
    ui: &mut egui::Ui,
    t: &Tokens,
    state: &mut AiState,
    doc: &pdfcraft_engine::Document,
    view: &mut DocView,
) -> Option<AiPanelAction> {
    widgets::begin_dialog(ui);
    state.sync_doc(doc, view);

    let mut action_out = None;

    // Header strip controls: settings, clear history, close
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Offline Sovereign AI").font(theme::semibold(14.0)).color(t.text));

        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if icons::button(ui, "trash-2", 24.0, false, "Clear conversation history").clicked() {
                state.history.clear();
            }
            if icons::button(ui, "settings-2", 24.0, state.show_settings, "AI Provider Configuration").clicked() {
                state.show_settings = !state.show_settings;
            }
        });
    });

    ui.add_space(6.0);

    // Settings drawer if open
    if state.show_settings {
        render_settings_drawer(ui, t, state);
    }

    // Telemetry Hero Card
    if let Some(telemetry) = &state.telemetry {
        render_telemetry_card(ui, t, telemetry);
        ui.add_space(6.0);
    }

    // Quick Prompts Row
    render_quick_prompts(ui, t, state, doc, view);
    ui.add_space(8.0);

    // Scrollable Chat Message Stream
    let available_height = (ui.available_height() - 44.0).max(120.0);
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .max_height(available_height)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 8.0;

            if state.history.is_empty() {
                let empty_frame = egui::Frame::new()
                    .fill(t.card)
                    .stroke(Stroke::new(1.0, t.border))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(egui::Margin::symmetric(14, 16));

                empty_frame.show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        icons::paint(ui, Rect::from_center_size(ui.cursor().center(), vec2(32.0, 32.0)), "wand-sparkles", 28.0, t.accent);
                        ui.add_space(36.0);
                        ui.label(egui::RichText::new("How can I assist you today?").font(theme::semibold(13.5)).color(t.text));
                        ui.add_space(2.0);
                        ui.label(
                            egui::RichText::new("Ask any question or click a Quick Action above to summarize, audit sensitive data, or inspect document structure.")
                                .font(theme::regular(12.0))
                                .color(t.text_muted),
                        );
                    });
                });
            } else {
                for msg in &state.history {
                    render_message_card(ui, t, msg, &mut action_out);
                }
            }
        });

    ui.add_space(6.0);

    // Pill Input Bar at the Bottom
    let input_frame = egui::Frame::new()
        .fill(t.field)
        .stroke(Stroke::new(1.0, t.border))
        .corner_radius(CornerRadius::same(16))
        .inner_margin(egui::Margin::symmetric(10, 4));

    input_frame.show(ui, |ui| {
        ui.horizontal(|ui| {
            let edit_width = (ui.available_width() - 32.0).max(40.0);
            let edit_resp = ui.add(
                egui::TextEdit::singleline(&mut state.input)
                    .hint_text("Ask anything about this PDF…")
                    .desired_width(edit_width)
                    .frame(egui::Frame::NONE),
            );

            let enter_pressed = edit_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let send_clicked = icons::button(ui, "send", 26.0, false, "Send question").clicked();

            if (enter_pressed || send_clicked) && !state.input.trim().is_empty() {
                let q = std::mem::take(&mut state.input);
                state.trigger_question(&q, doc, view);
            }
        });
    });

    action_out
}
