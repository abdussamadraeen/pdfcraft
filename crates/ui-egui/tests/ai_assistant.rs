//! Integration tests for Sovereign Local AI Assistant in the desktop shell (`egui_kittest`).

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use pdfcraft_engine::Session;
use pdfcraft_ui_egui::{PdfCraftApp, RightPanel, theme::ThemePreference};

#[test]
fn ai_assistant_panel_opens_and_renders() {
    let s = Session::new();
    let sample_text = "Acme Corp Master Services Agreement. \
The contractor shall deliver quarterly security reports by October 15. \
Payments shall be remitted to invoices@acme-corp.com or contact 555-123-4567. \
The client SSN on file is 123-45-6789. \
All services are rendered strictly confidential under Section 9.";

    let bytes = s.create_from_text("agreement.pdf", sample_text).unwrap().to_vec();

    let mut h = Harness::builder().with_size(egui::vec2(1400.0, 900.0)).build_eframe(move |_cc| {
        let mut app = PdfCraftApp::new();
        app.set_option("language", "en").unwrap();
        app.open_bytes("agreement.pdf", None, bytes.clone()).unwrap();
        app
    });

    h.run_steps(4);

    // 1. Open AI panel via command
    assert!(h.state_mut().execute("ai.panel"));
    h.run_steps(3);
    assert_eq!(h.state().right, Some(RightPanel::Ai));

    // 2. Verify AI Assistant header & sovereign status are present
    h.get_by_label("Offline Sovereign AI");
    h.get_by_label("Local & Sovereign");

    // 3. Trigger summarize action
    h.get_by_label("✨ Summarize").click();
    h.run_steps(4);

    // Check that history was populated with user and assistant messages
    assert_eq!(h.state().ai.history.len(), 2);
    assert_eq!(h.state().ai.history[0].role, pdfcraft_ui_egui::ai_ui::AiRole::User);
    assert_eq!(h.state().ai.history[1].role, pdfcraft_ui_egui::ai_ui::AiRole::Assistant);
    assert!(!h.state().ai.history[1].content.is_empty());
    assert!(!h.state().ai.history[1].citations.is_empty());

    // 4. Trigger sensitive PII audit
    h.get_by_label("🛡️ Scan Sensitive PII").click();
    h.run_steps(4);

    // Verify findings were extracted
    assert_eq!(h.state().ai.history.len(), 4);
    let pii_msg = &h.state().ai.history[3];
    assert!(!pii_msg.findings.is_empty(), "SSN, phone, or email should be detected");

    // 5. Toggle between Dark Mode and Light Mode to verify theme safety
    h.state_mut().set_theme_preference(ThemePreference::Dark);
    h.run_steps(3);
    assert_eq!(h.state().theme, pdfcraft_ui_egui::theme::ThemeKind::Dark);

    h.state_mut().set_theme_preference(ThemePreference::Light);
    h.run_steps(3);
    assert_eq!(h.state().theme, pdfcraft_ui_egui::theme::ThemeKind::Light);
}
