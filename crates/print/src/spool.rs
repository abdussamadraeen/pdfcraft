//! System print spooler supplied by craft-print. PDF composition and permission checks
//! stay in PdfCraft; these adapters retain the application's print-error surface.

pub use craft_print::{
    Duplex, HANDLED_OPTIONS, HAS_PRINTER_PREFERENCES, Job, Printer, PrinterOption, Scaling, lp_args, lpstat_command, parse_lpoptions, parse_lpstat,
    parse_lpstat_e, parse_ppd, parse_windows_printers, ppd_text, printers, printers_parsed,
};

/// The driver's job options for a printer (CUPS: its PPD, with the queue's current defaults);
/// empty when it has none or on other platforms. Reads files and runs `lpoptions`, so call it
/// when the user asks, not every frame.
pub fn printer_options(printer: &str) -> Vec<PrinterOption> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        printer_options_via(printer, std::path::Path::new("/etc/cups/ppd"), std::process::Command::new("lpoptions"))
    }
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        craft_print::printer_options(printer)
    }
}

/// Read a destination's options with the PPD directory and command supplied, so tests need
/// neither a system queue nor a physical printer. CUPS instances share their queue's PPD but
/// have their own defaults, queried with the complete `queue/instance` destination.
#[cfg(all(unix, not(target_arch = "wasm32")))]
pub(crate) fn printer_options_via(printer: &str, ppd_dir: &std::path::Path, mut lpoptions: std::process::Command) -> Vec<PrinterOption> {
    let (queue, instance) = printer.split_once('/').map_or((printer, None), |(q, i)| (q, Some(i)));
    // Retain the conservative name checks, applied to each component. Neither component is
    // a path: reject dot components, backslashes and any additional separator before I/O.
    let valid = |name: &str| {
        !name.is_empty()
            && name.len() <= 64
            && !matches!(name, "." | "..")
            && name.bytes().all(|b| b.is_ascii_graphic() && !matches!(b, b'/' | b'\\' | b':' | b'=' | b'"'))
    };
    if !valid(queue) || instance.is_some_and(|name| !valid(name)) {
        return Vec::new();
    }
    const PPD_CAP: u64 = 16 << 20;
    let ppd = std::fs::File::open(ppd_dir.join(format!("{queue}.ppd"))).ok().and_then(|f| {
        use std::io::Read as _;
        let mut bytes = Vec::new();
        f.take(PPD_CAP).read_to_end(&mut bytes).ok().map(|_| bytes)
    });
    let mut options = ppd.map(|b| parse_ppd(&ppd_text(&b))).unwrap_or_default();
    let current = lpoptions.args(["-p", printer, "-l"]).output().map(|o| parse_lpoptions(&String::from_utf8_lossy(&o.stdout))).unwrap_or_default();
    for (key, value) in current {
        if let Some(o) = options.iter_mut().find(|o| o.key == key)
            && o.choices.iter().any(|(c, _)| *c == value)
        {
            o.default = value;
        }
    }
    options
}

pub fn submit(pdf: &[u8], job: &Job) -> Result<String, crate::PrintError> {
    #[cfg(windows)]
    {
        match craft_print::submit(pdf, job) {
            Ok(msg) => Ok(msg),
            Err(e) => {
                let err_str = e.to_string();
                if is_windows_render_or_com_error(&err_str) {
                    log::warn!("craft_print failed with COM/render error ({err_str}); falling back to software raster printer...");
                    windows_software_raster_submit(pdf, job).map_err(|fb_err| {
                        crate::PrintError::Spool(format!("{err_str}\nSoftware raster fallback: {fb_err}"))
                    })
                } else {
                    Err(crate::PrintError::Spool(err_str))
                }
            }
        }
    }
    #[cfg(not(windows))]
    {
        craft_print::submit(pdf, job).map_err(|e| crate::PrintError::Spool(e.to_string()))
    }
}

#[cfg(windows)]
fn is_windows_render_or_com_error(err: &str) -> bool {
    let lower = err.to_lowercase();
    lower.contains("0x80004002")
        || lower.contains("e_nointerface")
        || lower.contains("schnittstelle")
        || lower.contains("interface not supported")
        || lower.contains("render error")
        || lower.contains("dxgi")
        || lower.contains("windows.data.pdf")
        || lower.contains("pdf renderer")
}

#[cfg(windows)]
pub fn windows_software_raster_render(pdf: &[u8]) -> Result<usize, crate::PrintError> {
    let mut renderer = pdfcraft_render::PageRenderer::new(
        std::sync::Arc::new(pdf.to_vec()),
        pdfcraft_render::RenderConfig::default(),
    );
    let count = renderer.page_count();
    if count == 0 {
        return Err(crate::PrintError::Spool("the print-ready PDF has no pages".into()));
    }
    for i in 0..count {
        let rendered = renderer.render(pdfcraft_render::RenderRequest {
            page: i,
            kind: pdfcraft_render::RequestKind::Pixels,
            tile: None,
            scale: 2.0833333,
            tag: 0,
            preview: false,
        });
        if let Some(err) = rendered.error {
            return Err(crate::PrintError::Spool(format!("rasterizer error on page {i}: {err}")));
        }
    }
    Ok(count)
}

#[cfg(windows)]
pub(crate) fn windows_software_raster_submit(pdf: &[u8], job: &Job) -> Result<String, crate::PrintError> {
    if std::env::var("CRAFT_PRINT_DRYRUN").as_deref() == Ok("1") {
        let count = windows_software_raster_render(pdf)?;
        return Ok(format!("rendered {count} sheet(s) via software fallback"));
    }

    let temp_dir = tempfile::Builder::new()
        .prefix("pdfcraft-print-fallback-")
        .tempdir()
        .map_err(|e| crate::PrintError::Spool(format!("could not create temp directory: {e}")))?;

    let mut renderer = pdfcraft_render::PageRenderer::new(
        std::sync::Arc::new(pdf.to_vec()),
        pdfcraft_render::RenderConfig::default(),
    );
    let count = renderer.page_count();
    if count == 0 {
        return Err(crate::PrintError::Spool("the print-ready PDF has no pages".into()));
    }

    for i in 0..count {
        let rendered = renderer.render(pdfcraft_render::RenderRequest {
            page: i,
            kind: pdfcraft_render::RequestKind::Pixels,
            tile: None,
            scale: 2.0833333,
            tag: 0,
            preview: false,
        });
        if let Some(err) = rendered.error {
            return Err(crate::PrintError::Spool(format!("rasterizer error on page {i}: {err}")));
        }
        let page_path = temp_dir.path().join(format!("sheet_{:04}.png", i));
        encode_png_file(&page_path, rendered.width, rendered.height, &rendered.rgba)?;
    }

    let script = r#"
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$printer = New-Object System.Drawing.Printing.PrintDocument
$printer.DocumentName = $env:CRAFT_PRINT_TITLE
if ($env:CRAFT_PRINT_PRINTER) {
    $printer.PrinterSettings.PrinterName = $env:CRAFT_PRINT_PRINTER
}
if (-not $printer.PrinterSettings.IsValid) {
    throw ('the printer is not available: ' + $env:CRAFT_PRINT_PRINTER)
}
$printer.PrinterSettings.Copies = [Math]::Min(32767, [Math]::Max(1, [int]$env:CRAFT_PRINT_COPIES))
$printer.PrinterSettings.Collate = ($env:CRAFT_PRINT_COLLATE -eq '1')

$edge = $env:CRAFT_PRINT_DUPLEX
if ($edge -eq 'long-edge') {
    $printer.PrinterSettings.Duplex = [System.Drawing.Printing.Duplex]::Vertical
} elseif ($edge -eq 'short-edge') {
    $printer.PrinterSettings.Duplex = [System.Drawing.Printing.Duplex]::Horizontal
} elseif ($edge -eq 'off') {
    $printer.PrinterSettings.Duplex = [System.Drawing.Printing.Duplex]::Simplex
}

if ($env:CRAFT_PRINT_GRAYSCALE -eq '1') {
    $printer.DefaultPageSettings.Color = $false
}

$files = @(Get-ChildItem -Path $env:CRAFT_PRINT_DIR -Filter "*.png" | Sort-Object Name)
if ($files.Count -eq 0) {
    throw 'no rendered sheets found to print'
}

$script:sheetIndex = 0
$printer.OriginAtMargins = $false
$printer.PrintController = New-Object System.Drawing.Printing.StandardPrintController

$printer.add_PrintPage({
    param($sender, $e)
    if ($script:sheetIndex -lt $files.Count) {
        $filePath = $files[$script:sheetIndex].FullName
        $img = [System.Drawing.Image]::FromFile($filePath)
        try {
            $e.Graphics.DrawImage($img, $e.PageBounds)
        } finally {
            $img.Dispose()
        }
        $script:sheetIndex++
        $e.HasMorePages = ($script:sheetIndex -lt $files.Count)
    } else {
        $e.HasMorePages = $false
    }
})

$printer.Print()
Write-Output ('rendered and sent ' + $files.Count + ' sheet(s) via software fallback')
"#;

    let mut cmd = std::process::Command::new("powershell.exe");
    cmd.args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]);
    cmd.env("CRAFT_PRINT_DIR", temp_dir.path());
    cmd.env("CRAFT_PRINT_PRINTER", job.printer.as_deref().unwrap_or_default());
    cmd.env("CRAFT_PRINT_TITLE", &job.title);
    cmd.env("CRAFT_PRINT_COPIES", job.copies.to_string());
    cmd.env("CRAFT_PRINT_COLLATE", if job.collate { "1" } else { "0" });
    cmd.env(
        "CRAFT_PRINT_DUPLEX",
        match job.duplex {
            None => "",
            Some(Duplex::Off) => "off",
            Some(Duplex::LongEdge) => "long-edge",
            Some(Duplex::ShortEdge) => "short-edge",
        },
    );
    cmd.env("CRAFT_PRINT_GRAYSCALE", if job.grayscale { "1" } else { "0" });

    let out = cmd.output().map_err(|e| crate::PrintError::Spool(format!("could not launch print process: {e}")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let err = if err.is_empty() { String::from_utf8_lossy(&out.stdout).trim().to_string() } else { err };
        return Err(crate::PrintError::Spool(if err.is_empty() { "the print job was refused".into() } else { err }));
    }

    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[cfg(windows)]
fn encode_png_file(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), crate::PrintError> {
    use std::fs::File;
    use std::io::BufWriter;
    let file = File::create(path).map_err(|e| crate::PrintError::Spool(format!("could not create raster file: {e}")))?;
    let mut writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(&mut writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut header_writer = encoder.write_header().map_err(|e| crate::PrintError::Spool(format!("png header error: {e}")))?;
    header_writer.write_image_data(rgba).map_err(|e| crate::PrintError::Spool(format!("png write error: {e}")))?;
    Ok(())
}

/// [`submit`], told `(sent, sheets)` as each sheet reaches the spooler (Windows; CUPS takes the
/// whole PDF at once and reports nothing).
pub fn submit_with_progress(pdf: &[u8], job: &Job, progress: &mut dyn FnMut(usize, usize)) -> Result<String, crate::PrintError> {
    craft_print::submit_with_progress(pdf, job, progress).map_err(|e| crate::PrintError::Spool(e.to_string()))
}

pub fn open_printer_preferences(printer: &str) -> Result<(), crate::PrintError> {
    craft_print::open_printer_preferences(printer).map_err(|e| crate::PrintError::Spool(e.to_string()))
}
