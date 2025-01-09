use std::path::Path;

use serde::Deserialize;
use zed_extension_api::{
    self as zed, serde_json, settings::ContextServerSettings, ContextServerId,
};

#[derive(Debug, Deserialize)]
struct PdfSearchContextServerSettings {
    pdf_path: String,
    extension_path: String,
    openai_api_key: String,
}

struct MyExtension {}

impl zed::Extension for MyExtension {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self {}
    }

    fn context_server_command(
        &mut self,
        context_server_id: &ContextServerId,
        project: &zed::Project,
    ) -> zed::Result<zed::Command> {
        let settings = ContextServerSettings::for_project("pdfsearch-context-server", project)?;
        let Some(settings) = settings.settings else {
            return Err("Missing `context_servers` settings for `pdfsearch-context-server`".into());
        };
        let settings: PdfSearchContextServerSettings =
            serde_json::from_value(settings).map_err(|e| e.to_string())?;

        let mcp_python_module = String::from("pdf_rag");
        let extension_path = Path::new(settings.extension_path.as_str());
        let mcp_server_path = extension_path.join(&mcp_python_module);

        Ok(zed::Command {
            command: "uv".to_string(),
            args: vec![
                format!("--directory={}", mcp_server_path.to_string_lossy()),
                "run".to_string(),
                mcp_python_module,
            ],
            env: vec![
                ("PDF_PATH".into(), settings.pdf_path),
                ("OPENAI_API_KEY".into(), settings.openai_api_key),
            ],
        })
    }
}

zed::register_extension!(MyExtension);
