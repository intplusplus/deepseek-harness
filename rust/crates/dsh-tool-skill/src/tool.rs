//! The model-facing `skill` loader tool.
//!
//! Transcribed from `packages/skill/tool-skill/src/index.ts`. The TypeScript
//! `execute` resolves the layered registry exactly as the calling agent's
//! composition sees it; the Rust [`ToolHandler`] is object-safe and
//! `Send + Sync`, while `dsh-skill`'s host is `Rc`-based and single-threaded,
//! so the tool takes an injected [`SkillCatalog`] seam instead of a host. A
//! deployment bridges its live registry to the seam on its own thread.

use std::sync::Arc;

use dsh_llm::{ContentBlock, ToolSchema};
use dsh_skill::{is_skill_name, render_skill_content};
use dsh_tools::{
    argument, CancelSignal, ToolCallInput, ToolExecutionMode, ToolExecutionResult, ToolHandler,
};
use serde_json::{json, Value};
use thiserror::Error;

/// The session-local tool name the model calls.
pub const SKILL_TOOL_NAME: &str = "skill";

/// Errors the loader surfaces as tool failures.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SkillToolError {
    /// The argument is not a valid public skill name.
    #[error("invalid skill name \"{0}\"")]
    InvalidName(String),
    /// The name resolves to nothing the agent can currently see.
    #[error("skill \"{0}\" is unknown or no longer available")]
    Unknown(String),
    /// The skill resolved but disables model invocation.
    #[error("skill \"{0}\" is not available for model invocation")]
    NotModelInvocable(String),
    /// The backing catalog lookup failed.
    #[error("skill lookup failed: {0}")]
    Lookup(String),
}

/// The catalog seam the tool resolves names through.
pub trait SkillCatalog: Send + Sync {
    /// Load one skill definition, or `None` when the name is unavailable.
    ///
    /// # Errors
    /// [`SkillToolError::Lookup`] when the backing registry fails.
    fn get(&self, name: &str) -> Result<Option<dsh_skill::SkillDefinition>, SkillToolError>;
}

/// A catalog seam backed by a closure.
pub struct FnSkillCatalog<F> {
    get: F,
}

impl<F> SkillCatalog for FnSkillCatalog<F>
where
    F: Fn(&str) -> Result<Option<dsh_skill::SkillDefinition>, SkillToolError>
        + Send
        + Sync
        + 'static,
{
    fn get(&self, name: &str) -> Result<Option<dsh_skill::SkillDefinition>, SkillToolError> {
        (self.get)(name)
    }
}

/// Build a catalog seam from one closure.
pub fn fn_skill_catalog<F>(get: F) -> FnSkillCatalog<F>
where
    F: Fn(&str) -> Result<Option<dsh_skill::SkillDefinition>, SkillToolError>
        + Send
        + Sync
        + 'static,
{
    FnSkillCatalog { get }
}

/// The `skill` tool: load the full instructions for a named skill.
pub struct SkillTool {
    catalog: Arc<dyn SkillCatalog>,
}

impl SkillTool {
    /// Mount the loader over a catalog seam.
    pub fn new(catalog: Arc<dyn SkillCatalog>) -> Self {
        Self { catalog }
    }

    /// The model-facing schema.
    pub fn tool_schema(&self) -> ToolSchema {
        ToolSchema::new(
            SKILL_TOOL_NAME,
            "Load the full instructions for a skill. Call it before acting on a task that names or clearly matches a skill in the session skill catalog.",
            json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "required": true,
                        "description": "The exact skill name from the available skills list.",
                    },
                },
                "required": ["name"],
            }),
        )
    }
}

impl ToolHandler for SkillTool {
    fn name(&self) -> &str {
        SKILL_TOOL_NAME
    }

    fn schema(&self) -> ToolSchema {
        self.tool_schema()
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    fn call(&self, input: &ToolCallInput, _cancel: &CancelSignal) -> ToolExecutionResult {
        let name = argument(input, "name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !is_skill_name(name) {
            return SkillToolError::InvalidName(name.to_string()).into_result();
        }
        let skill = match self.catalog.get(name) {
            Ok(skill) => skill,
            Err(error) => return error.into_result(),
        };
        let Some(skill) = skill else {
            return SkillToolError::Unknown(name.to_string()).into_result();
        };
        if !skill.summary.invocation.model_invocable {
            return SkillToolError::NotModelInvocable(name.to_string()).into_result();
        }
        let rendered = render_skill_content(&skill);
        let mut value = json!({
            "name": skill.summary.name.as_str(),
            "provider": skill.summary.provider,
            "content": skill.content,
        });
        if let Some(base) = &skill.summary.resource_base {
            if let Ok(base) = serde_json::to_value(base) {
                value["resourceBase"] = base;
            }
        }
        ToolExecutionResult::success(vec![ContentBlock::text(rendered)]).with_value(value)
    }
}

/// Render a tool failure the model sees.
trait IntoToolResult {
    fn into_result(self) -> ToolExecutionResult;
}

impl IntoToolResult for SkillToolError {
    fn into_result(self) -> ToolExecutionResult {
        ToolExecutionResult::failure(self.to_string())
    }
}
