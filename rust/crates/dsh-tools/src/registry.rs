//! The tool registry: name-keyed handlers and their declared schemas.

#![forbid(unsafe_code)]

use std::{collections::BTreeMap, sync::Arc};

use dsh_llm::ToolSchema;

use super::{cancel::CancelSignal, types::*};

/// One registered tool.
///
/// The handler owns its schema, its concurrency mode, and its body. It is
/// object-safe so the registry can hold heterogeneous tools, and `Send + Sync`
/// so a plugin can contribute it as a context service.
pub trait ToolHandler: Send + Sync {
    /// The session-local tool name the model calls.
    fn name(&self) -> &str;

    /// The JSON schema sent to the model.
    fn schema(&self) -> ToolSchema;

    /// Whether this tool may overlap sibling calls. Defaults to parallel, the
    /// safe default: exclusive is the restrictive declaration.
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    /// Run the call to completion.
    ///
    /// A handler returns a result rather than an error: a thrown failure is a
    /// *result* the model must see, and dropping it would leave a `tool/call`
    /// without a durable pair.
    fn call(&self, input: &ToolCallInput, cancel: &CancelSignal) -> ToolExecutionResult;
}

/// The live set of callable tools.
///
/// Backed by a `BTreeMap` so `schemas()` is deterministic: the assembled tool
/// list lands in a `request/header` that is compared for equality across
/// replays, and a HashMap's arbitrary order would churn it.
#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn ToolHandler>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolRegistry")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ToolRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool under its declared name, returning the tool it replaced.
    pub fn register(&mut self, tool: Arc<dyn ToolHandler>) -> Option<Arc<dyn ToolHandler>> {
        self.tools.insert(tool.name().to_owned(), tool)
    }

    /// Remove a tool, returning it when present.
    pub fn unregister(&mut self, name: &str) -> Option<Arc<dyn ToolHandler>> {
        self.tools.remove(name)
    }

    /// The handler registered under `name`.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ToolHandler>> {
        self.tools.get(name)
    }

    /// Whether a tool is registered under `name`.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// The number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Every declared schema, ordered by tool name.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|tool| tool.schema()).collect()
    }

    /// The concurrency mode of the named tool, or `None` when unregistered.
    pub fn execution_mode(&self, name: &str) -> Option<ToolExecutionMode> {
        self.tools.get(name).map(|tool| tool.execution_mode())
    }
}
