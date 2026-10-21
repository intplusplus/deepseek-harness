//! Host-side binding functions and the namespace table handed to evaluations.
//!
//! Ported from the seam in `packages/ptc-runtime/ptc-runtime/src/types.ts`
//! (`PtcBindingNamespace`): the program sees only namespace and member names —
//! the functions themselves stay host-side and are reached through `call`
//! frames. Declarations sent to the sandbox are derived from the function map,
//! then validated by `dsh-ptc`'s reserved-name rules exactly as the TypeScript
//! provider validates them.
//!
//! Unlike the TypeScript provider this host stores functions in a
//! `BTreeMap`, so the declaration's `names` array is alphabetical rather
//! than insertion-ordered. Field order inside `names` carries no protocol
//! meaning — the program resolves members by name.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use dsh_ptc::reserved::{self, BindingErrorClass};
use serde_json::Value;

/// One host function a program may call through the bridge.
///
/// `args` arrives decoded from the flat lossless-JSON wire; the resolution
/// value is encoded back onto the same wire. A rejected call surfaces inside
/// the program as a rejection carrying `Err`'s message.
#[async_trait]
pub trait PtcBinding: Send + Sync {
    /// Resolve one bridged call.
    async fn call(&self, args: Value) -> Result<Value, String>;
}

/// A shared host function object.
pub type BindingFn = Arc<dyn PtcBinding>;

/// One namespace of host functions, as the host holds it.
#[derive(Clone)]
pub struct HostNamespace {
    /// The global identifier the program sees.
    pub global: String,
    /// The callable members, keyed by the exact name the program calls.
    pub functions: HashMap<String, BindingFn>,
    /// Optional program-visible typed rejection contract.
    pub error_class: Option<BindingErrorClass>,
}

impl HostNamespace {
    /// An empty namespace under `global`.
    pub fn new(global: impl Into<String>) -> Self {
        Self {
            global: global.into(),
            functions: HashMap::new(),
            error_class: None,
        }
    }

    /// Add one callable member.
    pub fn with_fn(mut self, name: impl Into<String>, function: BindingFn) -> Self {
        self.functions.insert(name.into(), function);
        self
    }

    /// Attach the typed rejection contract.
    pub fn with_error_class(mut self, error_class: BindingErrorClass) -> Self {
        self.error_class = Some(error_class);
        self
    }

    /// The declaration sent to the sandbox for this namespace.
    pub fn declaration(&self) -> reserved::BindingNamespace {
        reserved::BindingNamespace {
            global: self.global.clone(),
            functions: self.functions.keys().cloned().collect(),
            error_class: self.error_class.clone(),
        }
    }
}

/// The host-side lookup table for one evaluation, keyed by namespace global.
pub(crate) type NamespaceTable = BTreeMap<String, HostNamespace>;

/// Build the lookup table and the sandbox declarations from one request's
/// namespaces.
pub(crate) fn build_table(
    bindings: Vec<HostNamespace>,
) -> (NamespaceTable, Vec<reserved::BindingNamespace>) {
    let mut table = NamespaceTable::new();
    let mut declarations = Vec::with_capacity(bindings.len());
    for namespace in bindings {
        declarations.push(namespace.declaration());
        table.insert(namespace.global.clone(), namespace);
    }
    (table, declarations)
}

/// A [`PtcBinding`] built from a closure returning a future.
pub struct FnBinding<F, Fut> {
    function: F,
    _future: std::marker::PhantomData<fn() -> Fut>,
}

impl<F, Fut> FnBinding<F, Fut>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Value, String>> + Send + 'static,
{
    /// Wrap one closure into a shared binding object.
    pub fn shared(function: F) -> Arc<Self> {
        Arc::new(Self {
            function,
            _future: std::marker::PhantomData,
        })
    }
}

#[async_trait]
impl<F, Fut> PtcBinding for FnBinding<F, Fut>
where
    F: Fn(Value) -> Fut + Send + Sync,
    Fut: Future<Output = Result<Value, String>> + Send + 'static,
{
    async fn call(&self, args: Value) -> Result<Value, String> {
        (self.function)(args).await
    }
}
