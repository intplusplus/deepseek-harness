//! The plugin primitive: named units of registration mounted into scopes whose
//! effects unwind on unload.

use std::{error::Error, sync::Arc};

use super::{context::Context, event::EventError};

/// A plugin: a named unit of registrations mounted into a [`Scope`].
pub trait Plugin: Send + Sync {
    /// Stable plugin name, used for diagnostics and unload.
    fn name(&self) -> &str;

    /// Mount the plugin.
    ///
    /// Every registration made through `scope` is a reversible effect. If
    /// `load` returns an error, the scope drops and every effect made so far
    /// unwinds — the rollback of failed initialization.
    fn load(&self, scope: &mut Scope) -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// Errors raised by the plugin host.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("plugin `{name}` failed to load: {message}")]
    Load { name: String, message: String },
}

/// One reversible effect held by a scope.
enum Effect {
    Service(super::context::ServiceGuard),
    Observer(Box<dyn std::any::Any + Send>),
    Waterfall(Box<dyn std::any::Any + Send>),
    Deferred(Box<dyn FnOnce() + Send + 'static>),
}

/// A load scope.
///
/// Registrations made through a scope unwind when it drops, in reverse
/// registration order (Cordis effect semantics).
pub struct Scope {
    context: Arc<Context>,
    effects: Vec<Effect>,
}

impl Scope {
    /// Open a scope over the shared context.
    pub fn new(context: Arc<Context>) -> Self {
        Self {
            context,
            effects: Vec::new(),
        }
    }

    /// The shared context this scope registers into.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Register a service provider as an effect of this scope.
    pub fn set<T: super::context::Service>(&mut self, value: Arc<T>) {
        let guard = self.context.insert(value);
        self.effects.push(Effect::Service(guard));
    }

    /// Register an observer as an effect of this scope.
    pub fn observe<E, F>(&mut self, handler: F)
    where
        E: super::event::Event,
        F: Fn(&E) + Send + Sync + 'static,
    {
        let guard = self.context.observe(handler);
        self.effects.push(Effect::Observer(Box::new(guard)));
    }

    /// Register a waterfall listener as an effect of this scope.
    pub fn waterfall<E, F>(&mut self, handler: F)
    where
        E: super::event::Event,
        F: Fn(E, &mut dyn FnMut(E) -> Result<E, EventError>) -> Result<E, EventError>
            + Send
            + Sync
            + 'static,
    {
        let guard = self.context.waterfall(handler);
        self.effects.push(Effect::Waterfall(Box::new(guard)));
    }

    /// Register an arbitrary reversible effect: the closure runs when the scope
    /// unwinds, which is how a plugin undoes anything its registration cannot.
    pub fn defer(&mut self, effect: impl FnOnce() + Send + 'static) {
        self.effects.push(Effect::Deferred(Box::new(effect)));
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        while let Some(effect) = self.effects.pop() {
            match effect {
                Effect::Service(guard) => drop(guard),
                Effect::Observer(guard) => drop(guard),
                Effect::Waterfall(guard) => drop(guard),
                Effect::Deferred(effect) => effect(),
            }
        }
    }
}

/// One loaded plugin and the scope its registrations live in.
struct LoadedPlugin {
    name: String,
    /// Held for its `Drop`: unwinding this scope unregisters everything the
    /// plugin contributed.
    #[allow(dead_code)]
    scope: Scope,
}

/// Owns the shared context and the scope of every loaded plugin.
///
/// Loading a plugin beside the others is how the product is composed; unloading
/// drops its scope and unwinds its effects. A failed load leaves no trace:
/// the partially built scope drops before the host records anything.
pub struct PluginHost {
    context: Arc<Context>,
    loaded: Vec<LoadedPlugin>,
}

impl PluginHost {
    pub fn new() -> Self {
        Self {
            context: Arc::new(Context::new()),
            loaded: Vec::new(),
        }
    }

    /// The shared context every loaded plugin registers into.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Mount a plugin beside the others.
    pub fn load(&mut self, plugin: &dyn Plugin) -> Result<(), PluginError> {
        let name = plugin.name().to_string();
        let mut scope = Scope::new(Arc::clone(&self.context));
        if let Err(error) = plugin.load(&mut scope) {
            // `scope` drops here: every effect it registered unwinds.
            return Err(PluginError::Load {
                name,
                message: error.to_string(),
            });
        }
        self.loaded.push(LoadedPlugin { name, scope });
        Ok(())
    }

    /// Unload a plugin by name; its scope drops and its effects unwind.
    pub fn unload(&mut self, name: &str) -> bool {
        match self.loaded.iter().position(|plugin| plugin.name == name) {
            Some(index) => {
                self.loaded.remove(index);
                true
            }
            None => false,
        }
    }

    /// Whether a plugin with this name is currently loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        self.loaded.iter().any(|plugin| plugin.name == name)
    }

    /// The names of the currently loaded plugins, in load order.
    pub fn loaded_names(&self) -> Vec<&str> {
        self.loaded
            .iter()
            .map(|plugin| plugin.name.as_str())
            .collect()
    }
}

impl Default for PluginHost {
    fn default() -> Self {
        Self::new()
    }
}
