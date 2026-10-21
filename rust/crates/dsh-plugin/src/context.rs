//! The typed service registry: a swappable capability with a service
//! definition (the [`Service`] marker), a provider (the registered value), and
//! consumers (`get` / `require`).

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, RwLock, Weak},
};

use super::event::{Event, EventBus, EventError, ObserverGuard, WaterfallGuard};

/// A service registered in the shared context.
///
/// The blanket implementation makes every `'static + Send + Sync` type a
/// service. The registry keys values by their concrete type, so `get` /
/// `require` recover the exact `Arc<T>` a provider registered.
pub trait Service: Any + Send + Sync {}

impl<T: Any + Send + Sync> Service for T {}

/// One registered service value.
struct ServiceSlot {
    value: Arc<dyn Any + Send + Sync>,
}

/// Errors raised by the service registry.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("required service `{0}` is not registered")]
    Missing(&'static str),
}

/// The registry behind `ctx.services`.
///
/// Registrations are reversible effects: dropping the returned
/// [`ServiceGuard`] restores the value the registration replaced, or clears
/// the slot when it was the first. Guards must unwind in reverse registration
/// order — [`super::plugin::Scope`] guarantees this by unwinding LIFO.
#[derive(Default)]
pub struct ServiceRegistry {
    slots: RwLock<HashMap<TypeId, ServiceSlot>>,
}

impl ServiceRegistry {
    /// Register `value` as the provider of `T`, replacing any current one.
    pub fn insert<T: Service>(self: &Arc<Self>, value: Arc<T>) -> ServiceGuard {
        let key = TypeId::of::<T>();
        let erased: Arc<dyn Any + Send + Sync> = value;
        let mut slots = self
            .slots
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let previous = slots
            .insert(key, ServiceSlot { value: erased })
            .map(|slot| slot.value);
        ServiceGuard {
            registry: Arc::downgrade(self),
            key,
            previous,
        }
    }

    /// The registered provider of `T`, if any.
    pub fn get<T: Service>(&self) -> Option<Arc<T>> {
        let slots = self
            .slots
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let slot = slots.get(&TypeId::of::<T>())?;
        Arc::clone(&slot.value).downcast::<T>().ok()
    }

    /// The registered provider of `T`, or an explicit failure.
    ///
    /// A host reader either requires this service during activation or fails
    /// explicitly; there is no silent default for a missing capability.
    pub fn require<T: Service>(&self) -> Result<Arc<T>, ContextError> {
        self.get::<T>()
            .ok_or(ContextError::Missing(std::any::type_name::<T>()))
    }
}

/// The reversible effect of one service registration.
///
/// On drop it restores the replaced provider, or removes the slot when this
/// registration was the first for its type.
pub struct ServiceGuard {
    registry: Weak<ServiceRegistry>,
    key: TypeId,
    previous: Option<Arc<dyn Any + Send + Sync>>,
}

impl Drop for ServiceGuard {
    fn drop(&mut self) {
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let mut slots = registry
            .slots
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.previous.take() {
            Some(previous) => {
                slots.insert(self.key, ServiceSlot { value: previous });
            }
            None => {
                slots.remove(&self.key);
            }
        }
    }
}

/// The shared context: the service registry plus the typed event bus.
pub struct Context {
    services: Arc<ServiceRegistry>,
    events: Arc<EventBus>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            services: Arc::new(ServiceRegistry::default()),
            events: Arc::new(EventBus::default()),
        }
    }

    /// The service registry half of the context.
    pub fn services(&self) -> &Arc<ServiceRegistry> {
        &self.services
    }

    /// The event bus half of the context.
    pub fn events(&self) -> &Arc<EventBus> {
        &self.events
    }

    /// Register a service provider; the returned guard is the reversible effect.
    pub fn insert<T: Service>(&self, value: Arc<T>) -> ServiceGuard {
        self.services.insert(value)
    }

    /// Read a service provider.
    pub fn get<T: Service>(&self) -> Option<Arc<T>> {
        self.services.get()
    }

    /// Read a service provider, failing explicitly when absent.
    pub fn require<T: Service>(&self) -> Result<Arc<T>, ContextError> {
        self.services.require()
    }

    /// Emit an observer event. Listener failures are contained.
    pub fn emit<E: Event>(&self, event: &E) {
        self.events.emit(event)
    }

    /// Dispatch a waterfall event; the final payload is returned.
    pub fn dispatch<E: Event>(&self, payload: E) -> Result<E, EventError> {
        self.events.dispatch(payload)
    }

    /// Register an observer; the returned guard is the reversible effect.
    pub fn observe<E: Event>(
        &self,
        handler: impl Fn(&E) + Send + Sync + 'static,
    ) -> ObserverGuard<E> {
        self.events.observe(handler)
    }

    /// Register a waterfall listener; the returned guard is the reversible effect.
    pub fn waterfall<E: Event>(
        &self,
        handler: impl Fn(E, &mut dyn FnMut(E) -> Result<E, EventError>) -> Result<E, EventError>
            + Send
            + Sync
            + 'static,
    ) -> WaterfallGuard<E> {
        self.events.waterfall(handler)
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
