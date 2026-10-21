//! The typed event bus: observer events whose failures are contained, and
//! waterfall events whose listeners must delegate through `next()`.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{Arc, RwLock, Weak},
};

/// Marker trait for typed events carried by the bus.
///
/// Events are the extension points: core and plugin code alike communicate
/// through typed values rather than concrete interfaces, exactly as the
/// TypeScript harness declares events in `interface Events` blocks.
pub trait Event: Any + Send + Sync {}

impl<T: Any + Send + Sync> Event for T {}

/// Errors raised while dispatching a waterfall event.
#[derive(Debug, thiserror::Error)]
pub enum EventError {
    #[error("event listener failed: {0}")]
    Listener(String),
}

/// An observer registration. Observers cannot veto the mutation they observe.
pub type ObserverFn<E> = Arc<dyn Fn(&E) + Send + Sync + 'static>;

/// A waterfall listener.
///
/// Receives the payload by value together with a `next` closure that delegates
/// to the remaining listeners. A listener MUST call `next(payload)` to
/// delegate; returning without calling it short-circuits the chain. Once every
/// listener has delegated, the chain terminates with the payload unchanged —
/// the core default listener sits at the bottom of the chain.
pub type WaterfallFn<E> = Arc<
    dyn Fn(E, &mut dyn FnMut(E) -> Result<E, EventError>) -> Result<E, EventError>
        + Send
        + Sync
        + 'static,
>;

/// Per-event-type handler storage, type-erased behind `Box<dyn Any + Send + Sync>`.
struct HandlerList<E> {
    observers: Vec<ObserverFn<E>>,
    waterfalls: Vec<WaterfallFn<E>>,
}

impl<E> Default for HandlerList<E> {
    fn default() -> Self {
        Self {
            observers: Vec::new(),
            waterfalls: Vec::new(),
        }
    }
}

/// The typed event bus.
///
/// Registrations return guards whose drop is the reversible effect that
/// unregisters the handler, so a plugin's event wiring unwinds when the
/// plugin unloads.
#[derive(Default)]
pub struct EventBus {
    lists: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>>,
}

impl EventBus {
    fn with_list_mut<E: Event, R>(&self, f: impl FnOnce(&mut HandlerList<E>) -> R) -> R {
        let mut lists = self
            .lists
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let entry = lists
            .entry(TypeId::of::<E>())
            .or_insert_with(|| Box::new(HandlerList::<E>::default()));
        let list = entry
            .downcast_mut::<HandlerList<E>>()
            .expect("event handler list type mismatch");
        f(list)
    }

    fn handlers<E: Event>(&self) -> Option<HandlerList<E>> {
        let lists = self
            .lists
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        lists
            .get(&TypeId::of::<E>())
            .and_then(|entry| entry.downcast_ref::<HandlerList<E>>())
            .map(|list| HandlerList {
                observers: list.observers.clone(),
                waterfalls: list.waterfalls.clone(),
            })
    }

    /// Register an observer for `E`.
    pub fn observe<E: Event>(
        self: &Arc<Self>,
        handler: impl Fn(&E) + Send + Sync + 'static,
    ) -> ObserverGuard<E> {
        let handler: ObserverFn<E> = Arc::new(handler);
        self.with_list_mut::<E, _>(|list| list.observers.push(Arc::clone(&handler)));
        ObserverGuard {
            bus: Arc::downgrade(self),
            handler,
        }
    }

    /// Register a waterfall listener for `E`.
    pub fn waterfall<E: Event>(
        self: &Arc<Self>,
        handler: impl Fn(E, &mut dyn FnMut(E) -> Result<E, EventError>) -> Result<E, EventError>
            + Send
            + Sync
            + 'static,
    ) -> WaterfallGuard<E> {
        let handler: WaterfallFn<E> = Arc::new(handler);
        self.with_list_mut::<E, _>(|list| list.waterfalls.push(Arc::clone(&handler)));
        WaterfallGuard {
            bus: Arc::downgrade(self),
            handler,
        }
    }

    /// Emit an observer event.
    ///
    /// Observer failures are contained and cannot veto the mutation that
    /// triggered the emit: a panicking observer is caught here and dropped.
    pub fn emit<E: Event>(&self, event: &E) {
        let Some(list) = self.handlers::<E>() else {
            return;
        };
        for handler in list.observers {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(event)));
            if outcome.is_err() {
                // Contained: the panic payload is intentionally dropped.
            }
        }
    }

    /// Dispatch a waterfall event, returning the final payload.
    pub fn dispatch<E: Event>(&self, payload: E) -> Result<E, EventError> {
        let handlers: Vec<WaterfallFn<E>> = self
            .handlers::<E>()
            .map(|list| list.waterfalls)
            .unwrap_or_default();
        run_waterfall(&handlers, payload)
    }
}

/// Run the chain: each listener may rewrite the payload and must call `next`
/// to delegate; the chain terminates with the payload once listeners run out.
fn run_waterfall<E>(handlers: &[WaterfallFn<E>], payload: E) -> Result<E, EventError> {
    match handlers.split_first() {
        None => Ok(payload),
        Some((head, rest)) => head(payload, &mut |next_payload| {
            run_waterfall(rest, next_payload)
        }),
    }
}

/// The reversible effect of one observer registration.
pub struct ObserverGuard<E: Event> {
    bus: Weak<EventBus>,
    handler: ObserverFn<E>,
}

impl<E: Event> Drop for ObserverGuard<E> {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade() {
            bus.with_list_mut::<E, _>(|list| {
                list.observers
                    .retain(|registered| !Arc::ptr_eq(registered, &self.handler));
            });
        }
    }
}

/// The reversible effect of one waterfall registration.
pub struct WaterfallGuard<E: Event> {
    bus: Weak<EventBus>,
    handler: WaterfallFn<E>,
}

impl<E: Event> Drop for WaterfallGuard<E> {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade() {
            bus.with_list_mut::<E, _>(|list| {
                list.waterfalls
                    .retain(|registered| !Arc::ptr_eq(registered, &self.handler));
            });
        }
    }
}
