//! The cooperative `system-prompt/assemble` waterfall.
//!
//! TypeScript dispatches the event through the Cordis waterfall:
//!
//! ```ts
//! const transformed = await this.ctx.waterfall(
//!   scopeTarget(this, scope), 'system-prompt/assemble', assembly, context,
//!   () => Promise.resolve(assembly),
//! )
//! ```
//!
//! Every listener receives the current assembly, the caller's assembly
//! context, and a `next` that delegates to the rest of the chain. A listener
//! **must** call `next(value)` to delegate; returning without calling it
//! short-circuits the chain and its own return value becomes the result.
//! Once every listener has delegated, the chain terminates with the payload
//! unchanged — the terminal `() => Promise.resolve(assembly)`.
//!
//! The runner mirrors `dsh-plugin`'s `run_waterfall` (recursive
//! `split_first` chain, every listener exactly once, deterministic order)
//! without the `Result` channel: this registry's fallible work all happens
//! *before* the waterfall, and a listener that fails is a host problem, not a
//! registry rule.

#![forbid(unsafe_code)]

use std::rc::Rc;

use super::types::{AssembleContext, PromptAssembly};

/// One `system-prompt/assemble` listener.
///
/// Receives the assembly by value together with a `next` closure that
/// delegates to the remaining listeners, passing the (possibly rewritten)
/// assembly down. Returning without calling `next` short-circuits the chain.
pub type AssembleListener = Rc<
    dyn Fn(
            PromptAssembly,
            &AssembleContext,
            &mut dyn FnMut(PromptAssembly) -> PromptAssembly,
        ) -> PromptAssembly
        + 'static,
>;

/// Run the chain: each listener may rewrite the assembly and must call `next`
/// to delegate; the chain terminates with the assembly once listeners run out.
///
/// The listener list is a snapshot taken at dispatch, so a listener that
/// registers or disposes another listener does not change the chain in
/// flight — the same contract `dsh-plugin`'s `EventBus::dispatch` keeps.
pub(crate) fn run_assemble_waterfall(
    listeners: &[AssembleListener],
    context: &AssembleContext,
    assembly: PromptAssembly,
) -> PromptAssembly {
    match listeners.split_first() {
        None => assembly,
        Some((head, rest)) => head(assembly, context, &mut |value| {
            run_assemble_waterfall(rest, context, value)
        }),
    }
}
