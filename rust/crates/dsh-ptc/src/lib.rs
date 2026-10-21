//! `dsh-ptc` — the offline-testable half of programmatic tool calling.
//!
//! Ported from two TypeScript packages:
//!
//! * `packages/core/tools/src/ptc.ts` + `ptc-calls.ts` — the PTC mode
//!   transport: nested tool dispatches scheduled under the native concurrency
//!   contract, each one recorded as a replayable `tool/ptc-dispatch*` pair.
//! * `packages/ptc-runtime/{ptc-runtime,ptc-runtime-node}` — the host ↔
//!   sandbox seam: reserved names, the flat lossless-JSON wire, the
//!   length-prefixed control channel, the combined output ledger, and the
//!   retained-process policy.
//!
//! What this crate deliberately does **not** do: spawn a process, open a
//! stream, consult a filesystem, or touch a network. Every module is a pure
//! function over values, so the whole protocol is testable in-process:
//!
//! | module | TypeScript source | shape |
//! | --- | --- | --- |
//! | [`events`] | `core/tools/src/types.ts` (`SessionEventMap`) | durable `tool/ptc-dispatch*` records |
//! | [`wire`] | `ptc-runtime-node/src/json-wire.ts` | flat pre-order token transport |
//! | [`frames`] | `ptc-runtime-node/src/channel.ts` | 4-byte big-endian length-prefixed frames |
//! | [`protocol`] | `ptc-runtime-node/src/protocol.ts` | the message vocabulary inside a frame |
//! | [`ledger`] | `ptc-runtime-node/src/{output-json,output-ledger}.ts` | exact JSON-byte accounting |
//! | [`calls`] | `core/tools/src/ptc.ts` (the dispatch lane) | submission-ordered ids and commit lane |
//! | [`reserved`] | `ptc-runtime/src/index.ts`, `ptc-runtime-node/src/bindings.ts` | backend-owned names |
//! | [`policy`] | `ptc-runtime-node/src/{persistent,policy-processes}.ts` | reuse / spawn / reject |
//!
//! Field order is observable: the workspace enables `serde_json`'s
//! `preserve_order`, and a session log is compared byte-for-byte against the
//! TypeScript wire format. [`events`] therefore emits `type` first and the
//! remaining fields in producer order.

#![forbid(unsafe_code)]

pub mod calls;
pub mod events;
pub mod frames;
pub mod ledger;
pub mod policy;
pub mod protocol;
pub mod reserved;
pub mod types;
pub mod wire;

pub use calls::{
    run_over_message, sub_call_id, Classification, DispatchLane, LaneStep, RunOverPhase,
    SubCallSequence, RUN_CODE_NAME,
};
pub use events::{
    PtcDispatch, PtcDispatchStart, PtcEvent, PTC_DISPATCH_START_TAG, PTC_DISPATCH_TAG,
    PTC_EVENT_TAGS,
};
pub use frames::{encode_frame, FrameDecoder, FrameError};
pub use ledger::{
    json_string_bytes_up_to, json_value_bytes_up_to, truncate_json_string_bytes, OutputLedger,
    ProgramLogBuffer, ProgramLogPush,
};
pub use policy::{
    clamp_timeout, decide_process, invalidated_processes, retains_state, ModeChange, Owner,
    PolicyError, PolicyKey, ProcessDecision, RetainedProcess, Timeout, TimeoutRequest,
};
pub use protocol::{
    HostToProgram, NamespaceDeclaration, ProgramBootData, ProgramEnvelope, ProgramFailure,
    ProgramFailureKind, ProgramToHost, ProtocolError, Reply,
};
pub use reserved::{
    is_dunder_member, is_identifier, validate_bindings, BindingErrorClass, BindingNamespace,
    ReservedError, ValidatedBindings, PORTABLE_RESERVED_WORDS, RESERVED_BINDING_GLOBALS,
    RESERVED_ERROR_MEMBERS,
};
pub use types::{
    PtcFailureKind, PtcRunFailure, PtcRunResult, PtcRunSandbox, SandboxEnforcement, SandboxMode,
};
pub use wire::{decode_json_wire, encode_json_wire, WireError, MAX_SAFE_INTEGER};
