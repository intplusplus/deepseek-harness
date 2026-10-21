//! The message vocabulary carried inside a frame.
//!
//! Ported from `packages/ptc-runtime/ptc-runtime-node/src/protocol.ts` and the
//! receive paths of `persistent.ts` / `persistent-process.ts`.
//!
//! Two directions, two shapes:
//!
//! * **One-shot** (`process.ts`): the program posts bare
//!   [`ProgramToHost`] messages and the host answers bare [`Reply`]s.
//! * **Persistent** (`persistent.ts`, `persistent-process.ts`): each
//!   program message is wrapped as `{ id, payload }` — the evaluation id — and
//!   the host opens with [`HostToProgram::Ready`], then one
//!   [`HostToProgram::Run`] per evaluation.
//!
//! Every decoder here is strict: an unknown `type`, a missing field, a
//! wrongly typed field, or an extra field is a [`ProtocolError`], never a
//! silently degraded message. The program is model-written code, so "this
//! frame does not make sense" has to stop the run rather than guess.

#![forbid(unsafe_code)]

use serde_json::{Map, Value};

use super::reserved::BindingErrorClass;

/// Why a frame's payload was refused.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProtocolError {
    /// The payload was not a JSON object.
    #[error("ptc protocol: {kind} must be a JSON object, got {actual}")]
    NotAnObject {
        /// The message tag being decoded.
        kind: &'static str,
        /// The JSON type actually found.
        actual: &'static str,
    },
    /// The `type` tag is not part of the vocabulary.
    #[error("ptc protocol: unknown message type {value:?}")]
    UnknownType {
        /// The refused tag.
        value: String,
    },
    /// A required field was absent.
    #[error("ptc protocol: {kind} requires field {field:?}")]
    MissingField {
        /// The message tag being decoded.
        kind: &'static str,
        /// The absent field.
        field: &'static str,
    },
    /// A field had the wrong JSON type.
    #[error("ptc protocol: {kind}.{field} must be {expected}")]
    WrongType {
        /// The message tag being decoded.
        kind: &'static str,
        /// The offending field.
        field: &'static str,
        /// What the field must be.
        expected: &'static str,
    },
    /// A field the vocabulary does not declare was present.
    #[error("ptc protocol: {kind} has undeclared field {field:?}")]
    UnknownField {
        /// The message tag being decoded.
        kind: &'static str,
        /// The unexpected field.
        field: String,
    },
    /// The `done` diagnostic named a failure class the program cannot report.
    #[error("ptc protocol: done.error.kind {value:?} is not a program failure class")]
    UnknownFailureKind {
        /// The refused kind.
        value: String,
    },
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Read the `type` discriminator, distinguishing a non-object payload from an
/// object that forgot its tag: `{}` must not be reported as "not an object".
fn message_tag(value: &Value, kind: &'static str) -> Result<String, ProtocolError> {
    let Value::Object(map) = value else {
        return Err(ProtocolError::NotAnObject {
            kind,
            actual: json_type(value),
        });
    };
    match map.get("type") {
        Some(Value::String(tag)) => Ok(tag.clone()),
        Some(_) => Err(ProtocolError::WrongType {
            kind,
            field: "type",
            expected: "a string",
        }),
        None => Err(ProtocolError::MissingField {
            kind,
            field: "type",
        }),
    }
}

/// A message's fields, drained so an undeclared remainder is detectable.
struct Fields {
    kind: &'static str,
    map: Map<String, Value>,
}

impl Fields {
    fn new(kind: &'static str, value: &Value) -> Result<Self, ProtocolError> {
        match value {
            Value::Object(map) => Ok(Self {
                kind,
                map: map.clone(),
            }),
            other => Err(ProtocolError::NotAnObject {
                kind,
                actual: json_type(other),
            }),
        }
    }

    fn take(&mut self, field: &'static str) -> Result<Value, ProtocolError> {
        self.map.remove(field).ok_or(ProtocolError::MissingField {
            kind: self.kind,
            field,
        })
    }

    fn optional(&mut self, field: &'static str) -> Option<Value> {
        self.map.remove(field)
    }

    fn string(&mut self, field: &'static str) -> Result<String, ProtocolError> {
        match self.take(field)? {
            Value::String(text) => Ok(text),
            _ => Err(ProtocolError::WrongType {
                kind: self.kind,
                field,
                expected: "a string",
            }),
        }
    }

    fn number(&mut self, field: &'static str) -> Result<u64, ProtocolError> {
        match self.take(field)? {
            Value::Number(number) => number.as_u64().ok_or(ProtocolError::WrongType {
                kind: self.kind,
                field,
                expected: "a non-negative integer",
            }),
            _ => Err(ProtocolError::WrongType {
                kind: self.kind,
                field,
                expected: "a non-negative integer",
            }),
        }
    }

    fn bool(&mut self, field: &'static str) -> Result<bool, ProtocolError> {
        match self.take(field)? {
            Value::Bool(value) => Ok(value),
            _ => Err(ProtocolError::WrongType {
                kind: self.kind,
                field,
                expected: "a boolean",
            }),
        }
    }

    /// A field that must be a wire token array.
    fn wire(&mut self, field: &'static str) -> Result<Value, ProtocolError> {
        match self.take(field)? {
            value @ Value::Array(_) => Ok(value),
            _ => Err(ProtocolError::WrongType {
                kind: self.kind,
                field,
                expected: "a flat JSON wire array",
            }),
        }
    }

    fn finish(self) -> Result<(), ProtocolError> {
        match self.map.keys().next() {
            None => Ok(()),
            Some(field) => Err(ProtocolError::UnknownField {
                kind: self.kind,
                field: field.clone(),
            }),
        }
    }
}

/// One namespace the program should materialize.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NamespaceDeclaration {
    /// The global identifier the program sees.
    pub global: String,
    /// The callable member names, in declaration order.
    pub names: Vec<String>,
    /// The rejection class this namespace injects, when declared.
    pub error_class: Option<BindingErrorClass>,
}

impl NamespaceDeclaration {
    fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("global".to_owned(), Value::String(self.global.clone()));
        map.insert(
            "names".to_owned(),
            Value::Array(self.names.iter().cloned().map(Value::String).collect()),
        );
        if let Some(error_class) = &self.error_class {
            map.insert(
                "errorClass".to_owned(),
                serde_json::to_value(error_class).expect("a descriptor is always serializable"),
            );
        }
        Value::Object(map)
    }

    fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let mut fields = Fields::new("namespaces[]", value)?;
        let global = fields.string("global")?;
        let names = match fields.take("names")? {
            Value::Array(items) => {
                let mut names = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::String(name) => names.push(name),
                        _ => {
                            return Err(ProtocolError::WrongType {
                                kind: "namespaces[]",
                                field: "names",
                                expected: "an array of strings",
                            })
                        }
                    }
                }
                names
            }
            _ => {
                return Err(ProtocolError::WrongType {
                    kind: "namespaces[]",
                    field: "names",
                    expected: "an array of strings",
                })
            }
        };
        let error_class = match fields.optional("errorClass") {
            None | Some(Value::Null) => None,
            Some(value) => Some(serde_json::from_value::<BindingErrorClass>(value).map_err(
                |_| ProtocolError::WrongType {
                    kind: "namespaces[]",
                    field: "errorClass",
                    expected: "a class descriptor",
                },
            )?),
        };
        fields.finish()?;
        Ok(Self {
            global,
            names,
            error_class,
        })
    }
}

/// Everything the program needs to start one evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramBootData {
    /// The program source, already stripped of TypeScript types.
    pub code: String,
    /// Binding namespaces to materialize; the functions stay host-side.
    pub namespaces: Vec<NamespaceDeclaration>,
    /// Hard cap for the combined serialized outer logs plus completion value or
    /// failure diagnostic.
    pub max_output_bytes: u64,
}

impl ProgramBootData {
    /// Serialize in the producer's field order (`code`, `namespaces`,
    /// `maxOutputBytes`).
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("code".to_owned(), Value::String(self.code.clone()));
        map.insert(
            "namespaces".to_owned(),
            Value::Array(
                self.namespaces
                    .iter()
                    .map(NamespaceDeclaration::to_json)
                    .collect(),
            ),
        );
        map.insert(
            "maxOutputBytes".to_owned(),
            Value::from(self.max_output_bytes),
        );
        Value::Object(map)
    }

    /// Decode one boot payload.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let mut fields = Fields::new("run.data", value)?;
        let code = fields.string("code")?;
        let namespaces = match fields.take("namespaces")? {
            Value::Array(items) => {
                let mut namespaces = Vec::with_capacity(items.len());
                for item in &items {
                    namespaces.push(NamespaceDeclaration::from_json(item)?);
                }
                namespaces
            }
            _ => {
                return Err(ProtocolError::WrongType {
                    kind: "run.data",
                    field: "namespaces",
                    expected: "an array",
                })
            }
        };
        let max_output_bytes = fields.number("maxOutputBytes")?;
        fields.finish()?;
        Ok(Self {
            code,
            namespaces,
            max_output_bytes,
        })
    }
}

/// The three failure classes a program may report about itself.
///
/// Budgets, aborts, and substrate death are observed host-side, so they are
/// not part of this vocabulary: a program claiming a timeout would be
/// asserting something only the host can know.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ProgramFailureKind {
    /// The program threw or failed to transform.
    #[error("exception")]
    Exception,
    /// The completion value was not lossless JSON.
    #[error("invalid-output")]
    InvalidOutput,
    /// Program-side capture or completion measurement crossed the cap.
    #[error("output-limit")]
    OutputLimit,
}

impl ProgramFailureKind {
    /// The wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exception => "exception",
            Self::InvalidOutput => "invalid-output",
            Self::OutputLimit => "output-limit",
        }
    }

    /// Parse the wire spelling.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "exception" => Some(Self::Exception),
            "invalid-output" => Some(Self::InvalidOutput),
            "output-limit" => Some(Self::OutputLimit),
            _ => None,
        }
    }
}

/// One program-reported failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramFailure {
    /// The failure class.
    pub kind: ProgramFailureKind,
    /// Human-readable detail, already admitted against the output cap.
    pub message: String,
}

/// Program → host.
#[derive(Clone, Debug, PartialEq)]
pub enum ProgramToHost {
    /// One bridged binding call.
    Call {
        /// Program-issued correlation id; the host answers each id at most once.
        id: u64,
        /// The namespace global the call targets.
        global: String,
        /// The function name within the namespace.
        name: String,
        /// The single argument as a flat lossless-JSON wire value.
        args: Value,
    },
    /// Captured text, streamed eagerly so it survives a mid-run termination.
    Log {
        /// The captured text.
        text: String,
    },
    /// Program-side capture crossed the outer cap.
    OutputLimit,
    /// The program settled.
    Done {
        /// The completion value, present only on a clean completion that
        /// produced one.
        value: Option<Value>,
        /// The program-reported failure, when it failed.
        error: Option<ProgramFailure>,
    },
}

impl ProgramToHost {
    /// The message's `type` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Call { .. } => "call",
            Self::Log { .. } => "log",
            Self::OutputLimit => "output-limit",
            Self::Done { .. } => "done",
        }
    }

    /// Serialize in the producer's field order; `type` is always first.
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        match self {
            Self::Call {
                id,
                global,
                name,
                args,
            } => {
                map.insert("type".to_owned(), Value::String("call".to_owned()));
                map.insert("id".to_owned(), Value::from(*id));
                map.insert("global".to_owned(), Value::String(global.clone()));
                map.insert("name".to_owned(), Value::String(name.clone()));
                map.insert("args".to_owned(), args.clone());
            }
            Self::Log { text } => {
                map.insert("type".to_owned(), Value::String("log".to_owned()));
                map.insert("text".to_owned(), Value::String(text.clone()));
            }
            Self::OutputLimit => {
                map.insert("type".to_owned(), Value::String("output-limit".to_owned()));
            }
            Self::Done { value, error } => {
                map.insert("type".to_owned(), Value::String("done".to_owned()));
                if let Some(value) = value {
                    map.insert("value".to_owned(), value.clone());
                }
                if let Some(error) = error {
                    let mut failure = Map::new();
                    failure.insert(
                        "kind".to_owned(),
                        Value::String(error.kind.as_str().to_owned()),
                    );
                    failure.insert("message".to_owned(), Value::String(error.message.clone()));
                    map.insert("error".to_owned(), Value::Object(failure));
                }
            }
        }
        Value::Object(map)
    }

    /// Decode one program message.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let tag = message_tag(value, "program message")?;
        let kind = match tag.as_str() {
            "call" => "call",
            "log" => "log",
            "output-limit" => "output-limit",
            "done" => "done",
            other => {
                return Err(ProtocolError::UnknownType {
                    value: other.to_owned(),
                })
            }
        };
        let mut fields = Fields::new(kind, value)?;
        fields.take("type")?;
        let message = match kind {
            "call" => Self::Call {
                id: fields.number("id")?,
                global: fields.string("global")?,
                name: fields.string("name")?,
                args: fields.wire("args")?,
            },
            "log" => Self::Log {
                text: fields.string("text")?,
            },
            "output-limit" => Self::OutputLimit,
            _ => {
                let value = match fields.optional("value") {
                    None => None,
                    Some(value @ Value::Array(_)) => Some(value),
                    Some(_) => {
                        return Err(ProtocolError::WrongType {
                            kind: "done",
                            field: "value",
                            expected: "a flat JSON wire array",
                        })
                    }
                };
                let error = match fields.optional("error") {
                    None | Some(Value::Null) => None,
                    Some(raw) => {
                        let mut failure = Fields::new("done.error", &raw)?;
                        let kind = failure.string("kind")?;
                        let kind = ProgramFailureKind::parse(&kind)
                            .ok_or(ProtocolError::UnknownFailureKind { value: kind })?;
                        let message = failure.string("message")?;
                        failure.finish()?;
                        Some(ProgramFailure { kind, message })
                    }
                };
                Self::Done { value, error }
            }
        };
        fields.finish()?;
        Ok(message)
    }
}

/// Host → program: the answer to one [`ProgramToHost::Call`].
#[derive(Clone, Debug, PartialEq)]
pub enum Reply {
    /// The call resolved to a lossless-JSON value.
    Ok {
        /// The correlation id being answered.
        id: u64,
        /// The resolution as a flat wire value.
        value: Value,
    },
    /// The call rejected; the program sees a rejection carrying this message.
    Error {
        /// The correlation id being answered.
        id: u64,
        /// The rejection message.
        message: String,
    },
}

impl Reply {
    /// Serialize in the producer's field order; `type` is always first.
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("type".to_owned(), Value::String("reply".to_owned()));
        match self {
            Self::Ok { id, value } => {
                map.insert("id".to_owned(), Value::from(*id));
                map.insert("ok".to_owned(), Value::Bool(true));
                map.insert("value".to_owned(), value.clone());
            }
            Self::Error { id, message } => {
                map.insert("id".to_owned(), Value::from(*id));
                map.insert("ok".to_owned(), Value::Bool(false));
                map.insert("message".to_owned(), Value::String(message.clone()));
            }
        }
        Value::Object(map)
    }

    /// Decode one host reply.
    ///
    /// The `ok` discriminator is authoritative: `ok: true` must carry
    /// `value`, `ok: false` must carry `message`, and the other field being
    /// present is an undeclared-field error.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let mut fields = Fields::new("reply", value)?;
        fields.take("type")?;
        let id = fields.number("id")?;
        let ok = fields.bool("ok")?;
        let reply = if ok {
            Self::Ok {
                id,
                value: fields.wire("value")?,
            }
        } else {
            Self::Error {
                id,
                message: fields.string("message")?,
            }
        };
        fields.finish()?;
        Ok(reply)
    }
}

/// Host → program.
#[derive(Clone, Debug, PartialEq)]
pub enum HostToProgram {
    /// The persistent child is ready for its first evaluation.
    Ready,
    /// Start one evaluation of `data` in `cwd`.
    Run {
        /// The evaluation id every program message is wrapped with.
        id: u64,
        /// Working directory for this evaluation.
        cwd: String,
        /// The boot payload.
        data: ProgramBootData,
    },
    /// The answer to one binding call.
    Reply(Reply),
}

impl HostToProgram {
    /// The message's `type` tag.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Run { .. } => "run",
            Self::Reply(_) => "reply",
        }
    }

    /// Serialize in the producer's field order; `type` is always first.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Ready => {
                let mut map = Map::new();
                map.insert("type".to_owned(), Value::String("ready".to_owned()));
                Value::Object(map)
            }
            Self::Run { id, cwd, data } => {
                let mut map = Map::new();
                map.insert("type".to_owned(), Value::String("run".to_owned()));
                map.insert("id".to_owned(), Value::from(*id));
                map.insert("cwd".to_owned(), Value::String(cwd.clone()));
                map.insert("data".to_owned(), data.to_json());
                Value::Object(map)
            }
            Self::Reply(reply) => reply.to_json(),
        }
    }

    /// Decode one host message.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let tag = message_tag(value, "host message")?;
        match tag.as_str() {
            "ready" => {
                let mut fields = Fields::new("ready", value)?;
                fields.take("type")?;
                fields.finish()?;
                Ok(Self::Ready)
            }
            "run" => {
                let mut fields = Fields::new("run", value)?;
                fields.take("type")?;
                let id = fields.number("id")?;
                let cwd = fields.string("cwd")?;
                let data = ProgramBootData::from_json(&fields.take("data")?)?;
                fields.finish()?;
                Ok(Self::Run { id, cwd, data })
            }
            "reply" => Ok(Self::Reply(Reply::from_json(value)?)),
            other => Err(ProtocolError::UnknownType {
                value: other.to_owned(),
            }),
        }
    }
}

/// The persistent channel's envelope: one program message tagged with the
/// evaluation that produced it.
///
/// The host refuses a frame whose `id` is not the active evaluation's — a
/// late reply from a superseded evaluation would otherwise be read as the
/// current one's.
#[derive(Clone, Debug, PartialEq)]
pub struct ProgramEnvelope {
    /// The evaluation id this message belongs to.
    pub id: u64,
    /// The program message.
    pub payload: ProgramToHost,
}

impl ProgramEnvelope {
    /// Serialize as `{ id, payload }`.
    pub fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("id".to_owned(), Value::from(self.id));
        map.insert("payload".to_owned(), self.payload.to_json());
        Value::Object(map)
    }

    /// Decode one envelope.
    pub fn from_json(value: &Value) -> Result<Self, ProtocolError> {
        let mut fields = Fields::new("envelope", value)?;
        let id = fields.number("id")?;
        let payload = ProgramToHost::from_json(&fields.take("payload")?)?;
        fields.finish()?;
        Ok(Self { id, payload })
    }
}
