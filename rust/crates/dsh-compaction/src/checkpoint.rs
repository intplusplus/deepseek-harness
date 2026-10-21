//! The compaction checkpoint message source.
//!
//! Every backend replaces a compacted span with one `user/message` carrying
//! this producer-declared source kind. The transaction identity lets a pure
//! surface fold pair the replacement with the immediately preceding
//! `compaction/summary` record without decoding the summary content.
//!
//! TypeScript owns the constructor and predicate in
//! `@deepseek-ai/dsh-compaction/checkpoint`; this module keeps them free of
//! session or plugin state so client and wire programs can name the source.

#![forbid(unsafe_code)]

use serde_json::{Map, Value};

use dsh_llm::{ExtensionSource, MessageSource};

use super::brand::{CommandId, CompactionId};

/// The backend-independent `MessageSource.kind` carried by a checkpoint.
pub const COMPACT_CHECKPOINT_KIND: &str = "compact-checkpoint";

/// A concrete compaction checkpoint source.
///
/// The `kind` marker is implicit in this type and stamped by
/// [`CompactionCheckpointSource::to_message_source`]. `source_command_id` is
/// present only for a manual compaction that named its initiating command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionCheckpointSource {
    /// Owning compaction identity.
    pub compaction_id: CompactionId,
    /// Initiating manual command, when present.
    pub source_command_id: Option<CommandId>,
}

impl CompactionCheckpointSource {
    /// Create a checkpoint source correlated with one transaction.
    pub fn new(compaction_id: CompactionId, source_command_id: Option<CommandId>) -> Self {
        Self {
            compaction_id,
            source_command_id,
        }
    }

    /// Stamp this checkpoint as a producer-declared message source.
    ///
    /// Field order matches the TypeScript constructor: `kind`, then
    /// `compactionId`, then the optional `sourceCommandId`.
    pub fn to_message_source(&self) -> MessageSource {
        let mut fields = Map::new();
        // Pre-insert `kind` so the extension source keeps it first; inserting
        // an existing key preserves its position.
        fields.insert(
            "kind".to_string(),
            Value::String(COMPACT_CHECKPOINT_KIND.to_string()),
        );
        fields.insert(
            "compactionId".to_string(),
            Value::String(self.compaction_id.as_str().to_string()),
        );
        if let Some(source_command_id) = &self.source_command_id {
            fields.insert(
                "sourceCommandId".to_string(),
                Value::String(source_command_id.as_str().to_string()),
            );
        }
        MessageSource::Extension(ExtensionSource::new(COMPACT_CHECKPOINT_KIND, fields))
    }

    /// Recover a checkpoint source from a persisted message source.
    ///
    /// Returns `None` for every other source kind and for a checkpoint marker
    /// whose `compactionId` is not a string.
    pub fn from_message_source(source: &MessageSource) -> Option<Self> {
        let MessageSource::Extension(extension) = source else {
            return None;
        };
        if extension.kind() != COMPACT_CHECKPOINT_KIND {
            return None;
        }
        let fields = extension.fields();
        let compaction_id = fields.get("compactionId")?.as_str()?;
        let source_command_id = match fields.get("sourceCommandId") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(CommandId::new(value)),
            Some(_) => return None,
        };
        Some(Self {
            compaction_id: CompactionId::new(compaction_id),
            source_command_id,
        })
    }
}

/// Create a checkpoint source correlated with one compaction transaction.
pub fn compact_checkpoint_source(
    compaction_id: CompactionId,
    source_command_id: Option<CommandId>,
) -> CompactionCheckpointSource {
    CompactionCheckpointSource::new(compaction_id, source_command_id)
}

/// Test whether a persisted source carries the checkpoint marker.
pub fn is_compact_checkpoint_source(source: &MessageSource) -> bool {
    source.kind() == COMPACT_CHECKPOINT_KIND
}
