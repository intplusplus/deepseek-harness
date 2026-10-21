//! Checkpoint source identity and its producer-declared wire representation.
//!
//! Ported from `packages/compaction/compaction/src/checkpoint.ts` and the
//! `MessageSourceMap` merge in `packages/compaction/compaction/tests/compaction.spec.ts`.

use dsh_compaction::{
    compact_checkpoint_source, is_compact_checkpoint_source, CommandId, CompactionCheckpointSource,
    CompactionId, COMPACT_CHECKPOINT_KIND,
};
use dsh_llm::{ExtensionSource, MessageSource};
use serde_json::{json, Map, Value};

#[test]
fn checkpoint_source_stamps_kind_before_the_compaction_identity() {
    let source = compact_checkpoint_source(CompactionId::new("c1"), Some(CommandId::new("cmd")));
    let value = serde_json::to_value(source.to_message_source()).unwrap();
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        r#"{"kind":"compact-checkpoint","compactionId":"c1","sourceCommandId":"cmd"}"#
    );
}

#[test]
fn checkpoint_source_omits_an_absent_command() {
    let source = CompactionCheckpointSource::new(CompactionId::new("c1"), None);
    assert_eq!(
        serde_json::to_string(&source.to_message_source()).unwrap(),
        r#"{"kind":"compact-checkpoint","compactionId":"c1"}"#
    );
}

#[test]
fn checkpoint_source_round_trips_through_the_message_source() {
    let source =
        CompactionCheckpointSource::new(CompactionId::new("c1"), Some(CommandId::new("cmd")));
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&source.to_message_source()),
        Some(source)
    );
}

#[test]
fn checkpoint_source_accepts_an_explicit_null_command() {
    let mut fields = Map::new();
    fields.insert(
        "kind".to_string(),
        Value::String(COMPACT_CHECKPOINT_KIND.to_string()),
    );
    fields.insert("compactionId".to_string(), Value::String("c1".to_string()));
    fields.insert("sourceCommandId".to_string(), Value::Null);
    let source = MessageSource::Extension(ExtensionSource::new(COMPACT_CHECKPOINT_KIND, fields));
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&source),
        Some(CompactionCheckpointSource::new(
            CompactionId::new("c1"),
            None
        ))
    );
}

#[test]
fn checkpoint_source_rejects_every_other_source_kind() {
    let other = MessageSource::Extension(ExtensionSource::new(
        "other",
        Map::from_iter([("compactionId".to_string(), Value::String("c1".to_string()))]),
    ));
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&other),
        None
    );
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&MessageSource::User),
        None
    );
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&MessageSource::SystemPrompt),
        None
    );
}

#[test]
fn checkpoint_source_rejects_a_missing_or_non_string_compaction_id() {
    let missing =
        MessageSource::Extension(ExtensionSource::new(COMPACT_CHECKPOINT_KIND, Map::new()));
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&missing),
        None
    );

    let wrong = MessageSource::Extension(ExtensionSource::new(
        COMPACT_CHECKPOINT_KIND,
        Map::from_iter([("compactionId".to_string(), json!(7))]),
    ));
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&wrong),
        None
    );
}

#[test]
fn checkpoint_source_rejects_a_non_string_command_id() {
    let source = MessageSource::Extension(ExtensionSource::new(
        COMPACT_CHECKPOINT_KIND,
        Map::from_iter([
            ("compactionId".to_string(), Value::String("c1".to_string())),
            ("sourceCommandId".to_string(), json!(7)),
        ]),
    ));
    assert_eq!(
        CompactionCheckpointSource::from_message_source(&source),
        None
    );
}

#[test]
fn checkpoint_predicate_matches_only_the_declared_kind() {
    let checkpoint = compact_checkpoint_source(CompactionId::new("c1"), None);
    assert!(is_compact_checkpoint_source(
        &checkpoint.to_message_source()
    ));
    assert!(!is_compact_checkpoint_source(&MessageSource::User));
    assert!(!is_compact_checkpoint_source(&MessageSource::Extension(
        ExtensionSource::new("other", Map::new())
    )));
}
