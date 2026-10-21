//! Preset migration through the catalog dispatch (TS
//! `session-format-catalog/tests/preset-migration.spec.ts`): every historical
//! entry generation migrates all preset selections before projection or fork.

use dsh_session_format_catalog::SessionFormatCatalog;
use serde_json::{json, Value};

/// The seeded preset scenario shared by v0/v1 (physical `seedLength`) and
/// v2 (an explicit inherited end-seed marker), from the TS spec.
fn seeded_preset_source(version: u32) -> String {
    let mut header = json!({
        "type": "session",
        "version": version,
        "id": "code",
        "createdAt": 1,
        "parentSession": "parent",
        "delegationDepth": 0,
        "agentPreset": "code",
    });
    if version == 2 {
        header["isSeeded"] = json!(true);
    } else {
        header["seedLength"] = json!(2);
    }
    let selections = ["code", "standard", "code", "minimal", "code"];
    let mut lines = vec![header.to_string()];
    for (index, agent_preset) in selections.iter().enumerate() {
        let seq = match version {
            // v0/v1 rows are dense; the inherited cut is inserted by the chain.
            2 if index >= 2 => index + 1,
            _ => index,
        };
        lines.push(
            json!({
                "type": "agent-preset/selected",
                "seq": seq,
                "time": index + 1,
                "data": {"agentPreset": agent_preset},
            })
            .to_string(),
        );
    }
    if version == 2 {
        // Insert the inherited marker between the local prefix and the tail,
        // exactly where the TS spec places it.
        lines.insert(
            3,
            json!({
                "type": "session/end-seed",
                "seq": 2,
                "time": 2,
                "data": {"inherited": true},
            })
            .to_string(),
        );
    }
    lines.join("\n")
}

/// The restored selections: `code` was renamed to `ptc` by the v2→v3 edge,
/// the other names pass through.
fn expected_selections() -> Vec<Value> {
    ["ptc", "standard", "ptc", "minimal", "ptc"]
        .iter()
        .map(|name| json!({"agentPreset": name}))
        .collect()
}

fn selection_events(catalog_selected: &[dsh_session::migration::Envelope]) -> Vec<Value> {
    catalog_selected
        .iter()
        .filter(|event| event.r#type == "agent-preset/selected")
        .map(|event| event.data.clone().into())
        .collect()
}

#[test]
fn a_seeded_v0_header_and_every_selection_migrate() {
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&seeded_preset_source(0))
        .expect("v0 migrates");
    assert_eq!(
        artifact.header,
        json!({
            "version": 4,
            "id": "code",
            "createdAt": 1,
            "parentSession": "parent",
            "isSeeded": true,
            "delegationDepth": 0,
            "agentPreset": "ptc",
        }),
    );
    assert_eq!(artifact.inherited_event_count, 2);
    let seqs: Vec<u64> = artifact.events.iter().map(|event| event.seq).collect();
    assert_eq!(
        seqs,
        [0, 1, 2, 3, 4, 5],
        "the inherited cut is inserted densely"
    );
    assert_eq!(artifact.events[2].r#type, "session/end-seed");
    assert_eq!(artifact.events[2].data["inherited"], json!(true));
    assert_eq!(selection_events(&artifact.events), expected_selections());
}

#[test]
fn a_seeded_v1_header_and_every_selection_migrate() {
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&seeded_preset_source(1))
        .expect("v1 migrates");
    assert_eq!(artifact.header["agentPreset"], json!("ptc"));
    assert_eq!(artifact.header["isSeeded"], json!(true));
    assert_eq!(artifact.inherited_event_count, 2);
    assert_eq!(selection_events(&artifact.events), expected_selections());
}

#[test]
fn a_seeded_v2_header_and_every_selection_migrate() {
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&seeded_preset_source(2))
        .expect("v2 migrates");
    assert_eq!(artifact.header["agentPreset"], json!("ptc"));
    assert_eq!(artifact.inherited_event_count, 2);
    let seqs: Vec<u64> = artifact.events.iter().map(|event| event.seq).collect();
    assert_eq!(seqs, [0, 1, 2, 3, 4, 5]);
    assert_eq!(artifact.events[2].r#type, "session/end-seed");
    assert_eq!(selection_events(&artifact.events), expected_selections());
}

#[test]
fn a_v3_custom_preset_named_code_is_not_reinterpreted() {
    // The v2→v3 rename is behind the catalog: a v3 `code` is a custom preset
    // and must reach v4 unchanged.
    let text = [
        json!({
            "type": "session",
            "version": 3,
            "id": "native",
            "createdAt": 1,
            "isSeeded": false,
            "delegationDepth": 0,
            "agentPreset": "code",
        })
        .to_string(),
        json!({
            "type": "agent-preset/selected",
            "seq": 0,
            "time": 1,
            "data": {"agentPreset": "code"},
        })
        .to_string(),
    ]
    .join("\n");
    let artifact = SessionFormatCatalog::shared()
        .restore_jsonl(&text)
        .expect("v3 migrates");
    assert_eq!(artifact.header["agentPreset"], json!("code"));
    assert_eq!(artifact.header["version"], json!(4));
    assert_eq!(
        selection_events(&artifact.events),
        [json!({"agentPreset": "code"})]
    );
}

#[test]
fn the_historical_catalog_applies_the_same_preset_rename_at_v3() {
    // The rename belongs to the v2→v3 edge, so the prerequisite catalog —
    // which stops at v3 — produces the same renamed selections.
    let artifact = SessionFormatCatalog::historical()
        .restore_jsonl(&seeded_preset_source(2))
        .expect("migrates to v3");
    assert_eq!(artifact.header["version"], json!(3));
    assert_eq!(artifact.header["agentPreset"], json!("ptc"));
    assert_eq!(selection_events(&artifact.events), expected_selections());
}
