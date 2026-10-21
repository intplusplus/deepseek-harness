//! Time-context rules: timestamps, injected zones, DST boundaries, browser
//! policy, refresh scheduling, projection replay, and exact model text.

use dsh_context::time_context::{
    derive_browser_time_zone_context, derive_from_sources, fold_time_context, format_duration,
    format_timestamp, observe_event, render_browser_time_zone_context, render_time_context,
    request_messages, resolve_config, should_inject, validate_refresh_interval_ms,
    BrowserTimeZoneContext, Config, TimeContextError, TimeContextObservation, TimeContextState,
    TimeReading, TimeZoneResolver, ZoneTable, ZoneTableError, DEFAULT_REFRESH_INTERVAL_MS,
};
use dsh_llm::{
    AssistantMessage, ContentBlock, ExtensionSource, MessageId, MessageSource, UserMessage,
};
use dsh_session::{CoreEvent, SessionEvent, TurnEndReason};
use serde_json::{json, Map, Value};

const SHANGHAI_EPOCH: i64 = 1_784_077_261_000;
const NEW_YORK_DST_START: i64 = 1_772_953_200_000;
const NEW_YORK_DST_END: i64 = 1_793_512_800_000;

fn user_message(id: &str, text: &str) -> UserMessage {
    UserMessage::new(MessageId::new(id), vec![ContentBlock::text(text)])
}

fn extension_source(kind: &str, fields: Value) -> MessageSource {
    let object = fields.as_object().expect("extension fields object").clone();
    MessageSource::Extension(ExtensionSource::new(kind, object))
}

fn browser_source(rpc_id: &str, zone: &str) -> MessageSource {
    extension_source("user", json!({ "rpcId": rpc_id, "clientTimeZone": zone }))
}

fn user_with_source(id: &str, text: &str, source: MessageSource) -> UserMessage {
    UserMessage::with_source(MessageId::new(id), vec![ContentBlock::text(text)], source)
}

fn zones() -> ZoneTable {
    let mut table = ZoneTable::new();
    table
        .insert_fixed("Asia/Shanghai", 8 * 3_600)
        .expect("fixed Shanghai");
    table
        .insert_fixed("America/New_York", -5 * 3_600)
        .expect("fixed New York");
    table
        .insert_transition("America/New_York", NEW_YORK_DST_START, -4 * 3_600)
        .expect("DST start");
    table
        .insert_transition("America/New_York", NEW_YORK_DST_END, -5 * 3_600)
        .expect("DST end");
    table
        .insert_fixed("Asia/Kolkata", 5 * 3_600 + 30 * 60)
        .expect("fixed Kolkata");
    table.insert_alias("Etc/UTC", "UTC").expect("UTC alias");
    table
}

fn reading(
    now_ms: i64,
    previous_ms: Option<i64>,
    step: u64,
    browser: BrowserTimeZoneContext,
) -> TimeReading {
    TimeReading {
        now_ms,
        turn: 3,
        step,
        previous_ms,
        time_zone: "Asia/Shanghai".to_string(),
        offset_seconds: 8 * 3_600,
        browser,
    }
}

#[test]
fn timestamp_formats_intl_samples() {
    assert_eq!(
        format_timestamp(SHANGHAI_EPOCH, 8 * 3_600, "Asia/Shanghai"),
        "2026-07-15T09:01:01+08:00[Asia/Shanghai]"
    );
    assert_eq!(
        format_timestamp(1_783_987_200_000, -4 * 3_600, "America/New_York"),
        "2026-07-13T20:00:00-04:00[America/New_York]"
    );
    assert_eq!(
        format_timestamp(1_783_987_200_000, 0, "UTC"),
        "2026-07-14T00:00:00+00:00[UTC]"
    );
    assert_eq!(
        format_timestamp(SHANGHAI_EPOCH + 999, 8 * 3_600, "Asia/Shanghai"),
        "2026-07-15T09:01:01+08:00[Asia/Shanghai]"
    );
}

#[test]
fn timestamp_formats_fractional_and_second_offsets() {
    assert_eq!(
        format_timestamp(0, 5 * 3_600 + 30 * 60, "Asia/Kolkata"),
        "1970-01-01T05:30:00+05:30[Asia/Kolkata]"
    );
    assert_eq!(
        format_timestamp(0, 5 * 3_600 + 30 * 60 + 15, "Historical/Zone"),
        "1970-01-01T05:30:15+05:30:15[Historical/Zone]"
    );
    assert_eq!(
        format_timestamp(-1, 0, "UTC"),
        "1969-12-31T23:59:59+00:00[UTC]"
    );
}

#[test]
fn zone_table_resolves_dst_boundaries_exactly() {
    let table = zones();
    assert_eq!(
        table.offset_seconds("America/New_York", NEW_YORK_DST_START - 1),
        Some(-5 * 3_600)
    );
    assert_eq!(
        table.offset_seconds("America/New_York", NEW_YORK_DST_START),
        Some(-4 * 3_600)
    );
    assert_eq!(
        table.offset_seconds("America/New_York", NEW_YORK_DST_END - 1),
        Some(-4 * 3_600)
    );
    assert_eq!(
        table.offset_seconds("America/New_York", NEW_YORK_DST_END),
        Some(-5 * 3_600)
    );
    assert_eq!(
        format_timestamp(
            NEW_YORK_DST_START - 1,
            table
                .offset_seconds("America/New_York", NEW_YORK_DST_START - 1)
                .expect("pre-transition offset"),
            "America/New_York"
        ),
        "2026-03-08T01:59:59-05:00[America/New_York]"
    );
    assert_eq!(
        format_timestamp(
            NEW_YORK_DST_START,
            table
                .offset_seconds("America/New_York", NEW_YORK_DST_START)
                .expect("post-transition offset"),
            "America/New_York"
        ),
        "2026-03-08T03:00:00-04:00[America/New_York]"
    );
    assert_eq!(
        format_timestamp(
            NEW_YORK_DST_END,
            table
                .offset_seconds("America/New_York", NEW_YORK_DST_END)
                .expect("fall-back offset"),
            "America/New_York"
        ),
        "2026-11-01T01:00:00-05:00[America/New_York]"
    );
}

#[test]
fn zone_table_fixed_offset_and_alias() {
    let table = zones();
    assert_eq!(table.canonical("Etc/UTC"), Some("UTC".to_string()));
    assert_eq!(table.canonical("UTC"), Some("UTC".to_string()));
    assert_eq!(table.canonical("Mars/Olympus"), None);
    assert_eq!(table.offset_seconds("Asia/Kolkata", 0), Some(19_800));
    assert_eq!(table.offset_seconds("Asia/Kolkata", i64::MIN), Some(19_800));
}

#[test]
fn zone_table_rejects_invalid_edits() {
    let mut table = ZoneTable::new();
    assert_eq!(
        table.insert_fixed("Bad", 24 * 3_600),
        Err(ZoneTableError::OffsetOutOfRange {
            offset_seconds: 24 * 3_600
        })
    );
    assert_eq!(
        table.insert_transition("Missing", 0, 0),
        Err(ZoneTableError::UnknownZone {
            zone: "Missing".to_string()
        })
    );
    assert_eq!(
        table.insert_alias("Alias", "Missing"),
        Err(ZoneTableError::UnknownZone {
            zone: "Missing".to_string()
        })
    );
}

#[test]
fn config_applies_defaults_and_resolves_aliases() {
    let table = zones();
    let resolved = resolve_config(&Config::default(), Some("UTC"), &table).expect("default config");
    assert_eq!(resolved.time_zone(), "UTC");
    assert_eq!(resolved.refresh_interval_ms(), DEFAULT_REFRESH_INTERVAL_MS);

    let alias = Config {
        time_zone: Some("Etc/UTC".to_string()),
        refresh_interval_ms: Some(0.0),
    };
    let resolved = resolve_config(&alias, None, &table).expect("alias config");
    assert_eq!(resolved.time_zone(), "UTC");
    assert_eq!(resolved.refresh_interval_ms(), 0);
}

#[test]
fn config_rejects_invalid_and_unresolved_zones() {
    let table = zones();
    let invalid = Config {
        time_zone: Some("Mars/Olympus".to_string()),
        refresh_interval_ms: None,
    };
    assert_eq!(
        resolve_config(&invalid, Some("UTC"), &table),
        Err(TimeContextError::InvalidTimeZone {
            zone: "\"Mars/Olympus\"".to_string()
        })
    );
    assert_eq!(
        resolve_config(&Config::default(), None, &table),
        Err(TimeContextError::UnresolvedSystemZone)
    );
    assert_eq!(
        resolve_config(&Config::default(), Some("Mars/Olympus"), &table),
        Err(TimeContextError::UnresolvedSystemZone)
    );
}

#[test]
fn browser_zone_derivation_is_sorted_unique_and_missing() {
    let table = zones();
    let shanghai = browser_source("rpc-1", "Asia/Shanghai");
    let new_york = browser_source("rpc-2", "America/New_York");
    let duplicate = browser_source("rpc-3", "Asia/Shanghai");
    let context = derive_from_sources([&shanghai, &new_york, &duplicate], &table)
        .expect("valid browser zones");
    assert_eq!(
        context,
        BrowserTimeZoneContext::Mixed {
            time_zones: vec!["America/New_York".to_string(), "Asia/Shanghai".to_string()]
        }
    );
    assert_eq!(
        derive_from_sources([&shanghai], &table).expect("resolved"),
        BrowserTimeZoneContext::Resolved {
            time_zone: "Asia/Shanghai".to_string()
        }
    );
    assert_eq!(
        derive_from_sources(std::iter::empty(), &table).expect("missing"),
        BrowserTimeZoneContext::Missing
    );
    let ordinary = MessageSource::User;
    assert_eq!(
        derive_from_sources([&ordinary], &table).expect("ordinary user"),
        BrowserTimeZoneContext::Missing
    );
}

#[test]
fn browser_zone_rejects_invalid_syntax_unsupported_and_noncanonical() {
    let table = zones();
    let invalid = browser_source("rpc-1", "Not A Zone");
    assert_eq!(
        derive_from_sources([&invalid], &table),
        Err(TimeContextError::BrowserZoneSyntax {
            value: "\"Not A Zone\"".to_string()
        })
    );
    let unsupported = browser_source("rpc-1", "Mars/Olympus");
    assert_eq!(
        derive_from_sources([&unsupported], &table),
        Err(TimeContextError::BrowserZoneUnsupported {
            value: "\"Mars/Olympus\"".to_string()
        })
    );
    let alias = browser_source("rpc-1", "Etc/UTC");
    assert_eq!(
        derive_from_sources([&alias], &table),
        Err(TimeContextError::BrowserZoneNotCanonical {
            value: "\"Etc/UTC\"".to_string()
        })
    );
    let canonical = browser_source("rpc-1", "UTC");
    assert_eq!(
        derive_from_sources([&canonical], &table).expect("canonical UTC"),
        BrowserTimeZoneContext::Resolved {
            time_zone: "UTC".to_string()
        }
    );
    let missing_rpc = extension_source("user", json!({ "clientTimeZone": "UTC" }));
    assert_eq!(
        derive_from_sources([&missing_rpc], &table).expect("no rpc id"),
        BrowserTimeZoneContext::Missing
    );
}

#[test]
fn render_time_context_is_exact() {
    let browser = BrowserTimeZoneContext::Resolved {
        time_zone: "Asia/Shanghai".to_string(),
    };
    let first = reading(
        SHANGHAI_EPOCH,
        Some(SHANGHAI_EPOCH - 90_061_000),
        1,
        browser,
    );
    assert_eq!(
        render_time_context(&first),
        "Time sampled while preparing turn 3, step 1: 2026-07-15T09:01:01+08:00[Asia/Shanghai]\nBrowser time zone for this request: Asia/Shanghai. Interpret otherwise-unqualified dates and times in this zone.\nElapsed since the preceding model-visible message: 1d 1h 1m 1s."
    );

    let later = reading(
        SHANGHAI_EPOCH,
        Some(SHANGHAI_EPOCH - 61_000),
        2,
        BrowserTimeZoneContext::Resolved {
            time_zone: "Asia/Shanghai".to_string(),
        },
    );
    assert_eq!(
        render_time_context(&later),
        "Time sampled while preparing turn 3, step 2: 2026-07-15T09:01:01+08:00[Asia/Shanghai]\nBrowser time zone for this request: Asia/Shanghai. Interpret otherwise-unqualified dates and times in this zone.\nElapsed since the preceding step context: 1m 1s."
    );
}

#[test]
fn render_time_context_handles_mixed_missing_and_unavailable() {
    let mixed = reading(
        SHANGHAI_EPOCH,
        None,
        1,
        BrowserTimeZoneContext::Mixed {
            time_zones: vec!["America/New_York".to_string(), "Asia/Shanghai".to_string()],
        },
    );
    assert_eq!(
        render_time_context(&mixed),
        "Time sampled while preparing turn 3, step 1: 2026-07-15T09:01:01+08:00[Asia/Shanghai]\nBrowser time zone for this request: mixed [\"America/New_York\",\"Asia/Shanghai\"]. Ask the user to clarify otherwise-unqualified dates and times.\nElapsed since the preceding model-visible message: unavailable."
    );
    let missing = reading(SHANGHAI_EPOCH, None, 1, BrowserTimeZoneContext::Missing);
    assert_eq!(
        render_browser_time_zone_context(&missing.browser),
        "Browser time zone for this request: unavailable. Ask the user to clarify otherwise-unqualified dates and times."
    );
}

#[test]
fn duration_clamps_negative_and_omits_zero_units() {
    assert_eq!(format_duration(0), "0s");
    assert_eq!(format_duration(-1), "0s");
    assert_eq!(format_duration(999), "0s");
    assert_eq!(format_duration(61_000), "1m 1s");
    assert_eq!(format_duration(3_600_000), "1h 0s");
    assert_eq!(format_duration(86_400_000), "1d 0s");
    assert_eq!(format_duration(90_061_000), "1d 1h 1m 1s");
}

#[test]
fn refresh_guard_obeys_interval_and_backward_clock() {
    assert!(should_inject(0, 100, Some(100)));
    assert!(should_inject(600_000, 100, None));
    assert!(!should_inject(600_000, 100, Some(100)));
    assert!(!should_inject(600_000, 699_999, Some(100_000)));
    assert!(should_inject(600_000, 700_000, Some(100_000)));
    assert!(should_inject(600_000, 99, Some(100)));
}

#[test]
fn refresh_interval_diagnostics_match_typescript() {
    assert_eq!(validate_refresh_interval_ms(0.0), Ok(0));
    assert_eq!(
        validate_refresh_interval_ms(9_007_199_254_740_991.0),
        Ok(9_007_199_254_740_991)
    );
    for (value, rendered) in [
        (-1.0, "-1"),
        (0.5, "0.5"),
        (f64::INFINITY, "Infinity"),
        (f64::NEG_INFINITY, "-Infinity"),
        (f64::NAN, "NaN"),
        (9_007_199_254_740_992.0, "9007199254740992"),
    ] {
        assert_eq!(
            validate_refresh_interval_ms(value),
            Err(TimeContextError::InvalidRefreshInterval {
                value: rendered.to_string()
            })
        );
    }
}

#[test]
fn projection_replay_matches_typescript_fold() {
    let mut state = TimeContextState::default();
    state = fold_time_context(&state, &TimeContextObservation::TurnBoundary);
    state = fold_time_context(
        &state,
        &TimeContextObservation::UserMessage {
            time_ms: 100,
            injected: false,
        },
    );
    assert_eq!(state.last_message_time, Some(100));
    assert_eq!(state.last_injection_time, None);
    state = fold_time_context(
        &state,
        &TimeContextObservation::UserMessage {
            time_ms: 200,
            injected: true,
        },
    );
    assert_eq!(state.last_message_time, Some(200));
    assert_eq!(state.last_injection_time, Some(200));
    assert_eq!(state.last_turn_injection_time, Some(200));
    state = fold_time_context(
        &state,
        &TimeContextObservation::OtherMessage { time_ms: 300 },
    );
    state = fold_time_context(&state, &TimeContextObservation::TurnBoundary);
    assert_eq!(state.last_message_time, Some(300));
    assert_eq!(state.last_injection_time, Some(200));
    assert_eq!(state.last_turn_injection_time, None);
    assert_eq!(
        serde_json::to_string(&state).expect("state JSON"),
        "{\"lastMessageTime\":300,\"lastInjectionTime\":200,\"lastTurnInjectionTime\":null}"
    );
}

#[test]
fn observe_event_classifies_core_events() {
    let ordinary = SessionEvent::user_message(user_message("user-1", "hello"));
    assert_eq!(
        observe_event(&ordinary, 10),
        TimeContextObservation::UserMessage {
            time_ms: 10,
            injected: false
        }
    );
    let injected = SessionEvent::user_message(user_with_source(
        "time-1",
        "reading",
        extension_source("time-context", json!({})),
    ));
    assert_eq!(
        observe_event(&injected, 20),
        TimeContextObservation::UserMessage {
            time_ms: 20,
            injected: true
        }
    );
    assert_eq!(
        observe_event(&SessionEvent::turn_start(1), 30),
        TimeContextObservation::TurnBoundary
    );
    assert_eq!(
        observe_event(&SessionEvent::turn_end(1, TurnEndReason::Completed), 31),
        TimeContextObservation::TurnBoundary
    );
    assert_eq!(
        observe_event(
            &SessionEvent::assistant_message(
                1,
                1,
                AssistantMessage::new(
                    MessageId::new("assistant-1"),
                    vec![ContentBlock::text("answer")],
                    "provider",
                    "model",
                ),
                Vec::new(),
                None,
                false,
            ),
            40
        ),
        TimeContextObservation::OtherMessage { time_ms: 40 }
    );
    assert_eq!(
        observe_event(&SessionEvent::step_start(1, 1), 50),
        TimeContextObservation::Ignored
    );
    let developer = dsh_llm::DeveloperMessage::new(
        MessageId::new("developer-1"),
        vec![ContentBlock::tool_addition("read")],
        MessageSource::User,
    );
    assert_eq!(
        observe_event(&SessionEvent::developer_message(1, 1, developer, None), 60),
        TimeContextObservation::Ignored
    );
    let core = CoreEvent::StepEnd { turn: 1, step: 1 };
    assert_eq!(
        observe_event(&SessionEvent::Core(core), 70),
        TimeContextObservation::Ignored
    );
}

#[test]
fn request_messages_collects_entered_and_proposed_for_open_turn() {
    let first = user_message("first", "first");
    let second = user_message("second", "second");
    let proposed = user_message("proposed", "proposed");
    let events = vec![
        SessionEvent::user_message(first),
        SessionEvent::turn_start(1),
        SessionEvent::user_message(second),
    ];
    let collected = request_messages(&events, 1, &[&proposed]);
    assert_eq!(collected.len(), 2);
    assert_eq!(collected[0].id.as_str(), "second");
    assert_eq!(collected[1].id.as_str(), "proposed");

    let collected = request_messages(&events, 2, &[&proposed]);
    assert_eq!(collected.len(), 1);
    assert_eq!(collected[0].id.as_str(), "proposed");
}

#[test]
fn reading_uses_unique_browser_zone_and_falls_back() {
    let table = zones();
    let config = resolve_config(&Config::default(), Some("UTC"), &table).expect("config");
    let shanghai = config
        .reading(
            &table,
            SHANGHAI_EPOCH,
            3,
            1,
            None,
            BrowserTimeZoneContext::Resolved {
                time_zone: "Asia/Shanghai".to_string(),
            },
        )
        .expect("browser reading");
    assert_eq!(shanghai.time_zone, "Asia/Shanghai");
    assert_eq!(shanghai.offset_seconds, 8 * 3_600);
    assert!(shanghai
        .render()
        .contains("2026-07-15T09:01:01+08:00[Asia/Shanghai]"));

    let fallback = config
        .reading(
            &table,
            SHANGHAI_EPOCH,
            3,
            1,
            None,
            BrowserTimeZoneContext::Mixed {
                time_zones: vec!["America/New_York".to_string(), "Asia/Shanghai".to_string()],
            },
        )
        .expect("fallback reading");
    assert_eq!(fallback.time_zone, "UTC");
    assert_eq!(fallback.offset_seconds, 0);

    let unknown = config.reading(
        &table,
        SHANGHAI_EPOCH,
        3,
        1,
        None,
        BrowserTimeZoneContext::Resolved {
            time_zone: "Mars/Olympus".to_string(),
        },
    );
    assert_eq!(
        unknown,
        Err(TimeContextError::MissingOffset {
            zone: "Mars/Olympus".to_string(),
            epoch_ms: SHANGHAI_EPOCH
        })
    );
}

#[test]
fn resolved_config_exposes_validated_values() {
    let table = zones();
    let resolved = resolve_config(
        &Config {
            time_zone: None,
            refresh_interval_ms: Some(1.0),
        },
        Some("UTC"),
        &table,
    )
    .expect("resolved");
    assert_eq!(resolved.refresh_interval_ms(), 1);
}

#[test]
fn user_message_without_zone_source_is_missing() {
    let table = zones();
    let message = user_message("plain", "hello");
    assert_eq!(
        derive_browser_time_zone_context(&[message], &table).expect("missing"),
        BrowserTimeZoneContext::Missing
    );
    let mut fields = Map::new();
    fields.insert("rpcId".to_string(), Value::String("rpc-1".to_string()));
    fields.insert(
        "clientTimeZone".to_string(),
        Value::String("UTC".to_string()),
    );
    let source = MessageSource::Extension(ExtensionSource::new("other", fields));
    let message = user_with_source("other", "hello", source);
    assert_eq!(
        derive_browser_time_zone_context(&[message], &table).expect("other kind"),
        BrowserTimeZoneContext::Missing
    );
}

#[test]
fn time_zone_resolver_trait_is_object_safe() {
    let table = zones();
    let resolver: &dyn TimeZoneResolver = &table;
    assert_eq!(resolver.canonical("UTC"), Some("UTC".to_string()));
    assert_eq!(resolver.offset_seconds("UTC", 0), Some(0));
}
