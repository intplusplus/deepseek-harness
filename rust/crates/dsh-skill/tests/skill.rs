//! Behavior of the pure skill catalog layer, taken from the TypeScript
//! registry's public contracts and rendering tests.

use dsh_skill::{
    escape_text, is_skill_name, merge_layers, render_skill_content, resolve_layer, runtime_layer,
    sort_summaries, validate_candidate, validate_definition, IndexedCandidate, SkillCandidate,
    SkillDefinition, SkillError, SkillInvocationForm, SkillInvocationPolicy, SkillInvocationSource,
    SkillInvocationSourceKind, SkillName, SkillRank, SkillRegistration, SkillResourceBase,
    SkillSource, SkillSummary, BUNDLED_SKILL_RANK, RUNTIME_PROVIDER, RUNTIME_RANK,
};

fn name(value: &str) -> SkillName {
    SkillName::new(value).expect("test skill names are valid")
}

fn summary(skill_name: &str, provider: &str) -> SkillSummary {
    SkillSummary {
        name: name(skill_name),
        path: None,
        description: format!("{skill_name} description"),
        when_to_use: None,
        invocation: SkillInvocationPolicy::default(),
        source: SkillSource::new("test"),
        provider: provider.to_string(),
        resource_base: None,
    }
}

fn candidate(skill_name: &str, provider: &str, rank: f64) -> SkillCandidate<()> {
    SkillCandidate::new(
        summary(skill_name, provider),
        SkillRank::new(rank).expect("finite"),
        (),
    )
}

fn definition(skill_name: &str) -> SkillDefinition {
    SkillDefinition::new(summary(skill_name, "memory"), "Body.")
}

#[test]
fn accepts_the_public_kebab_case_grammar() {
    for valid in ["a", "pdf-tools", "2026-report", "a1-b2-c3", "9"] {
        assert!(is_skill_name(valid), "{valid}");
    }
}

#[test]
fn rejects_every_shape_outside_the_grammar() {
    for invalid in [
        "", "-a", "a-", "a--b", "A", "a_b", "a b", "a.b", "ü", "a-1--b",
    ] {
        assert!(!is_skill_name(invalid), "{invalid}");
        assert!(matches!(
            SkillName::new(invalid),
            Err(SkillError::InvalidName { .. })
        ));
    }
}

#[test]
fn a_wire_name_is_validated_during_deserialization() {
    let error = serde_json::from_str::<SkillName>("\"Bad Name\"").expect_err("must reject");
    assert!(error.to_string().contains("invalid skill name"));
}

#[test]
fn a_rank_must_be_finite() {
    assert!(matches!(
        SkillRank::new(f64::NAN),
        Err(SkillError::NonFiniteRank { .. })
    ));
    assert!(matches!(
        SkillRank::new(f64::INFINITY),
        Err(SkillError::NonFiniteRank { .. })
    ));
}

#[test]
fn renders_a_directory_based_skill_with_the_shared_wrapper() {
    let mut skill = SkillDefinition::new(summary("demo-skill", "memory"), "Do the thing.");
    skill.summary.resource_base = Some(SkillResourceBase::Directory {
        path: "/tmp/demo".into(),
    });

    assert_eq!(
        render_skill_content(&skill),
        [
            "<skill_content name=\"demo-skill\">",
            "<skill_resources>",
            "Base directory for this skill: /tmp/demo",
            "Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.",
            "</skill_resources>",
            "",
            "<skill_instructions>",
            "Do the thing.",
            "</skill_instructions>",
            "</skill_content>",
        ]
        .join("\n")
    );
}

#[test]
fn renders_url_and_opaque_resource_hints() {
    let mut url_skill = definition("url-skill");
    url_skill.summary.resource_base = Some(SkillResourceBase::Url {
        url: "https://example.test/base/".into(),
    });
    let url = render_skill_content(&url_skill);
    assert!(url.contains("Base URL for this skill: https://example.test/base/"));
    assert!(url.contains(
        "Resolve relative URLs mentioned by this skill against the base URL before using them."
    ));

    let mut opaque_skill = definition("opaque-skill");
    opaque_skill.summary.resource_base = Some(SkillResourceBase::Opaque {
        description: "archive <bundle>".into(),
    });
    let opaque = render_skill_content(&opaque_skill);
    assert!(opaque.contains("Resources for this skill: archive &lt;bundle&gt;"));
    assert!(opaque.contains("Load referenced resources only as needed."));
}

#[test]
fn falls_back_to_the_provider_hint_without_a_resource_base() {
    let mut skill = definition("provider-skill");
    skill.summary.provider = "remote <hub>".into();
    let text = render_skill_content(&skill);
    assert!(
        text.contains("Resources for this skill are managed by provider \"remote &lt;hub&gt;\".")
    );
    assert!(text.contains("Load referenced resources only as needed."));
}

#[test]
fn keeps_the_body_verbatim_while_prose_is_escaped() {
    let skill = SkillDefinition::new(
        summary("safe-name", "memory"),
        "Keep </skill_content> and <tags> as-is.",
    );
    let text = render_skill_content(&skill);
    assert!(text.contains("<skill_content name=\"safe-name\">"));
    assert!(text.contains("Keep </skill_content> and <tags> as-is."));
    assert_eq!(escape_text("x\"&<y>"), "x\"&amp;&lt;y&gt;");
}

#[test]
fn escapes_prose_without_double_escaping_ampersands() {
    assert_eq!(escape_text("a&<b>"), "a&amp;&lt;b&gt;");
    assert_eq!(escape_text("&amp;"), "&amp;amp;");
}

#[test]
fn summaries_serialize_in_the_typescript_field_order() {
    let mut value = summary("demo-skill", "memory");
    value.path = Some("/skills/demo/SKILL.md".into());
    value.when_to_use = Some("When testing.".into());
    value.resource_base = Some(SkillResourceBase::Directory {
        path: "/tmp/demo".into(),
    });

    assert_eq!(
        serde_json::to_string(&value).expect("serializes"),
        "{\"name\":\"demo-skill\",\"path\":\"/skills/demo/SKILL.md\",\"description\":\"demo-skill description\",\"whenToUse\":\"When testing.\",\"invocation\":{\"modelInvocable\":true,\"userInvocable\":true},\"source\":\"test\",\"provider\":\"memory\",\"resourceBase\":{\"kind\":\"directory\",\"path\":\"/tmp/demo\"}}"
    );
}

#[test]
fn optional_summary_fields_are_omitted_not_null() {
    let value = summary("demo-skill", "memory");
    let encoded = serde_json::to_string(&value).expect("serializes");
    assert!(!encoded.contains("path"));
    assert!(!encoded.contains("whenToUse"));
    assert!(!encoded.contains("resourceBase"));
    assert!(!encoded.contains("null"));
}

#[test]
fn resource_bases_write_the_kind_discriminator_first() {
    assert_eq!(
        serde_json::to_string(&SkillResourceBase::Directory {
            path: "/tmp".into()
        })
        .unwrap(),
        "{\"kind\":\"directory\",\"path\":\"/tmp\"}"
    );
    assert_eq!(
        serde_json::to_string(&SkillResourceBase::Url {
            url: "https://x.test/".into()
        })
        .unwrap(),
        "{\"kind\":\"url\",\"url\":\"https://x.test/\"}"
    );
    assert_eq!(
        serde_json::to_string(&SkillResourceBase::Opaque {
            description: "archive".into()
        })
        .unwrap(),
        "{\"kind\":\"opaque\",\"description\":\"archive\"}"
    );
}

#[test]
fn invocation_sources_write_the_typescript_shape() {
    let source = SkillInvocationSource::new(name("demo-skill"));
    assert_eq!(
        serde_json::to_string(&source).unwrap(),
        "{\"kind\":\"skill-invocation\",\"name\":\"demo-skill\",\"form\":\"instructions\"}"
    );
    let decoded: SkillInvocationSource =
        serde_json::from_str(&serde_json::to_string(&source).unwrap()).unwrap();
    assert_eq!(decoded.kind, SkillInvocationSourceKind::SkillInvocation);
    assert_eq!(decoded.form, SkillInvocationForm::Instructions);
}

#[test]
fn rank_resolution_prefers_rank_then_provider_order_then_local_order() {
    let entries = vec![
        // Same name: the first provider at equal rank wins.
        IndexedCandidate::new(
            candidate("alpha", "p0", 100.0).map_locator(|_| "alpha-first"),
            0,
            0,
        ),
        IndexedCandidate::new(
            candidate("alpha", "p1", 100.0).map_locator(|_| "alpha-second"),
            1,
            0,
        ),
        // Lower rank wins even from a later provider.
        IndexedCandidate::new(
            candidate("beta", "p0", 50.0).map_locator(|_| "beta-low"),
            0,
            0,
        ),
        IndexedCandidate::new(
            candidate("beta", "p1", 10.0).map_locator(|_| "beta-high"),
            1,
            0,
        ),
        // Equal rank and provider: the earlier local order wins.
        IndexedCandidate::new(
            candidate("gamma", "p0", 10.0).map_locator(|_| "gamma-local-two"),
            0,
            2,
        ),
        IndexedCandidate::new(
            candidate("gamma", "p0", 10.0).map_locator(|_| "gamma-local-one"),
            0,
            1,
        ),
    ];

    let winners = resolve_layer(entries);
    let names: Vec<_> = winners
        .iter()
        .map(|entry| entry.summary.name.as_str())
        .collect();
    assert_eq!(names, ["gamma", "beta", "alpha"]);
    assert_eq!(winners[0].locator, "gamma-local-one");
    assert_eq!(winners[1].locator, "beta-high");
    assert_eq!(winners[2].locator, "alpha-first");
}

#[test]
fn runtime_rank_sits_between_project_and_bundled_candidates() {
    let runtime = runtime_layer(vec![definition("zeta"), definition("alpha")]);
    assert_eq!(
        runtime
            .iter()
            .map(|entry| entry.candidate.summary.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "zeta"]
    );
    assert!(runtime.iter().all(|entry| entry.provider_order == -1));
    assert!(runtime
        .iter()
        .all(|entry| entry.candidate.rank.get() == RUNTIME_RANK));
    assert_eq!(runtime[0].local_order, 0);
    assert_eq!(runtime[1].local_order, 1);

    let entries = vec![
        IndexedCandidate::new(
            candidate("alpha", RUNTIME_PROVIDER, RUNTIME_RANK).map_locator(|_| "runtime"),
            -1,
            0,
        ),
        IndexedCandidate::new(
            candidate("alpha", "project", 100.0).map_locator(|_| "project"),
            0,
            0,
        ),
        IndexedCandidate::new(
            candidate("alpha", "bundled", BUNDLED_SKILL_RANK).map_locator(|_| "bundled"),
            1,
            0,
        ),
    ];
    let winners = resolve_layer(entries);
    assert_eq!(winners.len(), 1);
    assert_eq!(winners[0].locator, "project");
}

#[test]
fn a_nearer_layer_wins_a_duplicate_name_regardless_of_rank() {
    let global = resolve_layer(vec![
        IndexedCandidate::new(
            candidate("shared", "global", 1.0).map_locator(|_| "global"),
            0,
            0,
        ),
        IndexedCandidate::new(
            candidate("global-only", "global", 1.0).map_locator(|_| "global-only"),
            0,
            1,
        ),
    ]);
    let scope = resolve_layer(vec![
        IndexedCandidate::new(
            candidate("shared", "scope", 999.0).map_locator(|_| "scope"),
            0,
            0,
        ),
        IndexedCandidate::new(
            candidate("scope-only", "scope", 999.0).map_locator(|_| "scope-only"),
            0,
            1,
        ),
    ]);

    let merged = merge_layers(vec![global, scope]);
    let names: Vec<_> = merged
        .iter()
        .map(|entry| entry.summary.name.as_str())
        .collect();
    assert_eq!(names, ["shared", "global-only", "scope-only"]);
    assert_eq!(merged[0].locator, "scope");
}

#[test]
fn summaries_sort_by_name() {
    let mut summaries = vec![
        summary("zeta", "p"),
        summary("alpha", "p"),
        summary("midway", "p"),
    ];
    sort_summaries(&mut summaries);
    assert_eq!(
        summaries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "midway", "zeta"]
    );
}

#[test]
fn provider_candidates_must_own_their_name_and_description() {
    let candidate_value = candidate("demo", "owner", 1.0);
    assert!(validate_candidate(&candidate_value, "owner").is_ok());
    assert!(matches!(
        validate_candidate(&candidate_value, "other"),
        Err(SkillError::ProviderMismatch { .. })
    ));

    let mut undescribed = candidate("demo", "owner", 1.0);
    undescribed.summary.description.clear();
    assert!(matches!(
        validate_candidate(&undescribed, "owner"),
        Err(SkillError::EmptyDescription { .. })
    ));
}

#[test]
fn definitions_require_a_description() {
    let mut value = definition("demo");
    assert!(validate_definition(&value).is_ok());
    value.summary.description.clear();
    assert!(matches!(
        validate_definition(&value),
        Err(SkillError::EmptyDescription { .. })
    ));
}

#[test]
fn registrations_default_to_both_surfaces_and_the_runtime_provider() {
    let registration = SkillRegistration {
        name: name("demo"),
        description: "Demo.".into(),
        when_to_use: None,
        path: None,
        source: SkillSource::new(SkillSource::RUNTIME),
        resource_base: None,
        content: "Body.".into(),
        metadata: None,
        invocation: None,
        provider: None,
    };
    let definition = registration.into_definition().expect("valid registration");
    assert_eq!(definition.summary.provider, RUNTIME_PROVIDER);
    assert_eq!(
        definition.summary.invocation,
        SkillInvocationPolicy {
            model_invocable: true,
            user_invocable: true
        }
    );

    let invalid = SkillRegistration {
        name: name("demo"),
        description: String::new(),
        when_to_use: None,
        path: None,
        source: SkillSource::new(SkillSource::RUNTIME),
        resource_base: None,
        content: "Body.".into(),
        metadata: None,
        invocation: None,
        provider: None,
    };
    assert!(matches!(
        invalid.into_definition(),
        Err(SkillError::EmptyDescription { .. })
    ));
}

#[test]
fn a_definition_flattens_its_summary_fields_on_the_wire() {
    let mut value = definition("demo");
    value.metadata = Some(serde_json::from_str("{\"author\":\"test\"}").unwrap());
    let encoded = serde_json::to_string(&value).expect("serializes");
    assert!(encoded
        .starts_with("{\"name\":\"demo\",\"description\":\"demo description\",\"invocation\":"));
    assert!(encoded.contains("\"content\":\"Body.\""));
    assert!(encoded.ends_with("\"metadata\":{\"author\":\"test\"}}"));

    let decoded: SkillDefinition = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, value);
}
