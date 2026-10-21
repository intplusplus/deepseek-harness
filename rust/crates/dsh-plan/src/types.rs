//! The plan domain: deployment guidance, the folded unit state, the projection's
//! wire value, and the process-local pending selection.
//!
//! Ported from `packages/plan/plan-mode/src/types.ts` and `resolveConfig` in
//! `src/index.ts`. The split between [`PlanUnitState`] and [`PlanProjection`] is
//! the TypeScript split between the host fold and the client-facing value:
//! `wanted`, `running`, and `activeAtLastHeader` are fold bookkeeping, while
//! `active` and `pending` are the only fields a carrier may expose.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::brand::CommandId;
use super::text::js_trim;

/// Section name of the deployment-owned plan guidance.
pub const PLAN_POLICY_SECTION: &str = "plan:policy";

/// The `PLAN_POLICY` position in the centrally allocated section table.
pub const PLAN_POLICY_ORDER: f64 = 500.0;

/// Deployment-owned plan guidance.
///
/// `section` is required and non-empty: a composition that omits it fails at
/// load rather than silently rendering no plan policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlanModeConfig {
    /// Guidance rendered as the `plan:policy` prompt section while plan mode is
    /// active.
    pub section: String,
}

impl PlanModeConfig {
    /// A config carrying the given guidance.
    pub fn new(section: impl Into<String>) -> Self {
        Self {
            section: section.into(),
        }
    }

    /// Validate one raw config, failing with the TypeScript load errors.
    ///
    /// The order is the TypeScript order: the section must be a string, then
    /// non-empty after trimming, then the only allowed key. Unknown keys keep
    /// the caller's insertion order and are joined with `, `.
    pub fn resolve(value: &Value) -> Result<Self, PlanModeConfigError> {
        let Some(fields) = value.as_object() else {
            return Err(PlanModeConfigError::MissingSection);
        };
        let Some(section) = fields.get("section") else {
            return Err(PlanModeConfigError::MissingSection);
        };
        let Some(section) = section.as_str() else {
            return Err(PlanModeConfigError::MissingSection);
        };
        if js_trim(section).is_empty() {
            return Err(PlanModeConfigError::EmptySection);
        }
        let unknown: Vec<&str> = fields
            .keys()
            .map(String::as_str)
            .filter(|key| *key != "section")
            .collect();
        if !unknown.is_empty() {
            return Err(PlanModeConfigError::UnknownKeys {
                keys: unknown.join(", "),
            });
        }
        Ok(Self {
            section: section.to_string(),
        })
    }
}

/// Why a raw plan config was rejected at load.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanModeConfigError {
    /// The section is missing or is not a string.
    #[error("PlanModeConfig needs a string `section`")]
    MissingSection,
    /// The section is empty after trimming.
    #[error("PlanModeConfig needs a non-empty `section`")]
    EmptySection,
    /// The config carries keys outside the declared surface.
    #[error("PlanModeConfig has unknown key(s) {keys} — config is {{ section }}")]
    UnknownKeys {
        /// The unknown keys, in caller order, joined with `, `.
        keys: String,
    },
}

/// The latest plan command awaiting its paired settlement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanRunning {
    /// Identity of the running `/plan` command.
    pub command_id: CommandId,
    /// The mode that command selected.
    pub wanted: bool,
}

/// Host state folded from the plan command lifecycle and `plan/mode` events.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanUnitState {
    /// Logged plan mode: the last `plan/mode`, inactive before the first.
    pub active: bool,
    /// The selection's target mode; `None` when no selection is outstanding.
    pub wanted: Option<bool>,
    /// The latest plan command awaiting its paired settlement.
    pub running: Option<PlanRunning>,
    /// Active state recorded by the latest `request/header`, or `None`.
    pub active_at_last_header: Option<bool>,
}

/// The plan projection's wire value.
///
/// `active` is the logged state in force; `pending` is true while a logged
/// `/plan` selection targets a state other than `active`, has not failed
/// through its paired `command/done`, and no later `plan/mode` recorded it.
/// Capability absence is the key's absence, never a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanProjection {
    /// Logged plan mode.
    pub active: bool,
    /// Whether a logged selection differs from the logged mode.
    pub pending: bool,
}

/// What the controller reports for one agent: logged mode plus any process-local
/// selection awaiting the next accepted in-turn pre-step.
///
/// `pending` is omitted when no selection is waiting, matching the TypeScript
/// `{ active }` / `{ active, pending }` union.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlanStatus {
    /// Logged plan mode.
    pub active: bool,
    /// The selected target, when one is waiting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<bool>,
}

/// A selection awaiting the next accepted in-turn pre-step.
///
/// `narrate` is true for user selections and false for the exit tool, whose
/// result already narrates the transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingIntent {
    /// The selected target mode.
    pub active: bool,
    /// Whether the flush should inject a user-switch notice.
    pub narrate: bool,
}
