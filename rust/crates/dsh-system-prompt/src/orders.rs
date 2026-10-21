//! Centrally allocated placement tables.
//!
//! TypeScript keeps these as `const SECTION_ORDERS` / `const CONTEXT_ORDERS`
//! objects and derives `PromptSectionOrderName` / `PromptContextOrderName`
//! from `keyof`. Rust gets one enum per table: the numeric values are the
//! contract every repository section positions itself with, and the enum
//! makes a typo'd placement name a compile error instead of a runtime
//! `undefined` order.

#![forbid(unsafe_code)]

/// Name of a centrally allocated prompt-section position.
///
/// The variant order matches the TypeScript table declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PromptSectionOrderName {
    HarnessIdentity,
    DeploymentPersonaPrefix,
    PlanPolicy,
    TeamPolicy,
    PtcOnly,
    FileReference,
    ToolBash,
    ToolPwsh,
    ToolRead,
    ToolWrite,
    ToolEdit,
    ToolGlob,
    ToolGrep,
    ToolJobs,
    ToolPty,
    ToolWebSearch,
    ToolWebFetch,
    ToolLsp,
    ToolSessionQuery,
    ToolGoal,
    ToolWorkflow,
    ToolRalph,
    ToolSubagent,
    ToolReport,
    ToolComputerUse,
    McpServers,
    ToolsSdk,
    DeliverableFileReferences,
    StructuredOutput,
    HarnessSource,
    WebSurface,
    DeploymentPersonaSuffix,
}

impl PromptSectionOrderName {
    /// Every allocated position, in table order.
    pub const ALL: &'static [Self] = &[
        Self::HarnessIdentity,
        Self::DeploymentPersonaPrefix,
        Self::PlanPolicy,
        Self::TeamPolicy,
        Self::PtcOnly,
        Self::FileReference,
        Self::ToolBash,
        Self::ToolPwsh,
        Self::ToolRead,
        Self::ToolWrite,
        Self::ToolEdit,
        Self::ToolGlob,
        Self::ToolGrep,
        Self::ToolJobs,
        Self::ToolPty,
        Self::ToolWebSearch,
        Self::ToolWebFetch,
        Self::ToolLsp,
        Self::ToolSessionQuery,
        Self::ToolGoal,
        Self::ToolWorkflow,
        Self::ToolRalph,
        Self::ToolSubagent,
        Self::ToolReport,
        Self::ToolComputerUse,
        Self::McpServers,
        Self::ToolsSdk,
        Self::DeliverableFileReferences,
        Self::StructuredOutput,
        Self::HarnessSource,
        Self::WebSurface,
        Self::DeploymentPersonaSuffix,
    ];

    /// The TypeScript key and numeric order of this position.
    fn entry(self) -> (&'static str, f64) {
        match self {
            Self::HarnessIdentity => ("HARNESS_IDENTITY", -1000.0),
            Self::DeploymentPersonaPrefix => ("DEPLOYMENT_PERSONA_PREFIX", 0.0),
            Self::PlanPolicy => ("PLAN_POLICY", 500.0),
            Self::TeamPolicy => ("TEAM_POLICY", 600.0),
            Self::PtcOnly => ("PTC_ONLY", 800.0),
            Self::FileReference => ("FILE_REFERENCE", 900.0),
            Self::ToolBash => ("TOOL_BASH", 1000.0),
            Self::ToolPwsh => ("TOOL_PWSH", 1010.0),
            Self::ToolRead => ("TOOL_READ", 1100.0),
            Self::ToolWrite => ("TOOL_WRITE", 1200.0),
            Self::ToolEdit => ("TOOL_EDIT", 1300.0),
            Self::ToolGlob => ("TOOL_GLOB", 1400.0),
            Self::ToolGrep => ("TOOL_GREP", 1500.0),
            Self::ToolJobs => ("TOOL_JOBS", 1600.0),
            Self::ToolPty => ("TOOL_PTY", 1700.0),
            Self::ToolWebSearch => ("TOOL_WEB_SEARCH", 2000.0),
            Self::ToolWebFetch => ("TOOL_WEB_FETCH", 2100.0),
            Self::ToolLsp => ("TOOL_LSP", 2200.0),
            Self::ToolSessionQuery => ("TOOL_SESSION_QUERY", 2300.0),
            Self::ToolGoal => ("TOOL_GOAL", 2400.0),
            Self::ToolWorkflow => ("TOOL_WORKFLOW", 2600.0),
            Self::ToolRalph => ("TOOL_RALPH", 2700.0),
            Self::ToolSubagent => ("TOOL_SUBAGENT", 2800.0),
            Self::ToolReport => ("TOOL_REPORT", 2900.0),
            Self::ToolComputerUse => ("TOOL_COMPUTER_USE", 3000.0),
            Self::McpServers => ("MCP_SERVERS", 3100.0),
            Self::ToolsSdk => ("TOOLS_SDK", 5000.0),
            Self::DeliverableFileReferences => ("DELIVERABLE_FILE_REFERENCES", 9000.0),
            Self::StructuredOutput => ("STRUCTURED_OUTPUT", 9900.0),
            Self::HarnessSource => ("HARNESS_SOURCE", 10000.0),
            Self::WebSurface => ("WEB_SURFACE", 10100.0),
            Self::DeploymentPersonaSuffix => ("DEPLOYMENT_PERSONA_SUFFIX", 10200.0),
        }
    }

    /// The TypeScript key of this position (`"TOOL_BASH"`, for example).
    pub fn name(self) -> &'static str {
        self.entry().0
    }

    /// The numeric sort order of this position.
    pub fn order(self) -> f64 {
        self.entry().1
    }
}

/// Name of a centrally allocated runtime-context position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PromptContextOrderName {
    WorkingDirectory,
    SandboxPolicy,
    ApprovalPolicy,
    SubagentDelegation,
    TeamIdentity,
    HarnessSource,
    WebSurface,
}

impl PromptContextOrderName {
    /// Every allocated position, in table order.
    pub const ALL: &'static [Self] = &[
        Self::WorkingDirectory,
        Self::SandboxPolicy,
        Self::ApprovalPolicy,
        Self::SubagentDelegation,
        Self::TeamIdentity,
        Self::HarnessSource,
        Self::WebSurface,
    ];

    /// The TypeScript key and numeric order of this position.
    fn entry(self) -> (&'static str, f64) {
        match self {
            Self::WorkingDirectory => ("WORKING_DIRECTORY", 100.0),
            Self::SandboxPolicy => ("SANDBOX_POLICY", 110.0),
            Self::ApprovalPolicy => ("APPROVAL_POLICY", 115.0),
            Self::SubagentDelegation => ("SUBAGENT_DELEGATION", 120.0),
            Self::TeamIdentity => ("TEAM_IDENTITY", 125.0),
            // Process-specific paths and endpoints stay out of the cached
            // system prompt.
            Self::HarnessSource => ("HARNESS_SOURCE", 200.0),
            Self::WebSurface => ("WEB_SURFACE", 210.0),
        }
    }

    /// The TypeScript key of this position.
    pub fn name(self) -> &'static str {
        self.entry().0
    }

    /// The numeric sort order of this position.
    pub fn order(self) -> f64 {
        self.entry().1
    }
}

/// The deployment persona prefix's section name.
///
/// Exported because a composition can replace this slot — an agent preset
/// shadows the deployment's persona with its own — and both sides naming
/// the same section is what makes the replacement work rather than
/// duplicate.
pub const PERSONA_PREFIX_SECTION: &str = "deployment:persona-prefix";

/// Deployment persona suffix section name shared by global and scoped
/// contributions.
pub const PERSONA_SUFFIX_SECTION: &str = "deployment:persona-suffix";

/// Section name of the fixed harness identity the plugin registers by
/// default.
pub const HARNESS_IDENTITY_SECTION: &str = "harness:identity";

/// Text of the fixed harness identity section.
pub const HARNESS_IDENTITY_TEXT: &str = "You are an AI agent powered by DeepSeek Harness.";

/// Reserved [`Config::tool_order`](crate::Config::tool_order) marker for
/// unlisted tools.
pub const TOOL_ORDER_REST: &str = "<unlisted-tools>";
