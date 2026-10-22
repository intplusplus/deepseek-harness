//! The unified plugin-host assembly layer.
//!
//! One [`SessionScope`] mounts every host component the harness needs — the
//! retry lifecycle, the PTC sandbox hosts, the skill host, and the interaction
//! services — onto a single [`dsh_plugin::Scope`], and guarantees that tearing
//! the scope down unwinds them in **LIFO** order, mirroring the TypeScript
//! Cordis `ctx.effect()` drain semantics.
//!
//! ## How the LIFO drain works
//!
//! [`dsh_plugin::Scope`] already holds its registrations as reversible effects
//! and pops them in reverse registration order on drop. This crate adds a
//! single, unified drain list to the *assembly* (not the kernel) so that hosts
//! whose cleanup is not a plain service unregistration — an `async` `dispose()`
//! or a `!Send` `Rc<RefCell>` host — still unwind in the same LIFO order as the
//! `Send` hosts registered through the kernel.
//!
//! Each `mount_*` wraps the host's own teardown (`RetryLifecycle::dispose`,
//! `PersistentHost::destroy_all`, dropping the `SkillHost`) in a guard pushed
//! onto that list. `SessionScope::drop` pops the list last-registered-first,
//! then the inner `Scope` drops its own service guards. The two phases together
//! are a single LIFO unwind.
//!
//! ## The `SkillHost` seam
//!
//! [`dsh_skill::host::SkillHost`] is `Rc<RefCell<…>>` and therefore **not**
//! `Send`. It cannot be stored in the kernel's `Send` effect list, nor
//! registered as a `Service` (which requires `Send + Sync`). The assembly still
//! drains it — its guard lives on the unified list, which is `!Send` for
//! exactly this reason. Consequently **`SessionScope` is `!Send`**. A
//! `Send`-only composition can drop `mount_skill` (the other five hosts are all
//! `Send`) or, in `dsh-skill`, back the host with an `Arc<Mutex<…>>` backend so
//! it can be mounted through the kernel like the rest. This crate does not
//! modify `dsh-skill`; it documents the trade-off and keeps the door open.

use std::future::Future;
use std::sync::{Arc, Mutex};

use dsh_agent::Agent;
use dsh_interaction::approval::{ApprovalPolicy, AuditedApprovalService};
use dsh_interaction::questions::UserQuestionService;
use dsh_plugin::{Context, Scope};
use dsh_ptc_host::{HostConfig, LaunchConfig, PersistentHost, PtcHost};
use dsh_retry::RetryLifecycle;
use dsh_session::Session;
use dsh_skill::host::SkillHost;
use dsh_tools::approval::{ApprovalResult, ApprovalService, ConstApprovalService};

use crate::error::HostAssemblyError;

/// A trace callback invoked (synchronously) when a host's drain guard fires.
///
/// The assembly calls it with the host's stable name so a test — or a
/// production deployment — can observe the unwind order without the hosts
/// themselves being observable. Stored as `Arc<dyn Fn>` so the same tracer can
/// be captured by both `Send` and `!Send` drain guards.
type Tracer = Arc<dyn Fn(&str) + Send + Sync>;

/// Dispatch an `async` drain to the tokio runtime when one is running.
///
/// The synchronous part of every host teardown (aborting the retry lifetime,
/// cancelling retained processes) has already taken effect by the time the
/// guard fires; awaiting the returned future only settles in-flight work. If
/// there is no runtime in scope — e.g. a synchronous teardown — the await is
/// skipped and the already-applied cancellation stands. This keeps `Drop`
/// infallible and `Send`, matching Cordis' synchronous effect disposers.
fn dispatch_drain(future: impl Future<Output = ()> + Send + 'static) {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn(future);
    }
}

/// The composition root: every host mounted here drains LIFO on drop.
///
/// See the [module docs](crate::scope) for the lifetime contract and the
/// `SkillHost` `Send` seam.
pub struct SessionScope {
    /// The kernel scope every host registers into.
    scope: Scope,
    /// Unified LIFO drain list. `!Send` because `SkillHost` (an `Rc<RefCell>`)
    /// drains here; that is what makes `SessionScope` itself `!Send`.
    drains: Vec<Box<dyn FnOnce()>>,
    /// Drain-order tracer, shared into every guard.
    tracer: Tracer,
    /// The session agent identity this scope is assembled for, if supplied.
    agent: Option<Agent>,
}

impl SessionScope {
    /// Open an assembly over a fresh kernel context with a no-op tracer.
    pub fn new() -> Self {
        Self::with_tracer(Arc::new(|_name| {}))
    }

    /// Open an assembly with an explicit drain tracer.
    ///
    /// The tracer is called once per mounted host, synchronously, at the moment
    /// its drain guard fires — in LIFO order. Tests use it to assert the unwind
    /// sequence; production deployments use it for ordered shutdown logging.
    pub fn with_tracer(tracer: Tracer) -> Self {
        Self {
            scope: Scope::new(Arc::new(Context::new())),
            drains: Vec::new(),
            tracer,
            agent: None,
        }
    }

    /// The kernel context every host is mounted into.
    pub fn context(&self) -> &Arc<Context> {
        self.scope.context()
    }

    /// Record the session agent identity this scope is assembled for.
    pub fn with_agent(mut self, agent: Agent) -> Self {
        self.agent = Some(agent);
        self
    }

    /// The session agent identity, if one was supplied.
    pub fn agent(&self) -> Option<&Agent> {
        self.agent.as_ref()
    }

    /// Mount the retry lifecycle.
    ///
    /// The lifecycle is registered as a retrievable `Service` and its drain
    /// guard calls [`RetryLifecycle::dispose`], aborting the lifetime signal and
    /// settling any in-flight recoveries.
    pub fn mount_retry(&mut self, lifecycle: RetryLifecycle) {
        self.scope.set(Arc::new(lifecycle.clone()));
        let tracer = Arc::clone(&self.tracer);
        self.drains.push(Box::new(move || {
            tracer("retry");
            let drain = lifecycle.dispose();
            dispatch_drain(drain);
        }));
    }

    /// Mount the one-shot PTC sandbox host.
    ///
    /// Each run spawns and tears down its own child, so the host holds no
    /// retained state between runs; the drain guard only unregisters the
    /// service. Construction is offline — no child process is spawned here.
    pub fn mount_ptc(
        &mut self,
        config: HostConfig,
        launch: LaunchConfig,
    ) -> Result<(), HostAssemblyError> {
        let host = PtcHost::new(config, launch).map_err(|e| HostAssemblyError::mount("ptc", e))?;
        self.scope.set(Arc::new(host));
        let tracer = Arc::clone(&self.tracer);
        self.drains.push(Box::new(move || {
            tracer("ptc");
        }));
        Ok(())
    }

    /// Mount the persistent PTC sandbox pool.
    ///
    /// The drain guard calls [`PersistentHost::destroy_all`], tearing down every
    /// retained child process. Construction is offline.
    pub fn mount_persistent(
        &mut self,
        config: HostConfig,
        launch: LaunchConfig,
    ) -> Result<(), HostAssemblyError> {
        let host = PersistentHost::new(config, launch)
            .map_err(|e| HostAssemblyError::mount("persistent", e))?;
        let arc = Arc::new(host);
        self.scope.set(Arc::clone(&arc));
        let tracer = Arc::clone(&self.tracer);
        self.drains.push(Box::new(move || {
            tracer("persistent");
            let retained = Arc::clone(&arc);
            dispatch_drain(async move {
                let _ = retained.destroy_all().await;
            });
        }));
        Ok(())
    }

    /// Mount the skill host.
    ///
    /// `SkillHost` is `Rc<RefCell<…>>` and therefore `!Send`; its guard lives on
    /// the unified drain list (see [crate::scope]). It cannot be registered as
    /// a kernel `Service`, so it is drained (dropped) here and is not
    /// retrievable through [`Context::require`].
    pub fn mount_skill<L>(&mut self, skill: SkillHost<L>)
    where
        L: Clone + 'static,
    {
        let tracer = Arc::clone(&self.tracer);
        self.drains.push(Box::new(move || {
            tracer("skill");
            drop(skill);
        }));
    }

    /// Mount the audited approval service.
    ///
    /// Composes [`AuditedApprovalService`] from the supplied session, answerer,
    /// and policy, and registers it as a retrievable `Service`. Its drain guard
    /// only unregisters the service; the service holds an `Arc<Mutex<Session>>`
    /// that outlives the scope by reference counting.
    pub fn mount_approval(
        &mut self,
        session: Arc<Mutex<Session>>,
        answerer: Arc<dyn ApprovalService>,
        policy: ApprovalPolicy,
    ) {
        let service = AuditedApprovalService::new(session, answerer, policy);
        self.scope.set(Arc::new(service));
        let tracer = Arc::clone(&self.tracer);
        self.drains.push(Box::new(move || {
            tracer("approval");
        }));
    }

    /// Mount the user-question service as a retrievable `Service`.
    pub fn mount_question(&mut self, service: UserQuestionService) {
        self.scope.set(Arc::new(service));
        let tracer = Arc::clone(&self.tracer);
        self.drains.push(Box::new(move || {
            tracer("question");
        }));
    }

    /// Convenience: mount approval with a constant allow answerer and the
    /// default `Ask` policy over a fresh in-memory session.
    pub fn mount_approval_default(&mut self) {
        let session = Arc::new(Mutex::new(Session::create(
            dsh_session::SessionId::new("dsh-host-session").expect("valid session id"),
            dsh_session::SessionMeta::default(),
            0,
        )));
        let answerer: Arc<dyn ApprovalService> =
            Arc::new(ConstApprovalService::new(ApprovalResult::AllowedOnce));
        self.mount_approval(session, answerer, ApprovalPolicy::Ask);
    }
}

impl Default for SessionScope {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SessionScope {
    fn drop(&mut self) {
        // LIFO: the most recently mounted host drains first.
        while let Some(drain) = self.drains.pop() {
            drain();
        }
        // `self.scope` drops here, unwinding the kernel's own service guards in
        // reverse registration order — after every host drain above.
    }
}
