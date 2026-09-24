# Public-clock origin guard: argument and scope

This allocation experiment inherits the fixed-cap extended-transcript argument
in [predictive_filter_argument.md](predictive_filter_argument.md). It does not
introduce a new privacy definition or endpoint guarantee.

Let q = B/(8H), with cap C = 4(2H−1). The early phase has release/test costs q;
later steps cost 4q per primitive. The phase is fixed by public time from the
first request; the first request is always in the early phase. A first fresh
anchor costs one primitive; reuse costs the test; refresh costs test + release.
Before reading GPS reserve one primitive initially or two thereafter. Stop
private reads permanently if reservation would exceed C.

At a fixed extended transcript, phase, remaining budget and stopping decisions
agree across neighboring secret trajectories. Each primitive contributes at
most its epsilon times current Euclidean distance to the log likelihood ratio.
Every path costs at most Cq=(2H−1)B/(2H), hence the marginal public transcript
has that ideal-kernel bound under D∞ and a fixed public clock. This remains
true after the public-context selector and progress planner. No assumption
that protecting the first minute protects the origin from later observations
is made. Known future routes, secret stopping times, person-level composition
and floating-point exact privacy remain outside this statement.

Two matched PublicAnchorModels share state grid/prior/context but differ in
emission epsilon and its full-domain normalizer. Switching that emission does
not reset the private anchor or the belief. Its mobility posterior remains an
approximation and is not a calibrated attacker posterior.
