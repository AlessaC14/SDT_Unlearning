# Amendment 03 Task C runner

The separate Task C runner is implemented and has not made any model calls. Prior source and run artifacts remain unchanged.

The runner rebuilds its request from the authoritative recorded question associations, schedules at most four distinct DIM per ENTITY, inlines each DIM value space, and requires an explicit structural-ceiling declaration. The selected core schedules 257 proposals across 145 ENTITY; 141 ENTITY have fewer than four available DIM.

Before rejecting an otherwise valid proposal whose real span is inexact, the runner deterministically selects the closest cited correct-text span and validates again. Each attempt is recorded in a separate deterministic repair log without storing either old or replacement values there. LLM raw logging and usage/cost logging remain unchanged.

The revised full-population threshold is `1.4344827586206899`, derived as 80% of the Task A ceiling `1.793103448275862`. The paired pilot threshold is separately `1.12`, derived as 80% of that exact ten-ENTITY composition's ceiling `1.4`. This explicit population-specific treatment avoids evaluating the paired pilot against an impossible threshold while retaining the full-population gate.

The paired comparison emits proposal count, proposals per ENTITY, accepted count, acceptance among parsed proposals, accepted attributes per ENTITY, deterministic repairs, and insufficient-DIM declarations. Execution remains blocked pending manager review.
