# Roadmap

Planned work. Nothing is definitely confirmed just a vague plan.

## Number of MGVI samples varying with τ

Different `MGVIOptions` per `advance`. Can do awkwardly right now. Maybe not even needed but I suspect it could help on very large problems.

## Halting on the local ESS

Stop once the local ESS fraction has levelled off at 1. Currently only available as a post-hoc check.

## Complex data

Complex-valued data (e.g. radio interferometry) in `gaussian` and the NIFTy.re adaptor. A good first contribution if you are interested in being involved.

## Checkpointing during a run

Save the sampler state and finished samples as a run progresses (every few τ steps, or as
batches finish), so a crashed run restarts from its last checkpoint instead of losing everything.
