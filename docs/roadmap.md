# Roadmap

Planned work. Nothing here is a commitment until it is released.

## Number of MGVI samples varying with τ

Different `MGVIOptions` per `advance` chunk already work; a schedule for them would make it
convenient.

## Halting on the local ESS

Stop once the local ESS fraction has levelled off at 1.

## Complex data

Complex-valued data (e.g. radio interferometry) in `gaussian` and the NIFTy.re adaptor.

## Checkpointing during a run

Save the sampler state and finished samples as a run progresses (every few τ steps, or as
batches finish), so a crashed run restarts from its last checkpoint instead of from scratch.
Restarting is exact already (see the user guide); this adds the saving loop.
