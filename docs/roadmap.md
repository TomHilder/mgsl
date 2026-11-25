# Roadmap

Planned work. Nothing here is a commitment until it is released.

## First release

A minimal MGSL sampler for standardised (white-prior) models, usable with NIFTy.re and other
JAX models without depending on NIFTy.re, with the ability to resume a run to a larger τ.

## Checkpointing during a run

Save the sampler state and finished samples as a run progresses (every few τ steps, or as paths
finish), so a crashed run restarts from its last checkpoint instead of from scratch.
