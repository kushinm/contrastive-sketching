# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A workspace for CLIP-guided sketch-generation approaches. Each approach lives in its own top-level directory with its own environment, README, and `CLAUDE.md`. There is no shared root package — treat each approach directory as self-contained.

## Approaches

- **[clipasso/](clipasso/)** — Contrastive CLIPasso: CLIP-guided Bézier-stroke sketch generation, extended with a contrastive mode (target + distractor → sketch that repels the distractor). See [clipasso/CLAUDE.md](clipasso/CLAUDE.md) for its environment, commands, and architecture.

When working inside an approach directory, `cd` into it first — commands, relative paths, and installation steps documented in that approach's own `CLAUDE.md`/`README.md` assume it is the working directory.
