# CosmoSliderWeb (fork)

**[Open the live version](https://oleg-savchenko.github.io/CosmoSliderWeb/)**

This is a fork of **CosmoSliderWeb** (Nygaard et al.), an educational, browser-based tool for exploring
how cosmological parameters shape the CMB power spectra and the matter power
spectrum, using neural network emulators running client-side via TensorFlow.js.

This fork adds a matter power spectrum (Pk) mode — linear and nonlinear — alongside
the original CMB temperature/polarization/lensing spectra, plus a few UI
improvements (dual-axis angular/spatial scale references, reference grids,
persisting the selected spectrum across page reloads).

## Credits

This project builds directly on the work of others:

- **CosmoSliderWeb** (the original tool this repo is forked from) — Nygaard et al.
  Original repository: https://github.com/AarhusCosmology/CosmoSliderWeb
  Paper: https://arxiv.org/abs/2601.16919

- **CosmoPower-JAX** — the matter power spectrum (linear and nonlinear boost)
  emulator and its trained model weights used in this fork were taken from this
  project.
  Code: https://github.com/dpiras/cosmopower-jax/tree/main
  Paper: https://arxiv.org/abs/2305.06347

- **mpk_compilation** — the observational/reference matter power spectrum
  compilation data shown on the Pk plots is taken from this project.
  Repository: https://github.com/marius311/mpk_compilation/tree/master
  Paper: https://arxiv.org/abs/1905.08103

OpenAI's CODEX and Anthropic's Claude Code have been heavily used for coding assistance when creating this extension.

## License

See [LICENSE](LICENSE).
