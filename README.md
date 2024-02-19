# An egui app for prompting a local offline LLM.

<p align="center">
  <img alt="Model loading" src="media/loading.gif" height="400">
  <img alt="Example prompt" src="media/prompt.gif" height="400">
</p>

## Description

`coze` is a small [`egui`](https://github.com/emilk/egui) application for prompting
a local offline LLM using the Huggingface [`candle`](https://github.com/huggingface/candle)
crate.

Currently it uses a [quantized version](https://huggingface.co/vincevas/coze-stablelm-2-1_6b)
of the [StableLM 2 Zephyr 1.6B](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
model that is a relatively small model that is fun to use.

The current version supports:

- Prompt history navigation with fuzzy matching.
- History persistence across runs.
- Token generation modes.
- Copy prompts and replies to clipboard.
- Light/Dark mode.

See the app `Edit/Config` menu for usage details.

## Installation

The latest version of `coze` can be installed or updated with `cargo install`:

```sh
cargo install --locked coze
```

or by getting the binaries generated by the release Github action for Linux, macOS,
and Windows in the [releases page][github-releases].

[github-releases]: https://github.com/vincev/coze/releases/latest

The first time it runs it will download the model weights from Huggingface to the
`~/.cache/coze` folder.

[github-releases]: https://github.com/vincev/coze/releases/latest

To build locally (debug build may be very slow):

```bash
git clone https://github.com/vincev/coze
cd coze
cargo r --release
```