# An egui app for prompting local offline LLMs.

<p align="center">
  <img alt="Example prompt" src="media/prompt.gif" height="450" width="370">
</p>

## Description

`coze` is an [`egui`](https://github.com/emilk/egui) application for prompting local
offline LLMs using the Huggingface [`candle`](https://github.com/huggingface/candle)
crate.

Currently it supports the following quantized models:

- [Mistral Instruct 7B v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [Mistral 7B v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Hugging Face Zephyr 7B β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- [StableLM 2 Zephyr 1.6B](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)

The first time a model is used its weights are downloaded from Huggingface and cached
to the `~/.cache/coze` folder for later use.

The current version supports:

- Prompt history navigation with fuzzy matching.
- History persistence across runs.
- Token generation modes.
- Copy prompts and replies to clipboard.
- Light/Dark mode.

See the app `Help` menu for usage details.

## Installation

The latest version can be installed by getting the binaries for Linux, MacOS, and
Windows from the [releases page][github-releases], or by using `cargo`:

```sh
cargo install --locked coze
```

To build locally (debug build may be very slow):

```bash
git clone https://github.com/vincev/coze
cd coze
cargo r --release
```

[github-releases]: https://github.com/vincev/coze/releases/latest
