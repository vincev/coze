# Model weights

This is a quantized version of the Stable LM 2 Zephyr 1.6B model, see the
[model card](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b)
on Hugging Face for a model description and license.

This quantized version has been generated from the
[model.safetensors](https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b/tree/main) weights file
using the [`Candle tensor-tools`](https://github.com/huggingface/candle/blob/main/candle-core/examples/tensor-tools.rs) application:

```shell
tensor-tools quantize --quantization q4_1 --out-file stablelm-2-zephyr-1_6b-Q4_1.gguf model.safetensors
```
