# LangExtract LlamaCpp Provider

A provider plugin for LangExtract that supports LlamaCpp models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

Model ID using the format as such:

1. With file name: `hf:<hf_repo_id>:<filename>`
2. Without file name: `hf:<hf_repo_id>`, in this case the filename will be `None`.

`hf_repo_id` is existing huggingface model repository.

## Usage

```python
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="hf:MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:*Q4_K_M.gguf",
    provider="LlamaCppLanguageModel", # optional as hf: will resolve to the model
    provider_kwargs=dict(
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
        max_workers=2,
    ),
)

result = lx.extract(
    config=config,
    text="Your document here",
    model_id="llamacpp-model",
    prompt_description="Extract entities",
    examples=[...]
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`
