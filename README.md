# LangExtract llama-cpp-python Provider

A provider plugin for LangExtract that supports llama-cpp-python models.

## Installation

```bash
pip install -e .
```

## Supported Model IDs

Model ID using the format as such:

1. HuggingFace repo with file name: `hf:<hf_repo_id>:<filename>`
2. HuggingFace repo without file name: `hf:<hf_repo_id>`, in this case the filename will be `None`
3. Local file: `file:<path_to_model>`

`hf_repo_id` is existing huggingface model repository.

## Usage

Using HuggingFace repository; this will call `.from_pretrained(...)`

```python
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="hf:MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:*Q4_K_M.gguf",
    provider="LlamaCppLanguageModel", # optional as hf: will resolve to the model
    provider_kwargs=dict(
        n_gpu_layers=-1,
        n_ctx=4096,
        verbose=False,
        completion_kwargs=dict(
            temperature=1.1,
            seed=42,
        ),
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

Using local file path

```python
import langextract as lx

config = lx.factory.ModelConfig(
    model_id="file:Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
    provider="LlamaCppLanguageModel", # optional as file: will resolve to the model
    provider_kwargs=dict(
        ...
    ),
)

...
```

## OpenAI compatible Web Server

When using llama-cpp-python server (or llama.cpp), you can use `OpenAILanguageModel` in the provider field as they implement OpenAI compatible web server.

To set this up, choose `OpenAILanguageModel` as the provider and supply the server’s base URL and an API key (any value) in `provider_kwargs`. The `model_id` field is optional.

```python
config = lx.factory.ModelConfig(
    model_id="local", # optional
    provider="OpenAILanguageModel", # Explicitly set the provider to `OpenAILanguageModel`
    provider_kwargs=dict(
        base_url="http://localhost:8000/v1/",
        api_key="llama-cpp", # Any value; mandatory
    ),
)

result = lx.extract(
    config=config,
    ...
)
```

## Development

1. Install in development mode: `pip install -e .`
2. Run tests: `python test_plugin.py`
3. Build package: `python -m build`
4. Publish to PyPI: `twine upload dist/*`
