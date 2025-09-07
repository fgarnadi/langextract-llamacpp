# LangExtract llama-cpp-python Provider

A provider plugin for LangExtract that supports llama-cpp-python models.

## Installation

```bash
pip install langextract-llamacpp
```

## Supported Model IDs

Model ID using the format as such:

1. HuggingFace repo with file name: `hf:<hf_repo_id>:<filename>`
2. HuggingFace repo without file name: `hf:<hf_repo_id>`, in this case the filename will be `None`
3. Local file: `file:<path_to_model>`

`hf_repo_id` is existing huggingface model repository.

## Usage

Using HuggingFace repository; this will call `Llama.from_pretrained(...)`.

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

model = lx.factory.create_model(config)

result = lx.extract(
    model=model,
    text_or_documents="Your input text",
    prompt_description="Extract entities",
    examples=[...],
)
```

Using local file path; this will call `Llama(...)`.

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

For `provider_kwargs` refer to [documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__) for `Llama` class.

For `completion_kwargs` refer to [documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion) for `crate_chat_completion` method.

## OpenAI compatible Web Server

When using llama-cpp-python server (or llama.cpp), you can use `OpenAILanguageModel` in the provider field as they implement OpenAI compatible web server.

To set this up, choose `OpenAILanguageModel` as the provider and supply the server’s base URL and an API key (any value) in `provider_kwargs`. The `model_id` field is optional.

```python
config = lx.factory.ModelConfig(
    model_id="local", # optional
    provider="OpenAILanguageModel", # explicitly set the provider to `OpenAILanguageModel`
    provider_kwargs=dict(
        base_url="http://localhost:8000/v1/",
        api_key="llama-cpp", # any value; mandatory
    ),
)

model = lx.factory.create_model(config)

result = lx.extract(
    model=model,
    ...
)
```

## Development

1. Install in development mode: `uv pip install -e .`
2. Run tests: `uv run test_plugin.py`
3. Build package: `uv build`
4. Publish to PyPI: `uv publish`
