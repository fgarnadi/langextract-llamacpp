"""Provider implementation for LlamaCpp."""

import ctypes
from typing import cast

from langextract.core import base_model, exceptions, types
from langextract.providers import router
from llama_cpp import (
    CreateChatCompletionResponse,
    Llama,
    llama_log_callback,
    llama_log_set,
)

LLAMACPP_PATTERNS = (
    r"^hf",
    r"^file",
)


@router.register(*LLAMACPP_PATTERNS, priority=10)
class LlamaCppLanguageModel(base_model.BaseLanguageModel):
    """LangExtract provider for llama-cpp-python.

    This provider handles model IDs matching: ['^hf', '^file']
    """

    def __init__(
        self,
        model_id: str,
        max_workers: int = 1,
        verbose: bool = False,
        **kwargs,
    ):
        """Initialize the LlamaCppProvider provider.

        Args:
            model_id: The model identifier.
            max_workers: The maximum number of workers to use for parallel
                inference.
            **kwargs: Additional provider-specific parameters.
        """

        super().__init__()

        self.model_id = model_id
        self.max_workers = max_workers
        self.verbose = verbose

        self._completion_kwargs = kwargs.pop("completion_kwargs", {})
        self._completion_kwargs["stream"] = False  # Disable stream

        self._client_kwargs = kwargs

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the llama-cpp client based on the model_id pattern.

        Parses the model_id and creates the appropriate Llama client instance.
        Supported patterns:
            - "hf:repo_id:filename"
            - "hf:repo_id"
            - "file:model_path"

        Raises:
            lx.exceptions.InferenceConfigError: If the model_id does not match
                a known pattern.
        """
        match self.model_id.split(":"):
            case ("hf", repo_id, filename):
                self._client = Llama.from_pretrained(
                    repo_id=repo_id,
                    filename=filename,
                    verbose=self.verbose,
                    **self._client_kwargs,
                )
            case ("hf", repo_id):
                self._client = Llama.from_pretrained(
                    repo_id=repo_id,
                    verbose=self.verbose,
                    **self._client_kwargs,
                )
            case ("file", model_path):
                self._client = Llama(
                    model_path=model_path,
                    verbose=self.verbose,
                    **self._client_kwargs,
                )
            case _:
                raise exceptions.InferenceConfigError("Can't find `model_id` configuration pattern.")

    def _suppress_logger(self):
        """Suppress llama-cpp logger.
        Reference : https://github.com/abetlen/llama-cpp-python/issues/478

        But not working as intended.
        """

        def noop_logger(*_, **__):
            pass

        llama_log_set(llama_log_callback(noop_logger), ctypes.c_void_p())

    def _process_single_prompt(self, prompt: str) -> types.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            response = self._client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self._completion_kwargs,
            )

            response = cast(CreateChatCompletionResponse, response)
            result = response["choices"][0]["message"]["content"]

            return types.ScoredOutput(score=1.0, output=result)
        except Exception as e:
            raise exceptions.InferenceRuntimeError(f"llama-cpp error: {str(e)}", original=e) from e

    def infer(self, batch_prompts, **_):
        """Run inference on a batch of prompts.

        Args:
            batch_prompts: List of prompts to process.
            **kwargs: Additional inference parameters.

        Yields:
            Lists of ScoredOutput objects, one per prompt.
        """
        # TODO : use batched inference for len(batch_prompts) > 1
        # https://github.com/abetlen/llama-cpp-python/issues/771
        # currently only support sequential processing
        for prompt in batch_prompts:
            result = self._process_single_prompt(prompt)
            yield [result]
