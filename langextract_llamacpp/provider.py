"""Provider implementation for LlamaCpp."""

import ctypes
from typing import cast

import langextract as lx
from llama_cpp import (
    CreateChatCompletionResponse,
    Llama,
    llama_log_callback,
    llama_log_set,
)


@lx.providers.registry.register(r"^hf", priority=10)
class LlamaCppLanguageModel(lx.inference.BaseLanguageModel):
    """LangExtract provider for LlamaCpp.

    This provider handles model IDs matching: ['^hf']
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
            max_workers: The maximum number of workers to use for parallel inference.
            **kwargs: Additional provider-specific parameters.
        """

        super().__init__()

        ids = model_id.split(":")
        self.repo_id = ids[1]
        self.filename = ids[2] if len(ids) > 2 else None
        self.max_workers = max_workers

        self._completion_kwargs = kwargs.pop("completion_kwargs", {})
        self._completion_kwargs["stream"] = False  # Disable stream

        self._client_kwargs = kwargs
        self._client = Llama.from_pretrained(
            repo_id=self.repo_id,
            filename=self.filename,
            verbose=verbose,
            **self._client_kwargs,
        )

    def _suppress_logger(self):
        """Suppress llama-cpp logger.
        Reference : https://github.com/abetlen/llama-cpp-python/issues/478

        But not working as intended.
        """

        def noop_logger(*args, **kwargs):
            pass

        llama_log_set(llama_log_callback(noop_logger), ctypes.c_void_p())

    def _process_single_prompt(self, prompt: str) -> lx.inference.ScoredOutput:
        """Process a single prompt and return a ScoredOutput."""
        try:
            response = self._client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self._completion_kwargs,
            )

            response = cast(CreateChatCompletionResponse, response)
            result = response["choices"][0]["message"]["content"]  # type: ignore

            return lx.inference.ScoredOutput(score=1.0, output=result)
        except Exception as e:
            raise lx.exceptions.InferenceRuntimeError(
                f"llama-cpp error: {str(e)}", original=e
            ) from e

    def infer(self, batch_prompts, **kwargs):
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
