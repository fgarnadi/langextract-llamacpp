"""Provider implementation for LlamaCpp."""

import concurrent.futures
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

        self._extra_kwargs = kwargs
        self._client = Llama.from_pretrained(
            repo_id=self.repo_id,
            filename=self.filename,
            verbose=verbose,
            **self._extra_kwargs,
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
                stream=False,  # disable stream
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
        # Use parallel processing for batches larger than 1
        if len(batch_prompts) > 1 and self.max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batch_prompts))
            ) as executor:
                future_to_index = {
                    executor.submit(self._process_single_prompt, prompt): i
                    for i, prompt in enumerate(batch_prompts)
                }

                results: list[lx.inference.ScoredOutput | None] = [None] * len(
                    batch_prompts
                )
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                    except Exception as e:
                        raise lx.exceptions.InferenceRuntimeError(
                            f"Parallel inference error: {str(e)}", original=e
                        ) from e

                for result in results:
                    if result is None:
                        raise lx.exceptions.InferenceRuntimeError(
                            "Failed to process one or more prompts"
                        )
                    yield [result]
        else:
            # Sequential processing for single prompt or worker
            for prompt in batch_prompts:
                result = self._process_single_prompt(prompt)
                yield [result]
