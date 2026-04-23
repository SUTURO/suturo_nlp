import json
import timeit

from ollama import chat


class LLMHandler:
    DEFAULT_TEMPERATURE: float = 0.4
    DEFAULT_TOP_P: float = 0.95
    DEFAULT_TOP_K: int = 64

    def __init__(
        self,
        model_id: str,
        user_input: str,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        testing: bool = False,
        local: bool = True,
    ):
        self.model_id = model_id
        self.user_input = user_input
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.testing = testing
        self.local = local

    def start_inference(self):
        if self.testing:
            return self._start_inference_in_test_mode()
        if self.local:
            return self._start_inference_local(
                user_input=self.user_input,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            return self._start_inference_in_test_mode()

    def _start_inference_local(
        self,
        user_input: str,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        # 1. Use ollama python
        # 2. insert whisper text as user prompt
        # 3. check output as JSON string
        # 4. prepare JSON if necessary
        # 5. Return output to mcrs

        t1 = timeit.default_timer()
        response = chat(
            model=self.model_id,
            messages=[
                {
                    "role": "system",
                    "content": "Never answer in Markdown. Always provide clear ans sharp answers.",
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ],
            think=False,
            stream=False,
            format="json",
            options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
        )
        t2 = timeit.default_timer()
        response = response.message.content
        response_time = t2 - t1
        # is_json = json_loads(response)

        print(f">>> {response} <<<\n")
        print(f"Inference time: {response_time:.2f}")
        return response

    def _start_inference_in_test_mode(self):
        raise NotImplementedError
