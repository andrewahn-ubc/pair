"""
Local Hugging Face Llama 2 Chat: one load can serve both attacker and target.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import LITELLM_TEMPLATES, Model
from language_models import LanguageModel


def _pick_dtype():
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

class LLMOutput:
    def __init__(self, responses, prompt_tokens, completion_tokens):
        self.responses = responses
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class LocalSharedLlamaChat(LanguageModel):
    """
    Single AutoModelForCausalLM + tokenizer load.
    - AttackLM uses batched_generate().
    - TargetLM (JailbreakBench path) uses query().
    """

    def __init__(self, model_path: str, model_name: str):
        super().__init__(model_name)
        # import jailbreakbench.config as jbb

        # self.jbb_model = jbb.Model(model_name.lower())
        # self.target_system_prompt = jbb.SYSTEM_PROMPTS[self.jbb_model]
        self.target_system_prompt = None # removed JBB dependency since I'm using a local judge

        dtype = _pick_dtype()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()
        self.use_open_source_model = True
        if self.model_name in LITELLM_TEMPLATES:
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
            self.eos_tokens = list(LITELLM_TEMPLATES[self.model_name]["eos_tokens"])
        else:
            self.post_message = ""
            self.eos_tokens = []

    def batched_generate(
        self,
        convs_list: list[list[dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
        extra_eos_tokens: list[str] | None = None,
    ) -> list[str]:
        responses: list[str] = []
        extra_eos_tokens = extra_eos_tokens or []

        for messages in convs_list:
            if self.tokenizer.chat_template is None:
                raise ValueError(
                    "Tokenizer has no chat_template; use a Llama 2 Chat snapshot with tokenizer_config.json."
                )

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            do_sample = temperature is not None and temperature > 0
            gen_kwargs = dict(
                max_new_tokens=max_n_tokens,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            with torch.inference_mode():
                if isinstance(input_ids, torch.Tensor):
                    out = self.model.generate(input_ids, **gen_kwargs)
                    prompt_len = input_ids.shape[-1]
                else:
                    out = self.model.generate(**input_ids, **gen_kwargs)
                    prompt_len = input_ids["input_ids"].shape[-1]

            new_tokens = out[0, prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            for stop in extra_eos_tokens:
                j = text.find(stop)
                if j != -1:
                    text = text[:j]
                    break

            responses.append(text)

        return responses

    def query(
        self,
        prompts: list[str],
        behavior: str = "unspecified_behavior",
        phase: str = "dev",
        max_new_tokens: int | None = None,
        defense: str | None = None,
    ):
        if defense is not None:
            raise NotImplementedError("Local target does not implement defenses.")

        max_new = max_new_tokens if max_new_tokens is not None else 150
        responses: list[str] = []
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []

        for prompt in prompts:
            if self.target_system_prompt is None:
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [
                    {"role": "system", "content": self.target_system_prompt},
                    {"role": "user", "content": prompt},
                ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                if isinstance(input_ids, torch.Tensor):
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                    prompt_len = input_ids.shape[-1]
                else:
                    out = self.model.generate(
                        **input_ids,
                        max_new_tokens=max_new,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )
                    prompt_len = input_ids["input_ids"].shape[-1]

            new_tokens = out[0, prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            responses.append(text)
            prompt_tokens.append(int(prompt_len))
            completion_tokens.append(int(new_tokens.shape[0]))

        return LLMOutput(
            responses=responses,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
