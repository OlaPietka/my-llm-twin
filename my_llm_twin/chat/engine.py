"""
Chat engine — loads the fine-tuned LoRA model and generates responses
in the same chat format used during training.
"""

import torch

from my_llm_twin import SYSTEM_PROMPT


class ChatEngine:
    """
    Loads a LoRA adapter on top of the base model and runs generation
    with the same chat template and special tokens from training.
    """

    def __init__(
        self,
        model_dir: str,
        separator: str = "<|msg|>",
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
    ):
        self.separator = separator
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

        self._load_model(model_dir)

    def _load_model(self, model_dir: str):
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # base model name is stored in the adapter config
        peft_config = PeftConfig.from_pretrained(model_dir)
        base_model_name = peft_config.base_model_name_or_path

        # tokenizer was saved with the adapter (includes <|msg|> special token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # 4-bit on CUDA (matches training), full precision otherwise
        load_kwargs: dict = {"device_map": "auto"}
        if torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **load_kwargs
        )
        base_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        self.model = PeftModel.from_pretrained(base_model, model_dir)
        self.model.eval()

    def generate(self, user_message: str) -> list[str]:
        """
        Generate a response to user_message, keeping conversation history.
        Returns a list of message parts (split on the burst separator).
        """
        self.history.append({"role": "user", "content": user_message})

        input_text = self.tokenizer.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # decode only new tokens — keep <|msg|> but strip other specials
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        for token in [self.tokenizer.eos_token, self.tokenizer.bos_token]:
            if token:
                response = response.replace(token, "")
        response = response.strip()

        # store full response (with separators) — same format as training
        self.history.append({"role": "assistant", "content": response})

        # split burst messages for display
        parts = [p.strip() for p in response.split(self.separator) if p.strip()]
        return parts if parts else [response]

    def reset(self):
        """Clear conversation history, keeping only the system prompt."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
