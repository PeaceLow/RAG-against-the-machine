import torch
from typing import List
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import Chunk


class LLMClient:
    """Client for generating answers using local LLM."""

    def __init__(
        self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "auto"
    ) -> None:
        """
        Initialize the LLM Client.
        Falls back to Qwen2.5-0.5B-Instruct if Qwen3-0.6B is not found.
        """
        self.model_name = model_name

        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(
            f"\033[93m\033[1mChargement\033[0m du modèle {self.model_name} "
            f"sur {self.device}..."
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16
                if self.device != "cpu"
                else torch.float32,
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
        except Exception as e:
            print(
                f"\033[93m\033[1mAttention\033[0m : Impossible de charger "
                f"{self.model_name}: {e}."
            )
            print(
                "\033[93m\033[1mFallback\033[0m : Retour sur "
                "Qwen/Qwen2.5-0.5B-Instruct."
            )
            self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16
                if self.device != "cpu"
                else torch.float32,
                trust_remote_code=True,
                attn_implementation="sdpa"
            )

        getattr(self.model, "eval")()

        if sys.platform.startswith("linux") and self.device == "cpu":
            try:
                print("\033[90mCompilation du modèle en cours "
                      "(le premier run sera plus long)...\033[0m")
                self.model = torch.compile(
                    self.model
                    )  # type: ignore[assignment]
            except Exception as e:
                print(f"\033[93mInfo : torch.compile "
                      f"non supporté sur cette version ({e})\033[0m")

    def generate_answer(
        self,
        question: str,
        chunks: List[Chunk],
        max_new_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        """
        Generate an answer to a question given a list of chunks as context.
        """
        context_text = "\n\n".join(
            [
                f"Source {i + 1}:\n{chunk.text}"
                for i, chunk in enumerate(chunks)
            ]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. You must answer the "
                    "user's question based ONLY on the provided context. "
                    "Cite your sources. If the answer is not in the context, "
                    "say so. Be as clear and concise as possible in "
                    "your answer, and get straight to the point."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback if the tokenizer doesn't support chat templates
            text = (
                f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
            )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.device
        )

        # Generate response
        if stream:
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs = dict(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            thread = Thread(
                target=self.model.generate,  # type: ignore[misc]
                kwargs=generation_kwargs,
            )
            with torch.no_grad():
                thread.start()

            response_text = ""
            for new_text in streamer:
                if "<think>" in new_text:
                    new_text = new_text.replace("<think>", "\033[90m<think>")
                if "</think>" in new_text:
                    new_text = new_text.replace("</think>", "</think>\033[0m")

                sys.stdout.write(new_text)
                sys.stdout.flush()
                response_text += new_text

            thread.join()
            sys.stdout.write("\033[0m")
            sys.stdout.flush()

            in_tokens = len(model_inputs.input_ids[0])
            out_tokens = len(self.tokenizer.encode(response_text))
            sys.stdout.write(
                f"\n\n\033[96m\033[1m[ Résultats \033[0m: Tokens consommés : "
                f"{in_tokens} (prompt) "
                f"| ~{out_tokens} (réponse) ]\033[0m\n"
            )
            sys.stdout.flush()

            return response_text.strip()
        else:
            with torch.no_grad():
                generated_ids = self.model.generate(  # type: ignore[misc]
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            new_generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(
                    model_inputs.input_ids, generated_ids
                )
            ]

            response = self.tokenizer.batch_decode(
                new_generated_ids, skip_special_tokens=True
            )[0]

            in_tokens = len(model_inputs.input_ids[0])
            out_tokens = len(new_generated_ids[0])
            sys.stdout.write(
                f"\n\n\033[96m\033[1m[ Résultats \033[0m: Tokens consommés : "
                f"{in_tokens} (prompt) "
                f"| {out_tokens} (réponse) ]\033[0m\n"
            )
            sys.stdout.flush()

            return response.strip()
