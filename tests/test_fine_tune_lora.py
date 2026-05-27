from __future__ import annotations

import importlib
import unittest

fine_tune_lora = importlib.import_module("fine_tune_lora")


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


class FakeDatasetsModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def load_dataset(self, name: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((name, kwargs))
        return {"name": name, "kwargs": kwargs}


class FineTuneLoraTest(unittest.TestCase):
    def test_parse_target_modules_trims_and_drops_empty_items(self) -> None:
        result = fine_tune_lora.parse_target_modules("q_proj, v_proj,, o_proj ")

        self.assertEqual(result, ("q_proj", "v_proj", "o_proj"))

    def test_extract_text_supports_common_sft_shapes(self) -> None:
        tokenizer = FakeTokenizer()

        self.assertEqual(
            fine_tune_lora.extract_text({"text": "hello"}, tokenizer, text_field="text"),
            "hello",
        )
        self.assertEqual(
            fine_tune_lora.extract_text(
                {"prompt": "Q", "completion": "A"},
                tokenizer,
                text_field="text",
            ),
            "Q\nA",
        )
        self.assertEqual(
            fine_tune_lora.extract_text(
                {"messages": [{"role": "user", "content": "hi"}]},
                tokenizer,
                text_field="text",
            ),
            "user: hi",
        )

    def test_load_training_dataset_uses_json_loader_for_jsonl_file(self) -> None:
        datasets = FakeDatasetsModule()
        config = fine_tune_lora.FineTuneConfig(dataset="train.jsonl", dataset_split="train")

        result = fine_tune_lora.load_training_dataset(datasets, config)

        self.assertEqual(result["name"], "json")
        self.assertEqual(datasets.calls[0][1]["data_files"], "train.jsonl")
        self.assertEqual(datasets.calls[0][1]["split"], "train")

    def test_build_training_arguments_passes_minimal_lora_defaults(self) -> None:
        captured: dict[str, object] = {}

        class FakeTrainingArguments:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        config = fine_tune_lora.FineTuneConfig(
            output_dir="out",
            num_train_epochs=2.0,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
        )

        fine_tune_lora.build_training_arguments(config, FakeTrainingArguments)

        self.assertEqual(captured["output_dir"], "out")
        self.assertEqual(captured["num_train_epochs"], 2.0)
        self.assertEqual(captured["per_device_train_batch_size"], 2)
        self.assertEqual(captured["gradient_accumulation_steps"], 4)
        self.assertEqual(captured["learning_rate"], 1e-4)
        self.assertEqual(captured["report_to"], "none")


if __name__ == "__main__":
    unittest.main()
