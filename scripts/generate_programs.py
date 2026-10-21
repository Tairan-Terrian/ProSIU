"""Regenerate the language-program proposals in a separate output directory."""

import argparse
import json
from pathlib import Path

import torch
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
from transformers import AutoModelForCausalLM, AutoTokenizer

from prosiu.common import DEVICE, write_json


@torch.inference_mode()
def generate(model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )
    operands = ["additive", "interaction", "novelty", "balance", "separation"]
    schema = {
        "type": "array",
        "minItems": 8,
        "maxItems": 8,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "weights", "rank"],
            "properties": {
                "name": {"type": "string"},
                "weights": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": operands,
                    "properties": {
                        key: {"enum": [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]}
                        for key in operands
                    },
                },
                "rank": {"enum": list(range(8))},
            },
        },
    }
    programs, responses = [], []
    for focus in [
        "Use additive improvement and favorable conditional interaction.",
        "Explore sparse context, balanced effects, and alternatives to the top-ranked candidate.",
    ]:
        prompt = (
            """Write eight distinct numerical candidate-ranking programs for a protein evidence agent.
The agent compares a pair of substitutions against an observed sequence background.
Available operands: additive = sum of measured constituent effects, interaction = local contextual estimate of epistasis,
novelty = 1/(1+context density), balance = negative absolute constituent-effect difference,
separation = normalized sequence distance. Fitness quantities use constituent standard-deviation units.
Each program computes a weighted sum of the five operands, sorts candidates descending, and chooses a rank 0..7.
Prefer larger additive and interaction values for exploitation. Use distinct coefficient patterns and ranks.
Return only a JSON array of objects with name, weights, rank. Include every operand in weights.
"""
            + focus
            + "\nExisting proposals: "
            + json.dumps(programs)
        )
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(DEVICE)
        generated = model.generate(
            **tokens,
            max_new_tokens=2500,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=build_transformers_prefix_allowed_tokens_fn(
                tokenizer, JsonSchemaParser(schema)
            )
        )
        response = tokenizer.decode(
            generated[0, tokens["input_ids"].shape[1] :], skip_special_tokens=True
        )
        responses.append({"prompt": prompt, "response": response})
        programs.extend(
            [
                {
                    "name": item["name"],
                    "weights": [item["weights"][key] for key in operands],
                    "rank": item["rank"],
                }
                for item in json.loads(response)
            ]
        )
        write_json(
            output_dir / "program_generation.json", {"model": model_name, "responses": responses}
        )
    write_json(output_dir / "programs.json", programs)
    print(
        json.dumps({"stage": "programs", "count": len(programs), "programs": programs}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated_programs"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    generate(args.model, args.output_dir)
