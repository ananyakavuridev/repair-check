from reasoning.prompt import SYSTEM_PROMPT
from dotenv import load_dotenv
import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in .env")


BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

client_kwargs = {"api_key": api_key}

if BASE_URL:
    client_kwargs["base_url"] = BASE_URL

client = OpenAI(**client_kwargs)


class CostRange(BaseModel):
    min: float | None
    max: float | None
    currency: str


class Diagnosis(BaseModel):
    fault: str
    severity: str
    what_to_check_first: List[str]
    fair_cost_range: CostRange
    quote_verdict: str
    quote_reason: str
    confidence: str


def generate_diagnosis(prompt):

    schema = Diagnosis.model_json_schema()

    # OpenAI strict JSON schemas require additionalProperties=False
    def make_strict(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False

                if "properties" in obj:
                    for value in obj["properties"].values():
                        make_strict(value)

            for value in obj.values():
                make_strict(value)

        elif isinstance(obj, list):
            for item in obj:
                make_strict(item)

    make_strict(schema)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "diagnosis",
                "strict": True,
                "schema": schema
            }
        }
    )

    content = response.choices[0].message.content

    return Diagnosis.model_validate_json(content)
