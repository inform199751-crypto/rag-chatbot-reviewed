from llm_providers.prompt import SYSTEM_TEMPLATE
from pydantic import BaseModel


class ModelSettings(BaseModel):
    # `url` and `file_name` describe a local GGUF file, so they are irrelevant to a hosted
    # provider and default to empty rather than forcing callers to invent values.
    url: str = ""
    name: str
    file_name: str = ""
    reasoning_start_tag: str | None
    reasoning_stop_tag: str | None
    system_template: str = SYSTEM_TEMPLATE
    reasoning: bool = False
