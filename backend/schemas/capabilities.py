from pydantic import BaseModel, Field


class Capabilities(BaseModel):
    """What this deployment can actually do.

    The UI ships toggles for every mode the project has ever considered, and the client has no
    way to know which of them the server behind it implements. Left to guess, it guesses
    optimistically: `reasoning` and `web_search` were both clickable, both lit up when clicked,
    and both were dropped on the floor -- the request never carried them and no code read them.
    A control that looks live and does nothing is worse than no control, because the user
    attributes the missing behaviour to the model.

    Reporting capability from the server puts the decision where the knowledge is. A mode
    becomes available when the server implements it, without a frontend change.
    """

    rag: bool = Field(
        default=True,
        description="Retrieval over the indexed documents. Always available; whether it finds "
        "anything depends on what has been indexed.",
    )
    reasoning: bool = Field(
        default=False,
        description="The configured model emits a reasoning block that the server strips before "
        "returning the answer. This is a property of the model, not a per-request choice -- a "
        "model that does not think cannot be asked to, so the client surfaces it as a state "
        "rather than a switch.",
    )
    web_search: bool = Field(
        default=False,
        description="Not implemented. The repository contains a design note for it and nothing else.",
    )
    rerank: bool = Field(
        default=False,
        description="Two-stage retrieval with a cross-encoder. Reported so the UI can show why "
        "results are ordered the way they are.",
    )
    answer_without_context: bool = Field(
        default=True,
        description="Whether the server will answer from the model's own knowledge when "
        "retrieval finds nothing. Lets the client set expectations before the user asks.",
    )
