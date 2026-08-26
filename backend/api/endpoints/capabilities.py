from config import settings
from fastapi import APIRouter
from schemas.capabilities import Capabilities

from api.deps import LlamaCppClientDep, RerankerDep

router = APIRouter()


@router.get("/capabilities", response_model=Capabilities)
async def get_capabilities(llm_client: LlamaCppClientDep, reranker: RerankerDep):
    """
    Report what this deployment supports, so the client stops guessing.

    Read from live objects rather than from settings where the two can disagree: `reranker` is
    None unless startup actually built one, and `model_settings.reasoning` is what the client
    was constructed with. Reporting the setting instead would describe the intended deployment
    rather than the running one, which is precisely the gap this endpoint exists to close.
    """
    return Capabilities(
        rag=True,
        reasoning=bool(llm_client.model_settings.reasoning),
        web_search=False,
        rerank=reranker is not None,
        answer_without_context=settings.ANSWER_WITHOUT_CONTEXT,
    )
