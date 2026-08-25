"""
Prompt construction shared by every LLM provider.

The templates and the way they are filled in do not depend on which backend serves the model, so
they live here rather than being duplicated in each client.
"""

from llm_providers.prompt import (
    CTX_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    REFINED_CTX_PROMPT_TEMPLATE,
    REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
    generate_conversation_awareness_prompt,
    generate_ctx_prompt,
    generate_qa_prompt,
    generate_refined_ctx_prompt,
)


class PromptBuilder:
    """Mixin providing the prompt builders every LLM client exposes."""

    @staticmethod
    def generate_qa_prompt(question: str) -> str:
        """
        Generates a question-answering (QA) prompt using predefined templates.

        Args:
            question: The question for which the prompt is generated

        Returns:
            str: The generated QA prompt
        """
        return generate_qa_prompt(
            template=QA_PROMPT_TEMPLATE,
            question=question,
        )

    @staticmethod
    def generate_ctx_prompt(question: str, context: str) -> str:
        """
        Generates a context-based prompt using predefined templates.

        Args:
            question: The question for which the prompt is generated
            context: The context information for the prompt

        Returns:
            str: The generated context-based prompt
        """
        return generate_ctx_prompt(
            template=CTX_PROMPT_TEMPLATE,
            question=question,
            context=context,
        )

    @staticmethod
    def generate_refined_ctx_prompt(question: str, context: str, existing_answer: str) -> str:
        """
        Generates a refined prompt for question-answering with existing answer.

        Args:
            question: The question for which the prompt is generated
            context: The context information for the prompt
            existing_answer: The existing answer to be refined

        Returns:
            str: The generated refined prompt
        """
        return generate_refined_ctx_prompt(
            template=REFINED_CTX_PROMPT_TEMPLATE,
            question=question,
            context=context,
            existing_answer=existing_answer,
        )

    @staticmethod
    def generate_refined_question_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        """
        Generates a refined question prompt with conversation awareness.

        Args:
            question: The question to be refined
            chat_history: The conversation history

        Returns:
            str: The generated conversation-aware prompt
        """
        return generate_conversation_awareness_prompt(
            template=REFINED_QUESTION_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )

    @staticmethod
    def generate_refined_answer_conversation_awareness_prompt(question: str, chat_history: str) -> str:
        """
        Generates a refined answer prompt with conversation awareness.

        Args:
            question: The question for the prompt
            chat_history: The conversation history

        Returns:
            str: The generated conversation-aware prompt
        """
        return generate_conversation_awareness_prompt(
            template=REFINED_ANSWER_CONVERSATION_AWARENESS_PROMPT_TEMPLATE,
            question=question,
            chat_history=chat_history,
        )
