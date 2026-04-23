"""Prompt message builder that combines templates, format instructions, and Pydantic output parsers."""

from pydantic import BaseModel

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

TSequence = tuple[str, str, type[BaseModel]]


class LangchainMessageBuilder:
    """Builds LangChain ``SystemMessage`` objects by merging prompt templates with parser format instructions.

    Maintains a registry of named prompt templates and their associated ``PydanticOutputParser``
    instances, keyed by task name. Use ``from_sequence`` for the standard construction pattern.

    Args:
        templates: Mapping from the task key to a raw prompt template string (must contain ``{fmt}``).
        parsers: Mapping from the task key to the corresponding ``PydanticOutputParser``.
    """

    def __init__(
            self,
            templates: dict[str, str],
            parsers: dict[str, PydanticOutputParser]
        ) -> None:
        self.templates = templates
        self.parsers = parsers

    @classmethod
    def from_sequence(cls, *args: TSequence) -> "LangchainMessageBuilder":
        """Construct from a variable number of ``(key, template, output_model)`` tuples.

        Args:
            *args: Each element is a 3-tuple of ``(task_key, prompt_template, pydantic_output_model)``.

        Returns:
            A fully initialized ``LangchainMessageBuilder``.
        """
        templates = {}
        parsers = {}
        for sequence in args:
            key, template, output_model = sequence
            templates[key] = template
            parsers[key] = PydanticOutputParser(pydantic_object=output_model)

        return cls(templates, parsers)

    def _get_message_templating_objects(self, key: str) -> tuple[str, str]:
        """Return the raw template and serialized format instructions for ``key``."""
        return self.templates[key], self.parsers[key].get_format_instructions()

    def get_parser(self, key: str) -> PydanticOutputParser:
        """Return the ``PydanticOutputParser`` registered under ``key``."""
        return self.parsers[key]

    def create_message(self, key: str, **kwargs) -> SystemMessage:
        """Render the template for ``key`` into a ``SystemMessage``, injecting ``**kwargs`` and format instructions."""
        template, fmt = self._get_message_templating_objects(key)
        return SystemMessage(template.format(fmt=fmt, **kwargs))
