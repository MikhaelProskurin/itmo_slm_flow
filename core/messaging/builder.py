from pydantic import BaseModel

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

TSequence = tuple[str, str, BaseModel]

class LangchainMessageBuilder:
    
    def __init__(
            self, 
            templates: dict[str, str], 
            parsers: dict[str, PydanticOutputParser]
        ) -> None:
        self.templates = templates
        self.parsers = parsers

    @classmethod
    def from_sequence(cls, *args: TSequence) -> "LangchainMessageBuilder":

        templates = {}
        parsers = {}
        for sequence in args:
            key, template, output_model = sequence
            templates[key] = template
            parsers[key] = PydanticOutputParser(pydantic_object=output_model)

        return cls(templates, parsers)
    

    def _get_message_templating_objects(self, key: str) -> tuple[str, str]:
        return self.templates[key], self.parsers[key].get_format_instructions()
    

    def get_parser(self, key: str) -> PydanticOutputParser:
        return self.parsers[key]
    
    
    def create_message(self, key: str, **kwargs) -> BaseMessage:
        template, fmt = self._get_message_templating_objects(key)
        return SystemMessage(template.format(fmt=fmt, **kwargs))
