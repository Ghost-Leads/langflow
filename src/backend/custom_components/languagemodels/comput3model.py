from typing import Any
from urllib.parse import urljoin

import httpx
from langchain_openai import ChatOpenAI
from typing_extensions import override

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import DictInput, DropdownInput, FloatInput, IntInput, SecretStrInput, StrInput


class Comput3ModelComponent(LCModelComponent):
    display_name = "Comput3"
    description = "Generate text using comput3 LLMs."
    icon = "LMStudio"
    name = "Comput3Model"

    @override
    async def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None):
        if field_name == "model_name":
            base_url_dict = build_config.get("base_url", {})
            base_url_load_from_db = base_url_dict.get("load_from_db", False)
            base_url_value = base_url_dict.get("value")
            if base_url_load_from_db:
                base_url_value = await self.get_variables(base_url_value, field_name)
            elif not base_url_value:
                base_url_value = "http://localhost:1234/v1"
            build_config["model_name"]["options"] = await self.get_model(base_url_value, self.api_key)

        return build_config

    @staticmethod
    async def get_model(base_url_value: str, api_key) -> list[str]:
        try:
            url = urljoin(base_url_value, "/0/api/tags")
            headers = {
                "Content-Type": "application/json",
                "X-C3-API-KEY": api_key,
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                return [model["model"] for model in data.get("models", [])]
        except Exception as e:
            msg = f"Could not retrieve models. Please, make sure the Comput3 server is running.{e}"
            raise ValueError(msg) from e

    inputs = [
        *LCModelComponent._base_inputs,
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="The maximum number of tokens to generate. Set to 0 for unlimited tokens.",
            range_spec=RangeSpec(min=0, max=128000),
        ),
        DictInput(name="model_kwargs", display_name="Model Kwargs", advanced=True),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            refresh_button=True,
        ),
        StrInput(
            name="base_url",
            display_name="Base URL",
            advanced=False,
            info="Endpoint of the Comput3 API. Defaults to 'http://localhost:1234/v1' if not specified.",
            value="http://localhost:1234/v1",
        ),
        SecretStrInput(
            name="api_key",
            display_name="Comput3 API Key",
            info="The Comput3 API Key to use for Comput3.",
            advanced=True,
            value="C3_API_KEY",
        ),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            advanced=True,
        ),
        IntInput(
            name="seed",
            display_name="Seed",
            info="The seed controls the reproducibility of the job.",
            advanced=True,
            value=1,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        try:
            lmstudio_api_key = self.api_key
            temperature = self.temperature
            model_name: str = self.model_name
            max_tokens = self.max_tokens
            model_kwargs = self.model_kwargs or {}
            base_url = self.base_url or "http://localhost:1234/v1"
            seed = self.seed
            
            base_url = urljoin(base_url, "/tags/all/v1")
    
            return ChatOpenAI(
                max_tokens=max_tokens or None,
                model_kwargs=model_kwargs,
                model=model_name,
                base_url=base_url,
                api_key=lmstudio_api_key,
                temperature=temperature if temperature is not None else 0.1,
                seed=seed
            )
        except Exception as e:
            msg = f"Could not build model.{e}"
            raise ValueError(msg) from e

    def _get_exception_message(self, e: Exception):
        """Get a message from an Comput3 exception.

        Args:
            e (Exception): The exception to get the message from.

        Returns:
            str: The message from the exception.
        """
        try:
            from openai import BadRequestError
        except ImportError:
            return None
        if isinstance(e, BadRequestError):
            message = e.body.get("message")
            if message:
                return message
        return None