from __future__ import annotations

import asyncio
import base64
from mimetypes import guess_file_type
from pathlib import Path

import openai
import voluptuous as vol
from openai.types.images_response import ImagesResponse

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    LOGGER,
    CONF_ASSISTANT_ID,
    CONF_PROMPT,
    CONF_FILENAMES,
)

SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_GENERATE_CONTENT = "generate_content"

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    mime_type, _ = guess_file_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as file:
        return mime_type, base64.b64encode(file.read()).decode("utf-8")


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up the integration and register services."""

    async def render_image(call: ServiceCall) -> ServiceResponse:
        """Generate an image from prompt using DALL-E."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data

        try:
            response: ImagesResponse = await client.images.generate(
                model="dall-e-3",
                prompt=call.data[CONF_PROMPT],
                size=call.data["size"],
                quality=call.data["quality"],
                style=call.data["style"],
                response_format="url",
                n=1,
            )
        except openai.OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        if not response.data or not response.data[0].url:
            raise HomeAssistantError("No image returned")

        return response.data[0].model_dump(exclude={"b64_json"})

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        """Send a prompt to OpenAI Assistant and return its reply."""
        entry_id = call.data["config_entry"]
        entry = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        client: openai.AsyncClient = entry.runtime_data
        assistant_id = entry.data.get(CONF_ASSISTANT_ID)

        if not assistant_id:
            raise HomeAssistantError("Missing assistant_id in configuration")

        try:
            thread = await client.beta.threads.create()
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=call.data[CONF_PROMPT]
            )

            run = await client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
            )

            while True:
                run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if run.status == "completed":
                    break
                elif run.status in ["failed", "cancelled", "expired"]:
                    raise HomeAssistantError(f"Run failed with status: {run.status}")
                await asyncio.sleep(1)

            messages = await client.beta.threads.messages.list(thread_id=thread.id)
            last = next((m for m in reversed(messages.data) if m.role == "assistant"), None)

            if not last or not last.content:
                raise HomeAssistantError("No assistant response")

            return {
                "text": "\n".join(part.text.value for part in last.content if part.type == "text")
            }

        except Exception as err:
            raise HomeAssistantError(f"Error using Assistant API: {err}") from err

    # Register service: generate content
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema({
            vol.Required("config_entry"): selector.ConfigEntrySelector({
                "integration": DOMAIN
            }),
            vol.Required(CONF_PROMPT): cv.string,
            vol.Optional(CONF_FILENAMES, default=[]): vol.All(cv.ensure_list, [cv.string])
        }),
        supports_response=SupportsResponse.ONLY
    )

    # Register service: generate image
    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_IMAGE,
        render_image,
        schema=vol.Schema({
            vol.Required("config_entry"): selector.ConfigEntrySelector({
                "integration": DOMAIN
            }),
            vol.Required(CONF_PROMPT): cv.string,
            vol.Optional("size", default="1024x1024"): vol.In(("1024x1024", "1024x1792", "1792x1024")),
            vol.Optional("quality", default="standard"): vol.In(("standard", "hd")),
            vol.Optional("style", default="vivid"): vol.In(("vivid", "natural"))
        }),
        supports_response=SupportsResponse.ONLY
    )

    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up integration from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass)
    )

    try:
        _ = await hass.async_add_executor_job(client.platform_headers)
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    # 🟢 Tady dojde k vytvoření ConversationEntity přes conversation.py
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)