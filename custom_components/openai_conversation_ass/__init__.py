from __future__ import annotations
import asyncio
import base64

from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform, CONF_API_KEY
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError

import voluptuous as vol
import openai

from .const import (
    DOMAIN,
    CONF_ASSISTANT_ID,
    CONF_PROMPT,
    CONF_FILENAMES
)

PLATFORMS = [Platform.CONVERSATION]
SERVICE_GENERATE_IMAGE = "generate_image"
SERVICE_GENERATE_CONTENT = "generate_content"

def encode_file(file_path: str) -> tuple[str, str]:
    """Return base64 version of file contents."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(file_path, "rb") as file:
        return mime_type, base64.b64encode(file.read()).decode()

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    client = openai.AsyncOpenAI(api_key=entry.data[CONF_API_KEY])
    entry.runtime_data = client

    async def send_prompt(call: ServiceCall) -> ServiceResponse:
        assistant_id = entry.data.get(CONF_ASSISTANT_ID)
        if not assistant_id:
            raise HomeAssistantError("Missing assistant_id")

        try:
            thread = await client.beta.threads.create()
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=call.data[CONF_PROMPT],
            )
            run = await client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id,
            )

            while True:
                run = await client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if run.status == "completed":
                    break
                elif run.status in ["failed", "cancelled"]:
                    raise HomeAssistantError(f"Run failed: {run.status}")
                await asyncio.sleep(1)

            messages = await client.beta.threads.messages.list(thread_id=thread.id)
            last = next((m for m in reversed(messages.data) if m.role == "assistant"), None)
            if not last or not last.content:
                raise HomeAssistantError("No assistant response")

            return {"text": "\n".join([part.text.value for part in last.content if part.type == "text"])}
        except Exception as e:
            raise HomeAssistantError(f"Error: {e}") from e

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        send_prompt,
        schema=vol.Schema({
            vol.Required("config_entry"): selector.ConfigEntrySelector({"integration": DOMAIN}),
            vol.Required(CONF_PROMPT): cv.string,
            vol.Optional(CONF_FILENAMES, default=[]): vol.All(cv.ensure_list, [cv.string]),
        }),
        supports_response=SupportsResponse.ONLY,
    )

    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return True
