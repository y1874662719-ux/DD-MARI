from __future__ import annotations

import json
import re
import time
from http.client import RemoteDisconnected
from typing import Any
from urllib import error, request

from app.core.config import settings


class OpenAICompatibleLLMProvider:
    def __init__(self) -> None:
        provider_name, api_key, base_url, model = self._resolve_provider_config()
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.model = model
        self.last_error = ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.enabled:
            self.last_error = "provider_not_configured"
            return None

        if self.provider_name in {"deepseek", "geval_proxy"}:
            return self._complete_json_without_response_format(system_prompt, user_prompt)

        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = self._post(payload)
        data = self._parse_response_json(body)
        if data is not None:
            return data
        return self._complete_json_without_response_format(system_prompt, user_prompt)

    def _complete_json_without_response_format(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = self._post(payload)
        if not body:
            return None
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        return self._extract_json_object(str(content))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=60) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (RemoteDisconnected, ConnectionResetError, TimeoutError) as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= 2:
                    return None
                time.sleep(0.6 * (attempt + 1))
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")[:500]
                self.last_error = f"HTTPError {exc.code}: {detail}"
                return None
            except error.URLError as exc:
                self.last_error = f"URLError: {exc}"
                return None
            except json.JSONDecodeError as exc:
                self.last_error = f"JSONDecodeError: {exc}"
                return None
        return None

    def _parse_response_json(self, body: dict[str, Any] | None) -> dict[str, Any] | None:
        if not body:
            return None
        try:
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return None

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _resolve_provider_config(self) -> tuple[str, str, str, str]:
        provider = settings.llm_provider.strip().lower()

        if provider == "geval_proxy":
            return self._geval_proxy_config()
        if provider == "deepseek":
            return self._deepseek_config()
        if provider == "legacy":
            return self._legacy_config()

        if self._is_valid(settings.llm_api_key, settings.llm_base_url, settings.llm_model):
            return self._legacy_config()
        if self._is_valid(settings.geval_proxy_api_key, settings.geval_proxy_base_url, settings.geval_proxy_model):
            return self._geval_proxy_config()
        if self._is_valid(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model):
            return self._deepseek_config()
        return "", "", "", ""

    @staticmethod
    def _is_valid(api_key: str, base_url: str, model: str) -> bool:
        return bool(api_key and base_url and model)

    @staticmethod
    def _legacy_config() -> tuple[str, str, str, str]:
        return "legacy", settings.llm_api_key, settings.llm_base_url, settings.llm_model

    @staticmethod
    def _geval_proxy_config() -> tuple[str, str, str, str]:
        return "geval_proxy", settings.geval_proxy_api_key, settings.geval_proxy_base_url, settings.geval_proxy_model

    @staticmethod
    def _deepseek_config() -> tuple[str, str, str, str]:
        return "deepseek", settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model
