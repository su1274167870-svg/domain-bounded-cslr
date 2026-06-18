from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class IntentTemplate:
    intent: str
    gloss: str
    text_zh: str
    safety_level: str


class IntentCatalog:
    def __init__(self, templates: dict[str, IntentTemplate], fallback: IntentTemplate) -> None:
        self._templates = templates
        self._fallback = fallback
        self._by_gloss: dict[str, list[IntentTemplate]] = {}
        for template in templates.values():
            gloss = template.gloss.upper()
            if gloss not in self._by_gloss:
                self._by_gloss[gloss] = []
            self._by_gloss[gloss].append(template)

    @classmethod
    def from_yaml(cls, path: Path) -> IntentCatalog:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        fallback_data = payload.get("fallback") or {}
        fallback = IntentTemplate(
            intent=str(fallback_data.get("intent", "unknown")),
            gloss=str(fallback_data.get("gloss", "UNKNOWN")),
            text_zh=str(fallback_data.get("text_zh", "未能可靠识别，请重新录制。")),
            safety_level="unknown",
        )
        templates = {}
        for intent, data in (payload.get("intents") or {}).items():
            templates[str(intent)] = IntentTemplate(
                intent=str(intent),
                gloss=str(data["gloss"]),
                text_zh=str(data["text_zh"]),
                safety_level=str(data.get("safety_level", "normal")),
            )
        if not templates:
            raise ValueError("intent catalog contains no intents")
        return cls(templates=templates, fallback=fallback)

    @property
    def intents(self) -> dict[str, IntentTemplate]:
        return dict(self._templates)

    def reconstruct(self, label: str, confidence: float, threshold: float) -> IntentTemplate:
        if confidence < threshold:
            return self._fallback
        
        # 先按标签精确匹配
        if label in self._templates:
            return self._templates[label]
        
        # 按 gloss 匹配（支持置信度分级）
        gloss = label.upper()
        if gloss in self._by_gloss:
            candidates = self._by_gloss[gloss]
            # 如果有多个匹配，按置信度选择
            if len(candidates) == 1:
                return candidates[0]
            # 根据置信度分级选择
            for candidate in candidates:
                # 从配置中读取置信度阈值（如果有）
                threshold_key = f"{candidate.intent}_threshold"
                # 默认：critical > 0.80, elevated > 0.60, normal > 0.40
                if candidate.safety_level == "critical" and confidence > 0.80:
                    return candidate
                elif candidate.safety_level == "elevated" and confidence > 0.60:
                    return candidate
                elif candidate.safety_level == "normal":
                    return candidate
            # 如果没有匹配的，返回第一个
            return candidates[0]
        
        return self._fallback
