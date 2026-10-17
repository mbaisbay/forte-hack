"""Reviewed rule registry. Regulation text always comes from here, never from the model."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    title: str | None
    section: str | None
    url: str | None
    quote: str | None
    short_title: str | None = None


@dataclass(frozen=True)
class Rule:
    rule_id: str
    type_ru: str
    visual_cue: str
    not_a_violation: str
    verified: bool
    source: Source

    def reference_ru(self) -> str:
        s = self.source
        if not self.verified or not s.title:
            return "Нормативный источник для этой категории ещё не подтверждён командой."
        parts = [s.short_title or s.title]
        if s.section:
            parts.append(s.section)
        ref = ", ".join(parts)
        return f"Основание: {ref}." + (f" {s.url}" if s.url else "")


class RuleRegistry:
    def __init__(self, rules: list[Rule], version: str):
        self.rules = rules
        self.version = version
        self._by_id = {r.rule_id: r for r in rules}

    @classmethod
    def load(cls, path: Path) -> "RuleRegistry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rules = []
        for r in raw["rules"]:
            if not r.get("enabled"):
                continue
            rules.append(
                Rule(
                    rule_id=r["rule_id"],
                    type_ru=r["type_ru"],
                    visual_cue=r["visual_cue"],
                    not_a_violation=r.get("not_a_violation", ""),
                    verified=bool(r.get("verified")),
                    source=Source(**r["source"]),
                )
            )
        ids = [r.rule_id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule_id in rules.json")
        if not rules:
            raise ValueError("no enabled rules in rules.json")
        return cls(rules, raw.get("version", "unknown"))

    def get(self, rule_id: str) -> Rule | None:
        return self._by_id.get(rule_id)

    def ids(self) -> list[str]:
        return list(self._by_id)

    def catalogue_text(self) -> str:
        lines = []
        for r in self.rules:
            lines.append(f"- {r.rule_id}: {r.visual_cue} NOT a violation: {r.not_a_violation}")
        return "\n".join(lines)
