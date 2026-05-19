"""
Prompt Templates — Jinja2-based prompt rendering.
"""

from typing import Optional
from jinja2 import Template


def render_prompt(template_str: str, **kwargs) -> str:
    """Render a prompt template with Jinja2."""
    template = Template(template_str)
    return template.render(**kwargs)


def build_system_context(
    prompts_config: dict,
    service_categories: list[str],
    cities: list[str],
    areas: Optional[list[str]] = None,
) -> str:
    """Build the orchestrator system prompt with dynamic context."""
    template = prompts_config.get("orchestrator", "")
    return template.format(
        service_categories=", ".join(service_categories),
        cities=", ".join(cities),
        areas=", ".join(areas or []),
    )
