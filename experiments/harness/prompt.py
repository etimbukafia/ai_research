"""
System prompt assembly and dynamic templating.

Provides a `PromptAssembler` that uses Jinja2 to render a static scaffold 
with dynamic context variables and auto-injected tool schemas.
"""

import json
import jinja2
from typing import Any, Dict, Optional

from experiments.harness.tools import ToolRegistry


class PromptAssembler:
    """
    Assembles a dynamic system prompt from a static scaffold and dynamic context.
    Uses Jinja2 for advanced templating (loops, conditionals, etc.).
    """

    def __init__(
        self,
        scaffold: str,
        tool_registry: Optional[ToolRegistry] = None,
        tool_variable_name: str = "tools",
    ) -> None:
        """
        Args:
            scaffold: The Jinja2 template string for the system prompt.
            tool_registry: Optional ToolRegistry. If provided, tool schemas will be available
                           in the template under `tool_variable_name`.
            tool_variable_name: The variable name used in the Jinja2 template to access the formatted tools.
        """
        self.scaffold = scaffold
        self.tool_registry = tool_registry
        self.tool_variable_name = tool_variable_name
        
        # Configure Jinja2 environment
        # I use strict_undefined=False (default is Undefined) so it doesn't fail 
        # on missing variables, allowing for partial rendering if desired.
        self.env = jinja2.Environment(
            loader=jinja2.DictLoader({"scaffold": self.scaffold}),
            autoescape=False, # Prompts are generally plain text, not HTML
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _format_tools(self) -> str:
        """Format the registered tools as a raw JSON string."""
        if not self.tool_registry or not self.tool_registry.names():
            return ""
        
        descriptors = self.tool_registry.descriptors()
        return json.dumps(descriptors, indent=2)

    def build(self, **dynamic_context: Any) -> str:
        """
        Render the final system prompt.

        Args:
            **dynamic_context: Arbitrary keyword arguments passed into the Jinja2 template.

        Returns:
            The fully rendered string.
        """
        template = self.env.get_template("scaffold")
        
        # Inject tool schemas if a registry is provided
        if self.tool_registry is not None and self.tool_variable_name not in dynamic_context:
            dynamic_context[self.tool_variable_name] = self._format_tools()

        return template.render(**dynamic_context)
