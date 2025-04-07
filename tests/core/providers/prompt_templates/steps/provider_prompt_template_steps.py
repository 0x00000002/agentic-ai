"""
Step definitions for provider prompt template tests.
"""
from behave import given, when, then
from unittest.mock import MagicMock, patch, AsyncMock
import json
import asyncio
from typing import Dict, Any, List

from src.core.providers.base_provider import BaseProvider
from src.core.providers.openai_provider import OpenAIProvider
from src.core.providers.anthropic_provider import AnthropicProvider
from src.core.providers.gemini_provider import GeminiProvider
from src.core.providers.ollama_provider import OllamaProvider
from src.core.models import ProviderResponse
from src.core.prompt_templates import (
    BaseTemplate,
    TemplateRegistry,
    TemplateValidator,
    TemplateCache,
    TemplateMetrics
)
from src.exceptions import AIProviderError

# Import shared fixtures
from tests.core.providers.fixtures import (
    create_mock_unified_config,
    create_mock_logger,
    create_provider_instance,
    create_mock_response,
    create_mock_tool_response,
    create_mock_tool_result
)

@given('the AI framework is initialized')
def step_impl(context):
    """Initialize the AI framework."""
    context.unified_config = create_mock_unified_config()

@given('provider configurations are available')
def step_impl(context):
    """Verify provider configurations are available."""
    assert context.unified_config is not None
    assert "openai" in context.unified_config.providers
    assert "anthropic" in context.unified_config.providers
    assert "gemini" in context.unified_config.providers
    assert "ollama" in context.unified_config.providers

@given('an OpenAI provider instance')
def step_impl(context):
    """Create an OpenAI provider instance."""
    context.provider = create_provider_instance(
        provider_type="openai",
        model_id="gpt-4",
        unified_config=context.unified_config,
        logger=create_mock_logger()
    )
    context.provider_type = "openai"

@given('a provider-specific template')
def step_impl(context):
    """Create a provider-specific template."""
    context.template = BaseTemplate(
        name="test_template",
        content="This is a {provider} specific template with {variable}",
        provider_type=context.provider_type,
        variables=["provider", "variable"]
    )

@given('a template with variables')
def step_impl(context):
    """Create a template with variables."""
    context.template = BaseTemplate(
        name="variable_template",
        content="Hello {name}, welcome to {service}!",
        variables=["name", "service"]
    )

@given('a template version')
def step_impl(context):
    """Create a template version."""
    context.template_version = "1.0.0"
    context.template.version = context.template_version

@given('a template performance tracker')
def step_impl(context):
    """Create a template performance tracker."""
    context.template_metrics = TemplateMetrics()
    context.template.metrics = context.template_metrics

@given('a template validator')
def step_impl(context):
    """Create a template validator."""
    context.template_validator = TemplateValidator()
    context.template.validator = context.template_validator

@given('a template cache')
def step_impl(context):
    """Create a template cache."""
    context.template_cache = TemplateCache()
    context.template.cache = context.template_cache

@given('a template registry')
def step_impl(context):
    """Create a template registry."""
    context.template_registry = TemplateRegistry()
    context.template.registry = context.template_registry

@given('a localized template')
def step_impl(context):
    """Create a localized template."""
    context.template = BaseTemplate(
        name="localized_template",
        content={
            "en": "Hello {name}!",
            "es": "¡Hola {name}!",
            "fr": "Bonjour {name}!"
        },
        variables=["name"],
        default_locale="en"
    )

@when('I request a prompt')
def step_impl(context):
    """Request a prompt from the template."""
    try:
        context.prompt = context.template.render(
            provider=context.provider_type,
            variable="test"
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.prompt = None

@when('I substitute variables')
def step_impl(context):
    """Substitute variables in the template."""
    try:
        context.rendered = context.template.render(
            name="John",
            service="AI Service"
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.rendered = None

@when('I track template performance')
def step_impl(context):
    """Track template performance."""
    try:
        context.template_metrics.record_render(
            template_name=context.template.name,
            render_time=0.1,
            success=True
        )
        context.error = None
    except Exception as e:
        context.error = e

@when('I validate the template')
def step_impl(context):
    """Validate the template."""
    try:
        context.is_valid = context.template_validator.validate(context.template)
        context.error = None
    except Exception as e:
        context.error = e
        context.is_valid = False

@when('I cache the template')
def step_impl(context):
    """Cache the template."""
    try:
        context.template_cache.set(
            key=context.template.name,
            value=context.template,
            ttl=3600
        )
        context.error = None
    except Exception as e:
        context.error = e

@when('I register the template')
def step_impl(context):
    """Register the template."""
    try:
        context.template_registry.register(context.template)
        context.error = None
    except Exception as e:
        context.error = e

@when('I localize the template')
def step_impl(context):
    """Localize the template."""
    try:
        context.localized = context.template.render(
            name="John",
            locale="es"
        )
        context.error = None
    except Exception as e:
        context.error = e
        context.localized = None

@then('the provider-specific template should be selected')
def step_impl(context):
    """Verify the provider-specific template is selected."""
    assert context.prompt is not None
    assert context.provider_type in context.prompt
    assert "test" in context.prompt
    assert context.error is None

@then('the variables should be substituted correctly')
def step_impl(context):
    """Verify the variables are substituted correctly."""
    assert context.rendered is not None
    assert "John" in context.rendered
    assert "AI Service" in context.rendered
    assert "{name}" not in context.rendered
    assert "{service}" not in context.rendered
    assert context.error is None

@then('the template version should be tracked')
def step_impl(context):
    """Verify the template version is tracked."""
    assert context.template.version == context.template_version
    assert context.error is None

@then('the performance metrics should be recorded')
def step_impl(context):
    """Verify the performance metrics are recorded."""
    assert context.error is None
    metrics = context.template_metrics.get_metrics(context.template.name)
    assert metrics is not None
    assert metrics["render_count"] > 0
    assert metrics["average_render_time"] > 0
    assert metrics["success_rate"] == 1.0

@then('the template should be valid')
def step_impl(context):
    """Verify the template is valid."""
    assert context.is_valid
    assert context.error is None

@then('the template should be cached')
def step_impl(context):
    """Verify the template is cached."""
    assert context.error is None
    cached = context.template_cache.get(context.template.name)
    assert cached is not None
    assert cached.name == context.template.name

@then('the template should be registered')
def step_impl(context):
    """Verify the template is registered."""
    assert context.error is None
    registered = context.template_registry.get(context.template.name)
    assert registered is not None
    assert registered.name == context.template.name

@then('the template should be localized')
def step_impl(context):
    """Verify the template is localized."""
    assert context.localized is not None
    assert "¡Hola John!" == context.localized
    assert context.error is None 