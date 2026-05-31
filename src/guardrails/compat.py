"""
Lightweight Guardrails AI compatibility shim.

Implements the exact subset of the guardrails-ai API used by this project:
  - Validator base class
  - register_validator decorator
  - PassResult / FailResult
  - Guard (use + validate)
  - OnFailAction enum

This allows the project to run without the guardrails-ai package while
keeping the same code structure and validator logic.  When guardrails-ai
becomes available in the environment, swap this module for the real one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OnFailAction enum  (mirrors guardrails.OnFailAction)
# ---------------------------------------------------------------------------

class OnFailAction(str, Enum):
    EXCEPTION = "exception"
    NOOP      = "noop"
    FIX       = "fix"
    FILTER    = "filter"
    REASK     = "reask"


# ---------------------------------------------------------------------------
# ValidationResult base + concrete results
# ---------------------------------------------------------------------------

class ValidationResult:
    """Base class for all validation results."""
    pass


@dataclass
class PassResult(ValidationResult):
    """Signals that validation passed without issues."""
    value_override: Any = None


@dataclass
class FailResult(ValidationResult):
    """Signals that validation failed."""
    error_message: str = ""
    fix_value: Any = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validator base class  (mirrors guardrails.validators.Validator)
# ---------------------------------------------------------------------------

class Validator:
    """Base class for all custom validators.

    Sub-classes must implement ``validate(self, value, metadata) -> ValidationResult``.
    """

    def __init__(self, on_fail: str | OnFailAction = OnFailAction.NOOP):
        if isinstance(on_fail, OnFailAction):
            self.on_fail = on_fail
        else:
            try:
                self.on_fail = OnFailAction(on_fail.lower())
            except ValueError:
                self.on_fail = OnFailAction.NOOP

    def validate(self, value: Any, metadata: dict | None = None) -> ValidationResult:  # pragma: no cover
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement validate()"
        )


# ---------------------------------------------------------------------------
# register_validator decorator  (mirrors guardrails.validators.register_validator)
# ---------------------------------------------------------------------------

_VALIDATOR_REGISTRY: dict[str, type] = {}


def register_validator(name: str, data_type: str = "string"):
    """Decorator that registers a validator class under a given name.

    Usage:
        @register_validator(name="my-validator", data_type="string")
        class MyValidator(Validator):
            ...
    """
    def decorator(cls: type) -> type:
        _VALIDATOR_REGISTRY[name] = cls
        cls._validator_name = name
        cls._data_type = data_type
        return cls
    return decorator


# ---------------------------------------------------------------------------
# ValidationOutcome  (mirrors guardrails ValidationOutcome)
# ---------------------------------------------------------------------------

@dataclass
class ValidationOutcome:
    """Returned by Guard.validate()."""
    validation_passed: bool
    validated_output: Any = None
    error: str = ""
    fail_results: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Guard class  (mirrors guardrails.Guard)
# ---------------------------------------------------------------------------

class Guard:
    """Chains multiple validators and runs them in sequence.

    Usage:
        guard = Guard()
        guard.use(MyValidator(on_fail=OnFailAction.EXCEPTION))
        outcome = guard.validate("some text")
        if not outcome.validation_passed:
            raise ValueError(outcome.error)
    """

    def __init__(self):
        self._validators: list[Validator] = []

    def use(self, validator: Validator) -> "Guard":
        """Register a validator instance.  Returns self for chaining."""
        self._validators.append(validator)
        return self

    def validate(
        self,
        value: Any,
        metadata: dict | None = None,
    ) -> ValidationOutcome:
        """Run all validators against *value*.

        Behaviour on failure depends on each validator's on_fail setting:
        - OnFailAction.EXCEPTION  → raise ValueError
        - OnFailAction.NOOP       → continue, accumulate fail_results
        - others                  → treated as NOOP for now
        """
        if metadata is None:
            metadata = {}

        fail_results = []
        current_value = value

        for validator in self._validators:
            try:
                result = validator.validate(current_value, metadata)
            except Exception as exc:
                # An unexpected error inside a validator is treated as a failure
                logger.debug(
                    "Validator %s raised unexpectedly: %s",
                    validator.__class__.__name__,
                    exc,
                )
                result = FailResult(error_message=str(exc))

            if isinstance(result, FailResult):
                fail_results.append(result)
                action = getattr(validator, "on_fail", OnFailAction.NOOP)
                if action == OnFailAction.EXCEPTION:
                    raise ValueError(result.error_message)
                # For NOOP / FIX / FILTER — log and continue
                logger.debug(
                    "Validator %s failed (on_fail=%s): %s",
                    validator.__class__.__name__,
                    action,
                    result.error_message,
                )

        passed = len(fail_results) == 0
        return ValidationOutcome(
            validation_passed=passed,
            validated_output=current_value if passed else None,
            error=fail_results[0].error_message if fail_results else "",
            fail_results=fail_results,
        )
