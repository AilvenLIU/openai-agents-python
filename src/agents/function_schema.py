from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import re
import sys
import typing
from collections.abc import Callable
from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
    MethodWrapperType,
    WrapperDescriptorType,
)
from typing import Annotated, Any, Literal, cast, get_args, get_origin, get_type_hints

# griffelib exposes the `griffe` package at runtime but currently does not ship typing markers.
from griffe import Docstring, DocstringSectionKind  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from typing_extensions import Self

from ._callable_utils import (
    expand_type_alias,
    get_callable_call_descriptor,
    get_type_parameters,
    resolve_typevar_substitutions,
    substitute_typevars,
    unwrap_callable_descriptor,
)
from .exceptions import UserError
from .run_context import RunContextWrapper
from .strict_schema import ensure_strict_json_schema
from .tool_context import ToolContext

_CONTEXT_NOT_PROVIDED = object()
_IMPLICIT_POSITIONAL_ARG = object()
_PARTIAL_PLACEHOLDER = getattr(functools, "Placeholder", object())
_NATIVE_SELF = getattr(typing, "Self", Self)
_SUPPORTED_CALL_METHOD_TYPES = (
    FunctionType,
    MethodType,
    BuiltinFunctionType,
    MethodDescriptorType,
    WrapperDescriptorType,
    MethodWrapperType,
)


@dataclass
class FuncSchema:
    """
    Captures the schema for a python function, in preparation for sending it to an LLM as a tool.
    """

    name: str
    """The name of the function."""
    description: str | None
    """The description of the function."""
    params_pydantic_model: type[BaseModel]
    """A Pydantic model that represents the function's parameters."""
    params_json_schema: dict[str, Any]
    """The JSON schema for the function's parameters, derived from the Pydantic model."""
    signature: inspect.Signature
    """The signature of the function."""
    takes_context: bool = False
    """Whether the function takes a RunContextWrapper argument (must be the first argument)."""
    strict_json_schema: bool = True
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input."""
    return_annotation: Any = inspect.Signature.empty
    """The resolved return annotation, including `Annotated` metadata when present."""

    def to_call_args(
        self,
        data: BaseModel,
        *,
        context: Any = _CONTEXT_NOT_PROVIDED,
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Converts validated data from the Pydantic model into (args, kwargs), suitable for calling
        the original function. When provided, context is inserted according to the declared
        parameter kind.
        """
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        seen_var_positional = False
        context_parameter: tuple[str, inspect.Parameter] | None = None

        # Use enumerate() so we can identify the first parameter when it is context.
        for idx, (name, param) in enumerate(self.signature.parameters.items()):
            if self.takes_context and idx == 0:
                context_parameter = (name, param)
                continue
            else:
                value = getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                # e.g. *args: extend positional args.
                positional_args.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                # e.g. **kwargs handling
                keyword_args.update(value or {})
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                # Before *args, add to positional args. After *args, add to keyword args.
                if not seen_var_positional:
                    positional_args.append(value)
                else:
                    keyword_args[name] = value
            else:
                # For KEYWORD_ONLY parameters, always use keyword args.
                keyword_args[name] = value
        if context_parameter is not None and context is not _CONTEXT_NOT_PROVIDED:
            name, parameter = context_parameter
            # Model-provided **kwargs must never replace or duplicate the protected live context.
            keyword_args.pop(name, None)
            if parameter.kind == parameter.KEYWORD_ONLY:
                keyword_args[name] = context
            else:
                positional_args.insert(0, context)
        return positional_args, keyword_args


@dataclass
class FuncDocumentation:
    """Contains metadata about a Python function, extracted from its docstring."""

    name: str
    """The resolved name of the callable."""
    description: str | None
    """The description of the function, derived from the docstring."""
    param_descriptions: dict[str, str] | None
    """The parameter descriptions of the function, derived from the docstring."""


DocstringStyle = Literal["google", "numpy", "sphinx"]
CallableContractSource = Literal[
    "routine",
    "class",
    "partial",
    "explicit_signature",
    "wrapped",
    "call_descriptor",
]
CallableAnnotationSource = Literal[
    "signature",
    "published",
    "routine",
    "wrapped",
    "partial",
    "call_descriptor",
]
CallableInvocationMode = Literal["async", "threaded"]


@dataclass(frozen=True)
class _CallableDescriptorPlan:
    """Normalized binding and dispatch behavior for a raw ``__call__`` descriptor."""

    call_method: Any
    binds_receiver: bool
    partial_args: tuple[Any, ...]
    partial_keywords: dict[str, Any]
    has_partialmethod: bool
    dispatches_dynamically: bool
    dispatches_async_only: bool


@dataclass(frozen=True)
class ResolvedCallableContract:
    """Normalized metadata for one supported Python callable."""

    func: Callable[..., Any]
    name: str
    doc: str | None
    signature: inspect.Signature
    type_hints: dict[str, Any]
    invocation_mode: CallableInvocationMode
    source: CallableContractSource
    annotation_source: CallableAnnotationSource
    annotation_owner: Any | None
    call_owner: type[Any] | None
    call_descriptor: Any | None
    call_method: Any | None
    descriptor_plan: _CallableDescriptorPlan | None
    partial_target: ResolvedCallableContract | None = None

    @property
    def is_async(self) -> bool:
        """Return whether invocation can await the callable directly."""
        return self.invocation_mode == "async"


@dataclass(frozen=True)
class _ResolvedCallableAnnotations:
    type_hints: dict[str, Any]
    source: CallableAnnotationSource
    owner: Any | None


def _is_context_annotation(annotation: Any) -> bool:
    """Return whether an annotation denotes an injected function-tool context."""
    annotation = expand_type_alias(annotation)
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        if not args:
            break
        annotation = args[0]
    origin = get_origin(annotation) or annotation
    return origin is RunContextWrapper or origin is ToolContext


def _validate_no_positionally_bound_context(
    signature: inspect.Signature,
    type_hints: dict[str, Any],
    *,
    positional_args: tuple[Any, ...],
    implicit_positional_count: int,
    partial_kind: str,
) -> None:
    """Reject partial callables that capture an injected context positionally."""
    bound_args = [*([_IMPLICIT_POSITIONAL_ARG] * implicit_positional_count), *positional_args]
    bound_arg_index = 0

    for name, parameter in signature.parameters.items():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            remaining_args = bound_args[bound_arg_index:]
            has_bound_value = any(
                value is not _IMPLICIT_POSITIONAL_ARG and value is not _PARTIAL_PLACEHOLDER
                for value in remaining_args
            )
            if has_bound_value and _is_context_annotation(
                type_hints.get(name, parameter.annotation)
            ):
                raise UserError(
                    f"{partial_kind} cannot positionally bind a "
                    "RunContextWrapper/ToolContext parameter because the current run context "
                    "must be supplied at invocation."
                )
            return

        if bound_arg_index >= len(bound_args):
            return

        value = bound_args[bound_arg_index]
        bound_arg_index += 1
        if value is _IMPLICIT_POSITIONAL_ARG or value is _PARTIAL_PLACEHOLDER:
            continue
        if _is_context_annotation(type_hints.get(name, parameter.annotation)):
            raise UserError(
                f"{partial_kind} cannot positionally bind a "
                "RunContextWrapper/ToolContext parameter because the current run context "
                "must be supplied at invocation."
            )


def _get_callable_name(
    func: Callable[..., Any],
    partial_target: ResolvedCallableContract | None = None,
) -> str:
    """Return a stable name for functions and callable objects."""
    name = getattr(func, "__name__", None)
    if isinstance(name, str):
        return name
    if isinstance(func, functools.partial):
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            return _get_callable_name(unwrapped)
        if partial_target is not None:
            return partial_target.name
        return _get_callable_name(func.func)
    return type(func).__name__


def _get_callable_doc(
    func: Callable[..., Any],
    *,
    partial_target: ResolvedCallableContract | None = None,
    call_method: Any | None = None,
) -> str | None:
    """Return documentation from a callable or its invocation method."""
    if isinstance(func, functools.partial):
        published_doc = vars(func).get("__doc__", _CONTEXT_NOT_PROVIDED)
        if published_doc is not _CONTEXT_NOT_PROVIDED:
            return inspect.cleandoc(published_doc) if isinstance(published_doc, str) else None
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            return _get_callable_doc(unwrapped)
        if partial_target is not None:
            return partial_target.doc
        return _get_callable_doc(func.func)
    doc = inspect.getdoc(func)
    if doc is not None or inspect.isroutine(func) or inspect.isclass(func):
        return doc
    if call_method is None:
        _, call_descriptor = get_callable_call_descriptor(func)
        call_method = unwrap_callable_descriptor(call_descriptor)
    return inspect.getdoc(call_method)


def _resolve_callable_descriptor_plan(descriptor: Any) -> _CallableDescriptorPlan:
    """Normalize supported descriptor wrappers into one invocation binding plan."""
    partial_args: list[Any] = []
    partial_keywords: dict[str, Any] = {}
    has_partialmethod = False
    dispatches_dynamically = False
    dispatches_async_only = True
    binds_receiver = True

    while True:
        if isinstance(descriptor, functools.partialmethod):
            has_partialmethod = True
            partial_args.extend(descriptor.args)
            partial_keywords.update(descriptor.keywords or {})
            descriptor = descriptor.func
        elif isinstance(descriptor, functools.singledispatchmethod):
            dispatches_dynamically = True
            dispatches_async_only = dispatches_async_only and all(
                inspect.iscoroutinefunction(unwrap_callable_descriptor(implementation))
                for implementation in descriptor.dispatcher.registry.values()
            )
            descriptor = descriptor.func
        elif isinstance(descriptor, staticmethod):
            binds_receiver = False
            descriptor = descriptor.__func__
        elif isinstance(descriptor, classmethod):
            binds_receiver = True
            descriptor = descriptor.__func__
        else:
            break

    return _CallableDescriptorPlan(
        call_method=descriptor,
        binds_receiver=binds_receiver,
        partial_args=tuple(partial_args),
        partial_keywords=partial_keywords,
        has_partialmethod=has_partialmethod,
        dispatches_dynamically=dispatches_dynamically,
        dispatches_async_only=dispatches_async_only,
    )


def _get_callable_signature(
    func: Callable[..., Any],
    *,
    call_owner: type[Any] | None = None,
    call_descriptor: Any | None = None,
    call_method: Any | None = None,
    descriptor_plan: _CallableDescriptorPlan | None = None,
) -> inspect.Signature:
    """Return the effective invocation signature without double-binding descriptors."""
    if inspect.isroutine(func) or inspect.isclass(func) or isinstance(func, functools.partial):
        return inspect.signature(func)

    if isinstance(getattr(func, "__signature__", None), inspect.Signature) or hasattr(
        func, "__wrapped__"
    ):
        return inspect.signature(func)

    if call_owner is None or call_descriptor is None or call_method is None:
        call_owner, call_descriptor = get_callable_call_descriptor(func)
        call_method = unwrap_callable_descriptor(call_descriptor)
    if descriptor_plan is None:
        descriptor_plan = _resolve_callable_descriptor_plan(call_descriptor)
    if descriptor_plan.dispatches_dynamically:
        dynamic_bound_args: list[Any] = [object()] if descriptor_plan.binds_receiver else []
        dynamic_bound_args.extend(descriptor_plan.partial_args)
        if dynamic_bound_args or descriptor_plan.partial_keywords:
            return inspect.signature(
                functools.partial(
                    descriptor_plan.call_method,
                    *dynamic_bound_args,
                    **descriptor_plan.partial_keywords,
                )
            )
        return inspect.signature(descriptor_plan.call_method)
    if hasattr(call_method, "__wrapped__"):
        resolved_signature = inspect.signature(call_method)
        resolved_params = list(resolved_signature.parameters.values())

        descriptor = call_descriptor
        while isinstance(descriptor, functools.partialmethod):
            descriptor = descriptor.func
        binds_receiver = not isinstance(descriptor, staticmethod)
        first_annotation = (
            resolved_params[0].annotation if resolved_params else inspect.Signature.empty
        )
        resolved_includes_receiver = (
            binds_receiver
            and resolved_params
            and (
                resolved_params[0].name in ("self", "cls")
                or first_annotation in (Self, _NATIVE_SELF, call_owner)
                or first_annotation in ("Self", call_owner.__name__)
            )
        )

        bound_args: list[Any] = [object()] if resolved_includes_receiver else []
        bound_kwargs: dict[str, Any] = {}
        if isinstance(call_descriptor, functools.partialmethod):
            bound_args.extend(call_descriptor.args)
            bound_kwargs.update(call_descriptor.keywords or {})
        if bound_args or bound_kwargs:
            return inspect.signature(functools.partial(call_method, *bound_args, **bound_kwargs))
        return resolved_signature

    return inspect.signature(cast(Any, func).__call__)


def _get_callable_protocol_owner(
    func: Callable[..., Any],
    attribute: str,
) -> type[Any] | None:
    """Return the class that publishes a callable protocol attribute."""
    if inspect.isroutine(func) or inspect.isclass(func):
        return None
    try:
        if attribute in vars(func):
            return type(func)
    except TypeError:
        pass
    return next(
        (owner for owner in type(func).__mro__ if attribute in owner.__dict__),
        None,
    )


def _get_signature_type_hints(
    func: Callable[..., Any], signature: inspect.Signature
) -> dict[str, Any]:
    """Resolve annotations published by an explicit callable signature."""
    annotations = {
        name: parameter.annotation
        for name, parameter in signature.parameters.items()
        if parameter.annotation is not inspect.Signature.empty
    }
    if signature.return_annotation is not inspect.Signature.empty:
        annotations["return"] = signature.return_annotation
    if not annotations:
        return {}

    def annotation_source() -> None:
        pass

    annotation_source.__annotations__ = annotations
    signature_owner = _get_callable_protocol_owner(func, "__signature__")
    globalns, localns = _get_callable_annotation_namespaces(func, local_owner=signature_owner)
    resolved_hints = get_type_hints(
        annotation_source,
        globalns=globalns,
        localns=localns,
        include_extras=True,
    )
    owner = signature_owner or (func if inspect.isclass(func) else type(func))
    return _apply_callable_type_specialization(resolved_hints, func, owner)


def _get_callable_annotation_namespaces(
    func: Callable[..., Any],
    *,
    local_owner: type[Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return globals and locals that own a callable's published annotations."""
    namespace_source: Any = func
    namespace_target = inspect.unwrap(namespace_source)
    if namespace_target is namespace_source and isinstance(func, functools.partial):
        namespace_source = func.func
        namespace_target = inspect.unwrap(namespace_source)
    wrapper_globalns = getattr(namespace_source, "__globals__", None)
    target_globalns = getattr(namespace_target, "__globals__", None)
    globalns = dict(target_globalns) if target_globalns is not None else {}
    if wrapper_globalns is not None:
        globalns.update(wrapper_globalns)

    if local_owner is None:
        if inspect.isclass(namespace_source):
            local_owner = namespace_source
        elif not inspect.isroutine(namespace_source):
            local_owner = type(namespace_source)

    localns: dict[str, Any] = {}
    if local_owner is not None:
        module = sys.modules.get(local_owner.__module__)
        if module is not None:
            globalns.update(vars(module))
        localns.update(vars(local_owner))
        localns[local_owner.__name__] = local_owner
        for type_param in get_type_parameters(local_owner):
            localns.setdefault(type_param.__name__, type_param)
    return globalns, localns


def _apply_callable_type_specialization(
    type_hints: dict[str, Any],
    func: Callable[..., Any],
    owner: type[Any],
) -> dict[str, Any]:
    """Apply an instance's generic specialization to resolved callable hints."""
    specialization = getattr(func, "__orig_class__", None)
    substitutions = resolve_typevar_substitutions(specialization, owner)
    if not substitutions:
        substitutions = resolve_typevar_substitutions(type(func), owner)
    self_owner = func if inspect.isclass(func) else type(func)
    substitutions = {
        Self: self_owner,
        _NATIVE_SELF: self_owner,
        **substitutions,
    }
    return {
        name: substitute_typevars(annotation, substitutions)
        for name, annotation in type_hints.items()
    }


def _get_callable_type_hints(
    func: Callable[..., Any],
    signature: inspect.Signature,
    *,
    partial_target: ResolvedCallableContract | None = None,
    call_owner: type[Any] | None = None,
    call_descriptor: Any | None = None,
    call_method: Any | None = None,
    descriptor_plan: _CallableDescriptorPlan | None = None,
) -> _ResolvedCallableAnnotations:
    """Resolve callable hints using signature, published, wrapped, then structural metadata."""
    if isinstance(func, functools.partial):
        if partial_target is None:
            partial_target = resolve_callable_contract(func.func)
        _validate_no_positionally_bound_context(
            partial_target.signature,
            partial_target.type_hints,
            positional_args=func.args,
            implicit_positional_count=0,
            partial_kind="functools.partial",
        )

    if isinstance(getattr(func, "__signature__", None), inspect.Signature):
        return _ResolvedCallableAnnotations(
            type_hints=_get_signature_type_hints(func, signature),
            source="signature",
            owner=_get_callable_protocol_owner(func, "__signature__"),
        )

    try:
        instance_vars = vars(func)
    except TypeError:
        instance_vars = {}

    if "__annotations__" in instance_vars or "__annotate__" in instance_vars:
        annotation_namespace_source = func
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            for attribute in ("__annotations__", "__annotate__"):
                published = instance_vars.get(attribute, _CONTEXT_NOT_PROVIDED)
                wrapped = getattr(unwrapped, attribute, _CONTEXT_NOT_PROVIDED)
                if published is not _CONTEXT_NOT_PROVIDED and published is wrapped:
                    annotation_namespace_source = unwrapped
                    break
        globalns, localns = _get_callable_annotation_namespaces(annotation_namespace_source)
        try:
            published_hints = get_type_hints(
                func,
                globalns=globalns,
                localns=localns,
                include_extras=True,
            )
        except TypeError:
            published_hints = inspect.get_annotations(
                func,
                globals=globalns,
                locals=localns,
                eval_str=True,
            )
        return _ResolvedCallableAnnotations(
            type_hints=_apply_callable_type_specialization(published_hints, func, type(func)),
            source="published",
            owner=type(func),
        )

    if isinstance(func, functools.partial):
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            wrapped_contract = resolve_callable_contract(unwrapped)
            return _ResolvedCallableAnnotations(
                type_hints=wrapped_contract.type_hints,
                source="wrapped",
                owner=wrapped_contract.annotation_owner,
            )
        assert partial_target is not None
        return _ResolvedCallableAnnotations(
            type_hints={
                name: annotation
                for name, annotation in partial_target.type_hints.items()
                if name == "return" or name in signature.parameters
            },
            source="partial",
            owner=partial_target.annotation_owner,
        )

    if inspect.isroutine(func) or inspect.isclass(func):
        return _ResolvedCallableAnnotations(
            type_hints=get_type_hints(func, include_extras=True),
            source="routine",
            owner=func if inspect.isclass(func) else None,
        )

    unwrapped = inspect.unwrap(func)
    if unwrapped is not func:
        wrapped_contract = resolve_callable_contract(unwrapped)
        return _ResolvedCallableAnnotations(
            type_hints=wrapped_contract.type_hints,
            source="wrapped",
            owner=wrapped_contract.annotation_owner,
        )

    if call_owner is None or call_descriptor is None or call_method is None:
        call_owner, call_descriptor = get_callable_call_descriptor(func)
        call_method = unwrap_callable_descriptor(call_descriptor)
    if descriptor_plan is None:
        descriptor_plan = _resolve_callable_descriptor_plan(call_descriptor)
    globalns, localns = _get_callable_annotation_namespaces(func)
    call_globalns = getattr(call_method, "__globals__", None)
    if call_globalns is not None:
        globalns.update(call_globalns)
    else:
        call_module = sys.modules.get(call_owner.__module__)
        if call_module is not None:
            globalns.update(vars(call_module))
    localns.update(vars(call_owner))
    localns[call_owner.__name__] = call_owner
    for type_param in get_type_parameters(call_owner):
        # Postponed annotations on an inherited __call__ belong to the defining owner.
        localns[type_param.__name__] = type_param
    call_type_hints = get_type_hints(
        call_method,
        globalns=globalns,
        localns=localns,
        include_extras=True,
    )
    call_type_hints = _apply_callable_type_specialization(
        call_type_hints,
        func,
        call_owner,
    )
    if descriptor_plan.has_partialmethod:
        implicit_positional_count = 1 if descriptor_plan.binds_receiver else 0
        _validate_no_positionally_bound_context(
            inspect.signature(call_method),
            call_type_hints,
            positional_args=descriptor_plan.partial_args,
            implicit_positional_count=implicit_positional_count,
            partial_kind="functools.partialmethod",
        )
        call_type_hints = {
            name: annotation
            for name, annotation in call_type_hints.items()
            if name == "return" or name in signature.parameters
        }
    return _ResolvedCallableAnnotations(
        type_hints=call_type_hints,
        source="call_descriptor",
        owner=call_owner,
    )


def _resolve_structural_call_descriptor(
    func: Callable[..., Any],
) -> tuple[
    type[Any] | None,
    Any | None,
    Any | None,
    _CallableDescriptorPlan | None,
]:
    """Resolve and validate the raw invocation descriptor for a callable object."""
    if inspect.isroutine(func) or inspect.isclass(func) or isinstance(func, functools.partial):
        return None, None, None, None

    call_owner, call_descriptor = get_callable_call_descriptor(func)
    descriptor_plan = _resolve_callable_descriptor_plan(call_descriptor)
    call_method = descriptor_plan.call_method
    if not isinstance(call_method, _SUPPORTED_CALL_METHOD_TYPES):
        raise UserError(
            "Unsupported callable object: __call__ must be a function, method, builtin, "
            "staticmethod, classmethod, functools.partialmethod, or "
            "functools.singledispatchmethod. Publish an explicit wrapper function instead of "
            f"the {type(call_descriptor).__name__} descriptor."
        )
    return call_owner, call_descriptor, call_method, descriptor_plan


def _get_callable_contract_source(
    func: Callable[..., Any],
) -> CallableContractSource:
    """Classify the public contract source selected for a callable."""
    if isinstance(getattr(func, "__signature__", None), inspect.Signature):
        return "explicit_signature"
    if isinstance(func, functools.partial):
        return "partial"
    if inspect.isclass(func):
        return "class"
    if inspect.isroutine(func):
        return "routine"
    if hasattr(func, "__wrapped__"):
        return "wrapped"
    return "call_descriptor"


def resolve_callable_contract(func: Callable[..., Any]) -> ResolvedCallableContract:
    """Resolve one normalized contract used by schema generation and invocation."""
    partial_target = (
        resolve_callable_contract(func.func) if isinstance(func, functools.partial) else None
    )
    call_owner, call_descriptor, call_method, descriptor_plan = _resolve_structural_call_descriptor(
        func
    )
    signature = _get_callable_signature(
        func,
        call_owner=call_owner,
        call_descriptor=call_descriptor,
        call_method=call_method,
        descriptor_plan=descriptor_plan,
    )
    annotations = _get_callable_type_hints(
        func,
        signature,
        partial_target=partial_target,
        call_owner=call_owner,
        call_descriptor=call_descriptor,
        call_method=call_method,
        descriptor_plan=descriptor_plan,
    )
    if partial_target is not None:
        invocation_mode = partial_target.invocation_mode
        resolved_call_owner = partial_target.call_owner
        resolved_call_descriptor = partial_target.call_descriptor
        resolved_call_method = partial_target.call_method
        resolved_descriptor_plan = partial_target.descriptor_plan
    else:
        is_direct_async = inspect.iscoroutinefunction(func) or (
            call_method is not None and inspect.iscoroutinefunction(call_method)
        )
        invocation_mode = (
            "async"
            if is_direct_async
            and (
                descriptor_plan is None
                or not descriptor_plan.dispatches_dynamically
                or descriptor_plan.dispatches_async_only
            )
            else "threaded"
        )
        resolved_call_owner = call_owner
        resolved_call_descriptor = call_descriptor
        resolved_call_method = call_method
        resolved_descriptor_plan = descriptor_plan

    return ResolvedCallableContract(
        func=func,
        name=_get_callable_name(func, partial_target),
        doc=_get_callable_doc(
            func,
            partial_target=partial_target,
            call_method=call_method,
        ),
        signature=signature,
        type_hints=annotations.type_hints,
        invocation_mode=invocation_mode,
        source=_get_callable_contract_source(func),
        annotation_source=annotations.source,
        annotation_owner=annotations.owner,
        call_owner=resolved_call_owner,
        call_descriptor=resolved_call_descriptor,
        call_method=resolved_call_method,
        descriptor_plan=resolved_descriptor_plan,
        partial_target=partial_target,
    )


# As of Feb 2025, the automatic style detection in griffe is an Insiders feature. This
# code approximates it.
def _detect_docstring_style(doc: str) -> DocstringStyle:
    scores: dict[DocstringStyle, int] = {"sphinx": 0, "numpy": 0, "google": 0}

    # Sphinx style detection: look for :param, :type, :return:, and :rtype:
    sphinx_patterns = [r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"]
    for pattern in sphinx_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1

    # Numpy style detection: look for headers like 'Parameters', 'Returns', or 'Yields' followed by
    # a dashed underline
    numpy_patterns = [
        r"^Parameters\s*\n\s*-{3,}",
        r"^Returns\s*\n\s*-{3,}",
        r"^Yields\s*\n\s*-{3,}",
    ]
    for pattern in numpy_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1

    # Google style detection: look for section headers with a trailing colon
    google_patterns = [r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"]
    for pattern in google_patterns:
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1

    max_score = max(scores.values())
    if max_score == 0:
        return "google"

    # Priority order: sphinx > numpy > google in case of tie
    styles: list[DocstringStyle] = ["sphinx", "numpy", "google"]

    for style in styles:
        if scores[style] == max_score:
            return style

    return "google"


@contextlib.contextmanager
def _suppress_griffe_logging():
    # Suppresses warnings about missing annotations for params
    logger = logging.getLogger("griffe")
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)


# Aliases of the Google-style parameter section header ("Args:") — the only section kind
# that generate_func_documentation below consumes for parameter descriptions. A header only
# counts when the whole line is exactly ``Header:`` (griffe anchors these at column 0), so
# inline mentions such as "see Args: below" never match.
_GOOGLE_SECTION_HEADER_RE = re.compile(
    r"^(args|arguments|params|parameters):\s*$",
    re.IGNORECASE,
)


def _ensure_blank_line_before_google_sections(doc: str) -> str:
    """Insert a blank line before a Google-style parameter section header (``Args:`` or an
    alias) that directly follows a non-blank line, such as a summary line or the indented body
    of a preceding section.

    griffe's Google parser silently skips a section header when there is no blank line above
    it and the following line is indented (it logs "Missing blank line above section"). That
    drops every parameter description and leaks the raw ``Args:`` block into the description.
    griffe applies that gate no matter how the line above is indented, so a header that follows
    another section's indented body (for example ``Note:`` or ``Example:``) needs the same
    normalization as one that follows the summary. numpy/sphinx parsing already tolerates the
    missing blank line, so this normalizes the Google case to match. Only the parameter section
    is normalized because generate_func_documentation only consumes parameter sections (plus
    the first text block); other griffe sections are intentionally left alone. The string is
    returned unchanged when no insertion is needed, which keeps well-formed docstrings
    byte-identical.
    """
    lines = doc.splitlines()
    output: list[str] = []
    inserted = False
    for index, line in enumerate(lines):
        if (
            index > 0
            and _GOOGLE_SECTION_HEADER_RE.match(line)
            # Preceding line is non-blank, so griffe would skip the header. Its indentation does
            # not matter, because the header itself is anchored at column 0 by the regex above.
            and output
            and output[-1].strip()
            # Following line is an indented block, matching griffe's "indented line below" gate.
            and index + 1 < len(lines)
            and lines[index + 1].startswith((" ", "\t"))
        ):
            output.append("")
            inserted = True
        output.append(line)

    if not inserted:
        # Preserve the original object (splitlines/join would drop a trailing newline).
        return doc
    return "\n".join(output)


def generate_func_documentation(
    func: Callable[..., Any], style: DocstringStyle | None = None
) -> FuncDocumentation:
    """
    Extracts metadata from a function docstring, in preparation for sending it to an LLM as a tool.

    Args:
        func: The function to extract documentation from.
        style: The style of the docstring to use for parsing. If not provided, we will attempt to
            auto-detect the style.

    Returns:
        A FuncDocumentation object containing the function's name, description, and parameter
        descriptions.
    """
    return _parse_func_documentation(
        name=_get_callable_name(func),
        doc=_get_callable_doc(func),
        style=style,
    )


def _generate_func_documentation(
    contract: ResolvedCallableContract,
    style: DocstringStyle | None = None,
) -> FuncDocumentation:
    """Extract model-facing documentation from a resolved callable contract."""
    return _parse_func_documentation(name=contract.name, doc=contract.doc, style=style)


def _parse_func_documentation(
    *,
    name: str,
    doc: str | None,
    style: DocstringStyle | None,
) -> FuncDocumentation:
    """Parse model-facing documentation from normalized callable metadata."""
    if not doc:
        return FuncDocumentation(name=name, description=None, param_descriptions=None)

    # Resolve the style against the original docstring before any normalization.
    resolved_style = style or _detect_docstring_style(doc)
    if resolved_style == "google":
        doc = _ensure_blank_line_before_google_sections(doc)

    with _suppress_griffe_logging():
        docstring = Docstring(doc, lineno=1, parser=resolved_style)
        parsed = docstring.parse()

    description: str | None = next(
        (section.value for section in parsed if section.kind == DocstringSectionKind.text), None
    )

    param_descriptions: dict[str, str] = {
        param.name: param.description
        for section in parsed
        if section.kind == DocstringSectionKind.parameters
        for param in section.value
    }

    return FuncDocumentation(
        name=name,
        description=description,
        param_descriptions=param_descriptions or None,
    )


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Returns the underlying annotation and any metadata from typing.Annotated."""

    metadata: tuple[Any, ...] = ()
    ann = annotation

    while get_origin(ann) is Annotated:
        args = get_args(ann)
        if not args:
            break
        ann = args[0]
        metadata = (*metadata, *args[1:])

    return ann, metadata


def _extract_description_from_metadata(metadata: tuple[Any, ...]) -> str | None:
    """Extracts a human readable description from Annotated metadata if present."""

    for item in metadata:
        if isinstance(item, str):
            return item
    return None


def _extract_field_info_from_metadata(metadata: tuple[Any, ...]) -> FieldInfo | None:
    """Returns the first FieldInfo in Annotated metadata, or None."""

    for item in metadata:
        if isinstance(item, FieldInfo):
            return item
    return None


def function_schema(
    func: Callable[..., Any],
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """
    Given a Python function, extracts a `FuncSchema` from it, capturing the name, description,
    parameter descriptions, and other metadata.

    Args:
        func: The function to extract the schema from.
        docstring_style: The style of the docstring to use for parsing. If not provided, we will
            attempt to auto-detect the style.
        name_override: If provided, use this name instead of the function's `__name__`.
        description_override: If provided, use this description instead of the one derived from the
            docstring.
        use_docstring_info: If True, uses the docstring to generate the description and parameter
            descriptions.
        strict_json_schema: Whether the JSON schema is in strict mode. If True, we'll ensure that
            the schema adheres to the "strict" standard the OpenAI API expects. We **strongly**
            recommend setting this to True, as it increases the likelihood of the LLM producing
            correct JSON input.

    Returns:
        A `FuncSchema` object containing the function's name, description, parameter descriptions,
        and other metadata.
    """
    return _function_schema_from_contract(
        resolve_callable_contract(func),
        docstring_style=docstring_style,
        name_override=name_override,
        description_override=description_override,
        use_docstring_info=use_docstring_info,
        strict_json_schema=strict_json_schema,
    )


def _function_schema_from_contract(
    contract: ResolvedCallableContract,
    docstring_style: DocstringStyle | None = None,
    name_override: str | None = None,
    description_override: str | None = None,
    use_docstring_info: bool = True,
    strict_json_schema: bool = True,
) -> FuncSchema:
    """Build a function schema from a previously resolved callable contract."""

    # 1. Grab docstring info
    if use_docstring_info:
        doc_info = _generate_func_documentation(contract, docstring_style)
        param_descs = dict(doc_info.param_descriptions or {})
    else:
        doc_info = None
        param_descs = {}

    sig = contract.signature
    type_hints_with_extras = contract.type_hints
    type_hints: dict[str, Any] = {}
    annotated_param_descs: dict[str, str] = {}
    param_metadata: dict[str, tuple[Any, ...]] = {}

    for name, annotation in type_hints_with_extras.items():
        if name == "return":
            continue

        stripped_ann, metadata = _strip_annotated(annotation)
        type_hints[name] = stripped_ann
        param_metadata[name] = metadata

        description = _extract_description_from_metadata(metadata)
        if description is not None:
            annotated_param_descs[name] = description

    for name, description in annotated_param_descs.items():
        param_descs.setdefault(name, description)

    # Ensure name_override takes precedence even if docstring info is disabled.
    func_name = name_override or (doc_info.name if doc_info else contract.name)

    # 2. Inspect function signature and get type hints
    params = list(sig.parameters.items())
    takes_context = False
    filtered_params = []

    if params:
        first_name, first_param = params[0]
        # Prefer the evaluated type hint if available
        ann = type_hints.get(first_name, first_param.annotation)
        if ann != inspect._empty:
            if _is_context_annotation(ann):
                if first_param.kind == first_param.VAR_KEYWORD:
                    raise UserError(
                        "RunContextWrapper/ToolContext cannot be used as a **kwargs parameter "
                        f"in function {func_name}"
                    )
                takes_context = True  # Mark that the function takes context
            else:
                filtered_params.append((first_name, first_param))
        else:
            filtered_params.append((first_name, first_param))

    # For parameters other than the first, raise error if any use RunContextWrapper or ToolContext.
    for name, param in params[1:]:
        ann = type_hints.get(name, param.annotation)
        if ann != inspect._empty:
            if _is_context_annotation(ann):
                raise UserError(
                    f"RunContextWrapper/ToolContext param found at non-first position in function"
                    f" {func_name}"
                )
        filtered_params.append((name, param))

    # We will collect field definitions for create_model as a dict:
    #   field_name -> (type_annotation, default_value_or_Field(...))
    fields: dict[str, Any] = {}

    for name, param in filtered_params:
        ann = type_hints.get(name, param.annotation)
        default = param.default

        # If there's no type hint, assume `Any`
        if ann == inspect._empty:
            ann = Any

        # If a docstring param description exists, use it
        field_description = param_descs.get(name, None)

        # Handle different parameter kinds
        if param.kind == param.VAR_POSITIONAL:
            # e.g. *args: extend positional args
            if get_origin(ann) is tuple:
                # e.g. def foo(*args: tuple[int, ...]) -> treat as List[int]
                args_of_tuple = get_args(ann)
                if len(args_of_tuple) == 2 and args_of_tuple[1] is Ellipsis:
                    ann = list[args_of_tuple[0]]  # type: ignore
                else:
                    ann = list[Any]
            else:
                # If user wrote *args: int, treat as List[int]
                ann = list[ann]  # type: ignore

            # Default factory to empty list
            fields[name] = (
                ann,
                Field(default_factory=list, description=field_description),
            )

        elif param.kind == param.VAR_KEYWORD:
            # **kwargs handling
            if get_origin(ann) is dict:
                # e.g. def foo(**kwargs: dict[str, int])
                dict_args = get_args(ann)
                if len(dict_args) == 2:
                    ann = dict[dict_args[0], dict_args[1]]  # type: ignore
                else:
                    ann = dict[str, Any]
            else:
                # e.g. def foo(**kwargs: int) -> Dict[str, int]
                ann = dict[str, ann]  # type: ignore

            fields[name] = (
                ann,
                Field(default_factory=dict, description=field_description),
            )

        else:
            # Normal parameter
            metadata = param_metadata.get(name, ())
            field_info_from_annotated = _extract_field_info_from_metadata(metadata)

            if field_info_from_annotated is not None:
                merged = FieldInfo.merge_field_infos(
                    field_info_from_annotated,
                    description=field_description or field_info_from_annotated.description,
                )
                if default != inspect._empty and not isinstance(default, FieldInfo):
                    merged = FieldInfo.merge_field_infos(merged, default=default)
                elif isinstance(default, FieldInfo):
                    merged = FieldInfo.merge_field_infos(merged, default)
                fields[name] = (ann, merged)
            elif default == inspect._empty:
                # Required field
                fields[name] = (
                    ann,
                    Field(..., description=field_description),
                )
            elif isinstance(default, FieldInfo):
                # Parameter with a default value that is a Field(...)
                fields[name] = (
                    ann,
                    FieldInfo.merge_field_infos(
                        default, description=field_description or default.description
                    ),
                )
            else:
                # Parameter with a default value
                fields[name] = (
                    ann,
                    Field(default=default, description=field_description),
                )

    # 3. Dynamically build a Pydantic model
    dynamic_model = create_model(f"{func_name}_args", __base__=BaseModel, **fields)

    # 4. Build JSON schema from that model
    json_schema = dynamic_model.model_json_schema()
    if strict_json_schema:
        json_schema = ensure_strict_json_schema(json_schema)

    # 5. Return as a FuncSchema dataclass
    return FuncSchema(
        name=func_name,
        # Ensure description_override takes precedence even if docstring info is disabled.
        description=description_override or (doc_info.description if doc_info else None),
        params_pydantic_model=dynamic_model,
        params_json_schema=json_schema,
        signature=sig,
        takes_context=takes_context,
        strict_json_schema=strict_json_schema,
        return_annotation=type_hints_with_extras.get("return", sig.return_annotation),
    )
