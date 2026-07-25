from __future__ import annotations

import functools
import typing
from collections.abc import Callable
from types import UnionType
from typing import Annotated, Any, get_args, get_origin

from typing_extensions import NoDefault, TypeAliasType

_NATIVE_TYPE_ALIAS_TYPE = getattr(typing, "TypeAliasType", TypeAliasType)


def is_type_alias_type(value: Any) -> bool:
    """Return whether a value is a native or typing-extensions PEP 695 alias."""
    return isinstance(value, TypeAliasType | _NATIVE_TYPE_ALIAS_TYPE)


def expand_type_alias(annotation: Any, seen: set[Any] | None = None) -> Any:
    """Expand PEP 695 aliases while preserving outer ``Annotated`` metadata."""
    metadata: list[Any] = []
    plain_annotation = annotation
    while get_origin(plain_annotation) is Annotated:
        annotated_args = get_args(plain_annotation)
        if not annotated_args:
            break
        plain_annotation = annotated_args[0]
        metadata.extend(annotated_args[1:])

    origin = get_origin(plain_annotation)
    alias = origin if is_type_alias_type(origin) else plain_annotation
    if not is_type_alias_type(alias):
        return annotation

    seen_aliases = seen or set()
    if alias in seen_aliases:
        return annotation

    parameters = get_type_parameters(alias)
    provided_args = get_args(plain_annotation)
    substitutions: dict[Any, Any] = {}
    for index, parameter in enumerate(parameters):
        if index < len(provided_args):
            resolved_arg = provided_args[index]
        else:
            default = getattr(parameter, "__default__", NoDefault)
            resolved_arg = (
                Any if default is NoDefault else substitute_typevars(default, substitutions)
            )
        substitutions[parameter] = resolved_arg
    expanded = substitute_typevars(alias.__value__, substitutions)
    expanded = expand_type_alias(expanded, {*seen_aliases, alias})
    return Annotated[(expanded, *metadata)] if metadata else expanded


def get_type_parameters(owner: Any) -> tuple[Any, ...]:
    """Return type parameters declared by a legacy or PEP 695 generic type."""
    parameters = getattr(owner, "__type_params__", ()) or getattr(owner, "__parameters__", ())
    return parameters if isinstance(parameters, tuple) else ()


def get_callable_call_descriptor(func: Callable[..., Any]) -> tuple[type[Any], Any]:
    """Return the class that defines ``__call__`` and its raw descriptor."""
    for owner in type(func).__mro__:
        if "__call__" in owner.__dict__:
            return owner, owner.__dict__["__call__"]
    raise TypeError(f"{func!r} has no __call__ descriptor")


def unwrap_callable_descriptor(descriptor: Any) -> Any:
    """Return the callable behind supported method descriptors."""
    while True:
        if isinstance(descriptor, classmethod | staticmethod):
            descriptor = descriptor.__func__
        elif isinstance(descriptor, functools.partialmethod | functools.singledispatchmethod):
            descriptor = descriptor.func
        else:
            return descriptor


def substitute_typevars(annotation: Any, substitutions: dict[Any, Any]) -> Any:
    """Apply type-variable substitutions within a generic annotation."""
    try:
        substituted = substitutions.get(annotation, annotation)
    except TypeError:
        substituted = annotation
    if substituted is not annotation:
        return substituted

    if get_origin(annotation) is Annotated:
        annotated_type, *metadata = get_args(annotation)
        resolved_type = substitute_typevars(annotated_type, substitutions)
        if resolved_type is annotated_type:
            return annotation
        return Annotated[(resolved_type, *metadata)]

    args = get_args(annotation)
    if not args:
        return annotation
    resolved_args = tuple(substitute_typevars(arg, substitutions) for arg in args)
    if resolved_args == args:
        return annotation

    copy_with = getattr(annotation, "copy_with", None)
    if callable(copy_with):
        return copy_with(resolved_args)

    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin is UnionType:
        resolved_union = resolved_args[0]
        for resolved_arg in resolved_args[1:]:
            resolved_union |= resolved_arg
        return resolved_union
    try:
        return origin[resolved_args[0] if len(resolved_args) == 1 else resolved_args]
    except TypeError:
        return annotation


def resolve_typevar_substitutions(
    specialization: Any,
    target_owner: type[Any],
) -> dict[Any, Any]:
    """Resolve a specialized generic instance's type variables for an inherited owner."""
    specialization_origin = get_origin(specialization) or specialization
    if not isinstance(specialization_origin, type):
        return {}

    specialization_args = get_args(specialization)
    pydantic_metadata = getattr(
        specialization_origin,
        "__pydantic_generic_metadata__",
        None,
    )
    if isinstance(pydantic_metadata, dict):
        pydantic_origin = pydantic_metadata.get("origin")
        pydantic_args = pydantic_metadata.get("args")
        if isinstance(pydantic_origin, type) and isinstance(pydantic_args, tuple):
            specialization_origin = pydantic_origin
            specialization_args = pydantic_args

    initial_substitutions = dict(
        zip(
            get_type_parameters(specialization_origin),
            specialization_args,
            strict=False,
        )
    )

    def resolve_owner(
        owner: type[Any],
        substitutions: dict[Any, Any],
        seen: set[type[Any]],
    ) -> dict[Any, Any]:
        if owner is target_owner:
            return substitutions
        if owner in seen:
            return {}

        next_seen = {*seen, owner}
        bases = [
            *getattr(owner, "__bases__", ()),
            *getattr(owner, "__orig_bases__", ()),
        ]
        for base in bases:
            base_origin = get_origin(base) or base
            if not isinstance(base_origin, type):
                continue
            base_args = get_args(base)
            pydantic_base_metadata = getattr(
                base_origin,
                "__pydantic_generic_metadata__",
                None,
            )
            if isinstance(pydantic_base_metadata, dict):
                pydantic_base_origin = pydantic_base_metadata.get("origin")
                pydantic_base_args = pydantic_base_metadata.get("args")
                if isinstance(pydantic_base_origin, type) and isinstance(pydantic_base_args, tuple):
                    base_origin = pydantic_base_origin
                    base_args = pydantic_base_args
            base_args = tuple(substitute_typevars(arg, substitutions) for arg in base_args)
            base_substitutions = dict(
                zip(get_type_parameters(base_origin), base_args, strict=False)
            )
            resolved = resolve_owner(base_origin, base_substitutions, next_seen)
            if resolved:
                return resolved
        return {}

    return resolve_owner(specialization_origin, initial_substitutions, set())
