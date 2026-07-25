from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast, get_args, get_origin, get_type_hints

# griffelib exposes the `griffe` package at runtime but currently does not ship typing markers.
from griffe import Docstring, DocstringSectionKind  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from ._callable_utils import (
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


def _is_context_annotation(annotation: Any) -> bool:
    """Return whether an annotation denotes an injected function-tool context."""
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


def _get_callable_name(func: Callable[..., Any]) -> str:
    """Return a stable name for functions and callable objects."""
    name = getattr(func, "__name__", None)
    if isinstance(name, str):
        return name
    if isinstance(func, functools.partial):
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            return _get_callable_name(unwrapped)
        return _get_callable_name(func.func)
    return type(func).__name__


def _get_callable_doc(func: Callable[..., Any]) -> str | None:
    """Return documentation from a callable or its invocation method."""
    if isinstance(func, functools.partial):
        published_doc = vars(func).get("__doc__", _CONTEXT_NOT_PROVIDED)
        if published_doc is not _CONTEXT_NOT_PROVIDED:
            return inspect.cleandoc(published_doc) if isinstance(published_doc, str) else None
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            return _get_callable_doc(unwrapped)
        return _get_callable_doc(func.func)
    doc = inspect.getdoc(func)
    if doc is not None or inspect.isroutine(func) or inspect.isclass(func):
        return doc
    _, call_descriptor = get_callable_call_descriptor(func)
    return inspect.getdoc(unwrap_callable_descriptor(call_descriptor))


def _get_callable_signature(func: Callable[..., Any]) -> inspect.Signature:
    """Return the effective invocation signature without double-binding descriptors."""
    if inspect.isroutine(func) or inspect.isclass(func) or isinstance(func, functools.partial):
        return inspect.signature(func)

    try:
        instance_vars = vars(func)
    except TypeError:
        instance_vars = {}
    if "__signature__" in instance_vars or "__wrapped__" in instance_vars:
        return inspect.signature(func)

    return inspect.signature(cast(Any, func).__call__)


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
    globalns, localns = _get_callable_annotation_namespaces(func)
    resolved_hints = get_type_hints(
        annotation_source,
        globalns=globalns,
        localns=localns,
        include_extras=True,
    )
    owner = func if inspect.isclass(func) else type(func)
    return _apply_callable_type_specialization(resolved_hints, func, owner)


def _get_callable_annotation_namespaces(
    func: Callable[..., Any],
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

    if inspect.isclass(namespace_source):
        local_owner: type[Any] | None = namespace_source
    elif inspect.isroutine(namespace_source):
        local_owner = None
    else:
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
    if not substitutions:
        return type_hints
    return {
        name: substitute_typevars(annotation, substitutions)
        for name, annotation in type_hints.items()
    }


def _get_callable_type_hints(
    func: Callable[..., Any], signature: inspect.Signature
) -> dict[str, Any]:
    """Resolve callable hints using signature, published, wrapped, then structural metadata."""
    if isinstance(getattr(func, "__signature__", None), inspect.Signature):
        return _get_signature_type_hints(func, signature)

    try:
        instance_vars = vars(func)
    except TypeError:
        instance_vars = {}

    if "__annotations__" in instance_vars or "__annotate__" in instance_vars:
        globalns, localns = _get_callable_annotation_namespaces(func)
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
        return _apply_callable_type_specialization(published_hints, func, type(func))

    if isinstance(func, functools.partial):
        unwrapped = inspect.unwrap(func)
        if unwrapped is not func:
            return _get_callable_type_hints(unwrapped, signature)
        wrapped_signature = inspect.signature(func.func)
        partial_type_hints = _get_callable_type_hints(func.func, wrapped_signature)
        _validate_no_positionally_bound_context(
            wrapped_signature,
            partial_type_hints,
            positional_args=func.args,
            implicit_positional_count=0,
            partial_kind="functools.partial",
        )
        return {
            name: annotation
            for name, annotation in partial_type_hints.items()
            if name == "return" or name in signature.parameters
        }

    if inspect.isroutine(func) or inspect.isclass(func):
        return get_type_hints(func, include_extras=True)

    unwrapped = inspect.unwrap(func)
    if unwrapped is not func:
        return _get_callable_type_hints(unwrapped, signature)

    call_owner, call_descriptor = get_callable_call_descriptor(func)
    call_method = unwrap_callable_descriptor(call_descriptor)
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
    if isinstance(call_descriptor, functools.partialmethod):
        implicit_positional_count = 0 if isinstance(call_descriptor.func, staticmethod) else 1
        _validate_no_positionally_bound_context(
            inspect.signature(call_method),
            call_type_hints,
            positional_args=call_descriptor.args,
            implicit_positional_count=implicit_positional_count,
            partial_kind="functools.partialmethod",
        )
        return {
            name: annotation
            for name, annotation in call_type_hints.items()
            if name == "return" or name in signature.parameters
        }
    return call_type_hints


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
    name = _get_callable_name(func)
    doc = _get_callable_doc(func)
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

    # 1. Grab docstring info
    if use_docstring_info:
        doc_info = generate_func_documentation(func, docstring_style)
        param_descs = dict(doc_info.param_descriptions or {})
    else:
        doc_info = None
        param_descs = {}

    sig = _get_callable_signature(func)
    type_hints_with_extras = _get_callable_type_hints(func, sig)
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
    func_name = name_override or (doc_info.name if doc_info else _get_callable_name(func))

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
