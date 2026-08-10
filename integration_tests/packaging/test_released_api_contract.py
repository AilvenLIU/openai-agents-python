import os
from importlib.metadata import version
from pathlib import Path

import pytest

from integration_tests._contract_support import (
    load_api_contract,
    validate_released_api_contract,
)

pytestmark = pytest.mark.packaging

CONTRACT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "released_api_contract.json"
PROSPECTIVE_CONTRACT_ENV = "OPENAI_AGENTS_PROSPECTIVE_RELEASE_CONTRACT"


def _prospective_contract_failure_message(errors: list[str]) -> str:
    details = "\n".join(f"- {error}" for error in errors)
    return (
        "Prospective release API contract check failed before release preparation.\n"
        "The installed distribution does not match the generated public API contract:\n"
        f"{details}\n\n"
        "If a missing name is absent from the clean module's `__all__` because it depends "
        "on an optional extra, add it to `tests/fixtures/released_api_contract_policy.json` "
        "under the module's `optional_exports` mapping. If the name remains in `__all__` "
        "but resolving its binding requires the optional dependency, add it under "
        "`optional_bindings` instead. If the export is required, make its defining module "
        "importable without the optional package. Then run `make sync` and "
        "`make check-prospective-released-api-contract` again."
    )


@pytest.mark.packaging_dependency
def test_installed_distribution_preserves_released_public_api_contract() -> None:
    contract = load_api_contract(CONTRACT)
    assert contract["baseline"] == f"v{version('openai-agents')}"
    assert len(contract["baseline_commit"]) == 40

    errors = validate_released_api_contract(
        contract,
        require_all_optional_dependencies=(
            os.environ.get("OPENAI_AGENTS_INTEGRATION_REQUIRE_OPTIONAL_EXPORTS") == "1"
        ),
    )

    assert errors == []


@pytest.mark.packaging_dependency
def test_installed_distribution_is_ready_for_prospective_release_contract() -> None:
    configured_path = os.environ.get(PROSPECTIVE_CONTRACT_ENV)
    if not configured_path:
        return

    path = Path(configured_path)
    if not path.is_file():
        pytest.fail(
            f"Prospective release API contract does not exist: {path}. "
            "Run `make check-prospective-released-api-contract` from the repository root."
        )

    contract = load_api_contract(path)
    errors = validate_released_api_contract(
        contract,
        require_all_optional_dependencies=(
            os.environ.get("OPENAI_AGENTS_INTEGRATION_REQUIRE_OPTIONAL_EXPORTS") == "1"
        ),
    )
    if errors:
        pytest.fail(_prospective_contract_failure_message(errors))


def test_prospective_contract_failure_guidance_distinguishes_optional_shapes() -> None:
    message = _prospective_contract_failure_message(
        [
            "Missing released agents.example exports: ['ConditionalExport']",
            "Missing released agents.example bindings: ['LazyBinding']",
        ]
    )

    assert "absent from the clean module's `__all__`" in message
    assert "`optional_exports`" in message
    assert "remains in `__all__`" in message
    assert "`optional_bindings` instead" in message
    assert "make check-prospective-released-api-contract" in message
