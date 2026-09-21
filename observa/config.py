import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

load_dotenv()


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    oracle_host: str
    oracle_port: int
    oracle_service_name: str
    oracle_user: str
    oracle_password: str
    oracle_dsn: str
    oracle_pool_min: int
    oracle_pool_max: int

    llm_provider: str
    llm_model: str
    llm_api_key: str

    # Per-agent overrides — model_id string keyed by agent name. Agents not
    # listed fall back to llm_model. Mixing providers is allowed (gpt-* and
    # claude-* can coexist; make_llm() routes by prefix).
    llm_agent_models: dict[str, str]
    # Master consolidator and CoVe verifier model overrides. Empty string
    # means "use llm_model".
    llm_consolidator_model: str
    llm_cove_model: str
    llm_master_chat_model: str

    # OCI GenAI provider — present only when llm.providers.oci section exists.
    oci_enabled: bool
    oci_default_model: str
    oci_compartment_id: str
    oci_service_endpoint: str
    oci_auth_profile: str
    oci_auth_type: str

    agents_enable_hypotheses: bool
    agents_enable_cove: bool
    agents_contradiction_threshold: float
    agents_verification_support_threshold: float
    llm_hypotheses_model: str

    turns_count: int
    turns_time_budget_seconds: int
    turns_max_concurrent_agents: int

    mcp_sqlcl_path: str
    mcp_sqlcl_connection_name: str
    mcp_query_row_limit: int
    mcp_file_read_byte_limit: int
    mcp_enforce_window: Literal["warn", "reject", "off"]

    # Case intake (Feature 1 — windows + timeline).
    case_timeline_days: int
    case_baseline_offset_days: int

    ui_theme: str
    ui_show_token_cost: bool
    ui_log_level: str

    hitl_chat_enabled: bool
    hitl_chat_max_chars: int

    agents_enabled: dict[str, bool]

    # Refinement (Level 5 — turn-to-turn agent selection).
    refinement_enabled: bool
    refinement_universal_agents: tuple[str, ...]
    refinement_keep_severities: frozenset[str]
    refinement_enable_cross_reference: bool
    refinement_min_active: int

    def refinement_cfg(self):
        from observa.graph.refinement import RefinementConfig
        return RefinementConfig(
            enabled=self.refinement_enabled,
            universal_agents=self.refinement_universal_agents,
            keep_severities=self.refinement_keep_severities,
            enable_cross_reference=self.refinement_enable_cross_reference,
            min_active=self.refinement_min_active,
        )

    def redacted_dict(self) -> dict:
        d = self.model_dump()
        d["oracle_password"] = "***"
        d["llm_api_key"] = "***"
        return d

    def model_for_agent(self, agent_name: str) -> str:
        """Resolve the LLM model id to use for a given agent.

        Returns the per-agent override from ``llm.agent_models`` when present,
        otherwise the default ``llm.providers[default_provider].model``.
        """
        return self.llm_agent_models.get(agent_name) or self.llm_model

    def model_for_consolidator(self) -> str:
        return self.llm_consolidator_model or self.llm_model

    def model_for_cove(self) -> str:
        return self.llm_cove_model or self.llm_model

    def model_for_hypotheses(self) -> str:
        return self.llm_hypotheses_model or self.llm_model

    def model_for_master_chat(self) -> str:
        return self.llm_master_chat_model or self.llm_model

    def referenced_models(self) -> list[str]:
        """Every model id that may be instantiated this session — for env-var checks."""
        models = {
            self.llm_model,
            self.model_for_hypotheses(),
            self.model_for_consolidator(),
            self.model_for_cove(),
            self.model_for_master_chat(),
        }
        models.update(self.llm_agent_models.values())
        return sorted(m for m in models if m)

    def required_providers(self) -> set[str]:
        """Infer the set of providers ('anthropic'/'openai') needed for this config."""
        provs: set[str] = set()
        for m in self.referenced_models():
            provs.add(provider_for_model(m))
        return provs


def provider_for_model(model: str) -> str:
    """Infer the provider from a model id by prefix.

    Routing rules (must match observa.llm.rate_limited.make_llm):
      * ``oci/<model_id>`` → "oci"
      * model contains ``claude``  → "anthropic"
      * everything else            → "openai"
    """
    if model.startswith("oci/"):
        return "oci"
    return "anthropic" if "claude" in model.lower() else "openai"


def get_settings(config_path: str | None = None) -> Settings:
    resolved = Path(config_path).resolve() if config_path else Path("config/config.yaml").resolve()
    return _load_settings(str(resolved))


@lru_cache(maxsize=8)
def _load_settings(resolved_path: str) -> Settings:
    resolved = Path(resolved_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    oracle = data.get("oracle", {})
    host = oracle.get("host", "")
    port = int(oracle.get("port", 1521))
    service_name = oracle.get("service_name", "")
    oracle_user = oracle.get("user", "")
    pwd_env = oracle.get("password_env", "ORACLE_PWD")
    oracle_password = os.environ.get("ORACLE_PWD", os.environ.get(pwd_env, ""))
    oracle_dsn = f"{host}:{port}/{service_name}"

    llm = data.get("llm", {})
    provider = llm.get("default_provider", "anthropic")
    providers = llm.get("providers", {})
    provider_cfg = providers.get(provider, {})
    llm_model = provider_cfg.get("model", "")
    # When the default provider is OCI, the model id needs the "oci/" routing
    # prefix so make_llm dispatches to ChatOCIGenAI. Accept both bare
    # ("xai.grok-4-fast") and prefixed ("oci/xai.grok-4-fast") in the YAML to
    # avoid a foot-gun where llm_model leaks into make_llm without routing.
    if provider == "oci" and llm_model and not llm_model.startswith("oci/"):
        llm_model = f"oci/{llm_model}"
    api_key_env = provider_cfg.get("api_key_env", "")
    llm_api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    oci_cfg = providers.get("oci") or {}
    oci_enabled = bool(oci_cfg)
    oci_default_model = str(oci_cfg.get("model", "") or "")
    oci_compartment_id = str(oci_cfg.get("compartment_id", "") or "")
    oci_service_endpoint = str(oci_cfg.get("service_endpoint", "") or "")
    oci_auth_profile = str(oci_cfg.get("auth_profile", "DEFAULT") or "DEFAULT")
    oci_auth_type = str(oci_cfg.get("auth_type", "API_KEY") or "API_KEY")

    raw_agent_models = llm.get("agent_models") or {}
    llm_agent_models: dict[str, str] = {
        str(k): str(v) for k, v in raw_agent_models.items() if isinstance(v, str) and v
    }
    llm_consolidator_model = str(llm.get("consolidator_model", "") or "")
    llm_cove_model = str(llm.get("cove_model", "") or "")
    llm_master_chat_model = str(llm.get("master_chat_model", "") or "")

    agents = data.get("agents", {}) or {}
    ui = data.get("ui", {}) or {}
    hitl = data.get("hitl", {}) or {}
    turns = data.get("turns", {}) or {}
    mcp = data.get("mcp", {}) or {}
    case = data.get("case", {}) or {}
    enforce_window = str(mcp.get("enforce_window", "warn") or "warn").lower()
    if enforce_window not in ("warn", "reject", "off"):
        enforce_window = "warn"

    known_agents = (
        "wait_time", "ash", "infra", "sql", "memory", "io_storage",
        "concurrency", "segment_object", "rac", "exadata", "dg",
        "file_reader",
    )
    agents_enabled_raw = agents.get("enabled") or {}
    agents_enabled = {name: bool(agents_enabled_raw.get(name, True)) for name in known_agents}

    refinement = agents.get("refinement") or {}
    refinement_enabled = bool(refinement.get("enabled", False))
    refinement_universal = tuple(
        str(x) for x in (refinement.get("universal_agents") or ["wait_time", "ash"])
    )
    refinement_keep_sev = frozenset(
        str(x) for x in (refinement.get("keep_severities") or ["high", "critical"])
    )
    refinement_xref = bool(refinement.get("enable_cross_reference", True))
    refinement_min_active = int(refinement.get("min_active", 3))

    return Settings(
        oracle_host=host,
        oracle_port=port,
        oracle_service_name=service_name,
        oracle_user=oracle_user,
        oracle_password=oracle_password,
        oracle_dsn=oracle_dsn,
        oracle_pool_min=int(oracle.get("pool_min", 2)),
        oracle_pool_max=int(oracle.get("pool_max", 10)),
        llm_provider=provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_agent_models=llm_agent_models,
        llm_consolidator_model=llm_consolidator_model,
        llm_cove_model=llm_cove_model,
        llm_master_chat_model=llm_master_chat_model,
        oci_enabled=oci_enabled,
        oci_default_model=oci_default_model,
        oci_compartment_id=oci_compartment_id,
        oci_service_endpoint=oci_service_endpoint,
        oci_auth_profile=oci_auth_profile,
        oci_auth_type=oci_auth_type,
        agents_enable_hypotheses=bool(agents.get("enable_hypotheses", True)),
        agents_enable_cove=bool(agents.get("enable_cove", True)),
        agents_contradiction_threshold=float(agents.get("contradiction_threshold", 0.3)),
        agents_verification_support_threshold=float(
            agents.get("verification_support_threshold", 0.7)
        ),
        llm_hypotheses_model=str(llm.get("hypotheses_model", "") or ""),
        turns_count=int(turns.get("count", 3)),
        turns_time_budget_seconds=int(turns.get("time_budget_seconds", 120)),
        turns_max_concurrent_agents=int(turns.get("max_concurrent_agents", 4)),
        mcp_sqlcl_path=str(mcp.get("sqlcl_path", "")),
        mcp_sqlcl_connection_name=str(mcp.get("sqlcl_connection_name", "")),
        mcp_query_row_limit=int(mcp.get("query_row_limit", 500)),
        mcp_file_read_byte_limit=int(mcp.get("file_read_byte_limit", 65536)),
        mcp_enforce_window=enforce_window,
        case_timeline_days=int(case.get("timeline_days", 15)),
        case_baseline_offset_days=int(case.get("baseline_offset_days", 7)),
        ui_theme=ui.get("theme", "flexoki"),
        ui_show_token_cost=bool(ui.get("show_token_cost", True)),
        ui_log_level=ui.get("log_level", "INFO"),
        hitl_chat_enabled=bool(hitl.get("chat_enabled", True)),
        hitl_chat_max_chars=int(hitl.get("chat_max_chars", 2000)),
        agents_enabled=agents_enabled,
        refinement_enabled=refinement_enabled,
        refinement_universal_agents=refinement_universal,
        refinement_keep_severities=refinement_keep_sev,
        refinement_enable_cross_reference=refinement_xref,
        refinement_min_active=refinement_min_active,
    )
