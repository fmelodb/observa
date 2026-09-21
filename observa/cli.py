# observa/cli.py
import asyncio

import typer
import yaml

app = typer.Typer(name="observa", help="Oracle performance troubleshooting CLI")


# ---------------------------------------------------------------------------
# Top-level commands
# ---------------------------------------------------------------------------


@app.command()
def new(debug: bool = typer.Option(False, "--debug", help="Write debug log to observa-debug.log")) -> None:
    """Start a new performance case (launches TUI)."""
    if debug:
        import logging

        # File handler captures EVERYTHING from observa.* and a few key third-party
        # modules. Using basicConfig on the root would also capture Textual's very
        # chatty internals — route explicitly instead.
        handler = logging.FileHandler("observa-debug.log", mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        handler.setLevel(logging.DEBUG)
        for logger_name in ("observa", "mcp", "langgraph", "langchain"):
            lg = logging.getLogger(logger_name)
            lg.setLevel(logging.DEBUG)
            lg.addHandler(handler)
            lg.propagate = False
        logging.getLogger("observa").info("debug logging enabled → observa-debug.log")

    from observa.config import get_settings
    from observa.llm.oci_capabilities import ConfigError as _LLMConfigError
    from observa.llm.validate_config import validate_llm_config

    settings = get_settings()
    try:
        validate_llm_config(settings)
    except _LLMConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    from observa.tui.app import ObservaApp

    ObservaApp().run()


@app.command()
def chat() -> None:
    """Placeholder for future CLI-only chat mode."""
    typer.echo(
        "Observa 'chat' command is not wired in this phase — "
        "use 'observa new' to launch the TUI which has the post-analysis chat panel."
    )
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# config sub-commands
# ---------------------------------------------------------------------------

config_app = typer.Typer()
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show() -> None:
    """Display current config with secrets redacted."""
    from observa.config import get_settings

    settings = get_settings()
    typer.echo(yaml.dump(settings.redacted_dict(), default_flow_style=False, sort_keys=True))


# ---------------------------------------------------------------------------
# setup sub-commands
# ---------------------------------------------------------------------------

setup_app = typer.Typer()
app.add_typer(setup_app, name="setup")


def _get_settings_safe():
    """Return Settings or a minimal stub if config file is missing."""
    try:
        from observa.config import get_settings

        return get_settings()
    except Exception:  # noqa: BLE001
        return None


@setup_app.command("check")
def setup_check() -> None:
    """Verify Python dependencies, env vars, config file, and Oracle connectivity."""
    from observa.setup_check import run_all_checks

    settings = _get_settings_safe()
    results = asyncio.run(run_all_checks(settings))

    any_fail = False
    for r in results:
        if r.status == "ok":
            icon = "[OK]  "
        elif r.status == "warn":
            icon = "[WARN]"
        else:
            icon = "[FAIL]"
            any_fail = True
        typer.echo(f"  {icon} {r.name}: {r.message}")

    if any_fail:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
