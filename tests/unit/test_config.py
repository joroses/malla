# New unit tests for configuration loader

from pathlib import Path

import pytest

from malla.config import AppConfig, _clear_config_cache, load_config


def test_yaml_loading(tmp_path: Path, monkeypatch):
    """Ensure that values from a YAML file are loaded into AppConfig."""

    # Clear any cached config from other imports
    _clear_config_cache()

    # Clear any environment variables that might override the YAML
    monkeypatch.delenv("MALLA_NAME", raising=False)
    monkeypatch.delenv("MALLA_PORT", raising=False)
    monkeypatch.delenv("MALLA_HOME_MARKDOWN", raising=False)

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("""
name: CustomName
home_markdown: "# Welcome\nThis is **markdown** content."
port: 9999
""")

    cfg = load_config(config_path=yaml_file)

    assert isinstance(cfg, AppConfig)
    assert cfg.name == "CustomName"
    assert "markdown" in cfg.home_markdown
    assert cfg.port == 9999


def test_env_override(monkeypatch):
    """Environment variables with the `MALLA_` prefix override YAML/defaults."""

    # Clear any cached config from other imports
    _clear_config_cache()

    monkeypatch.setenv("MALLA_NAME", "EnvName")
    monkeypatch.setenv("MALLA_DEBUG", "true")
    cfg = load_config(config_path=None)

    assert cfg.name == "EnvName"
    assert cfg.debug is True


class TestLoRaSettings:
    def test_defaults(self, tmp_path, monkeypatch):
        _clear_config_cache()
        monkeypatch.delenv("MALLA_LORA_PRESET", raising=False)
        monkeypatch.delenv("MALLA_LORA_SPREADING_FACTOR", raising=False)

        cfg = load_config(config_path=tmp_path / "does-not-exist.yaml")

        assert cfg.lora_preset == "LongFast"
        assert cfg.lora_spreading_factor is None

    def test_yaml_loading(self, tmp_path, monkeypatch):
        _clear_config_cache()
        monkeypatch.delenv("MALLA_LORA_PRESET", raising=False)
        monkeypatch.delenv("MALLA_LORA_SPREADING_FACTOR", raising=False)

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("lora_preset: LongSlow\nlora_spreading_factor: 9\n")

        cfg = load_config(config_path=yaml_file)

        assert cfg.lora_preset == "LongSlow"
        assert cfg.lora_spreading_factor == 9

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        _clear_config_cache()
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("lora_preset: LongSlow\nlora_spreading_factor: 9\n")

        monkeypatch.setenv("MALLA_LORA_PRESET", "ShortFast")
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", "8")

        cfg = load_config(config_path=yaml_file)

        assert cfg.lora_preset == "ShortFast"
        assert cfg.lora_spreading_factor == 8

    @pytest.mark.parametrize("raw", ["", "   "])
    def test_blank_optional_env_sf_is_none(self, tmp_path, monkeypatch, raw):
        """Compose forwards MALLA_LORA_SPREADING_FACTOR="" when unset."""
        _clear_config_cache()
        monkeypatch.delenv("MALLA_LORA_PRESET", raising=False)
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", raw)

        cfg = load_config(config_path=tmp_path / "does-not-exist.yaml")

        assert cfg.lora_spreading_factor is None

    def test_invalid_optional_env_sf_is_none_not_a_string(self, tmp_path, monkeypatch):
        """The coercer must not leave invalid integer input as a raw string."""
        _clear_config_cache()
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", "banana")

        cfg = load_config(config_path=tmp_path / "does-not-exist.yaml")

        assert cfg.lora_spreading_factor is None

    def test_invalid_required_int_keeps_historic_raw_string(
        self, tmp_path, monkeypatch
    ):
        """Non-optional fields keep the pre-existing raw-string fallback."""
        _clear_config_cache()
        monkeypatch.setenv("MALLA_PORT", "not-a-port")

        cfg = load_config(config_path=tmp_path / "does-not-exist.yaml")

        assert cfg.port == "not-a-port"

    @pytest.mark.parametrize(
        ("raw_sf", "expected_sf"),
        [
            ("sf7", 7),
            ("SF8", 8),
            ("sf11", 11),
            ("12", 12),
            ("  sf7  ", 7),
        ],
    )
    def test_documented_sfn_env_formats_coerce_to_int(
        self, tmp_path, monkeypatch, raw_sf, expected_sf
    ):
        """Documented sfN formats must coerce to integers without being discarded."""
        _clear_config_cache()
        monkeypatch.setenv("MALLA_LORA_SPREADING_FACTOR", raw_sf)

        cfg = load_config(config_path=tmp_path / "does-not-exist.yaml")

        assert cfg.lora_spreading_factor == expected_sf

    def test_yaml_lora_settings_preserved_when_compose_env_unset(
        self, tmp_path, monkeypatch
    ):
        """When Compose forwards only explicitly set variables and they are unset on the host,
        mounted YAML configuration is preserved without being overridden by LongFast / empty SF.
        """
        _clear_config_cache()
        monkeypatch.delenv("MALLA_LORA_PRESET", raising=False)
        monkeypatch.delenv("MALLA_LORA_SPREADING_FACTOR", raising=False)

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("lora_preset: ShortFast\nlora_spreading_factor: 8\n")

        cfg = load_config(config_path=yaml_file)

        assert cfg.lora_preset == "ShortFast"
        assert cfg.lora_spreading_factor == 8

    @pytest.mark.parametrize("yaml_sf,expected_sf", [("sf8", 8), ("SF11", 11), (9, 9)])
    def test_yaml_sfn_spreading_factor_normalizes(
        self, tmp_path, monkeypatch, yaml_sf, expected_sf
    ):
        """YAML files containing sfN formatted spreading factors normalize to integers."""
        _clear_config_cache()
        monkeypatch.delenv("MALLA_LORA_PRESET", raising=False)
        monkeypatch.delenv("MALLA_LORA_SPREADING_FACTOR", raising=False)

        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(f"lora_preset: ShortFast\nlora_spreading_factor: {yaml_sf}\n")

        cfg = load_config(config_path=yaml_file)

        assert cfg.lora_spreading_factor == expected_sf

