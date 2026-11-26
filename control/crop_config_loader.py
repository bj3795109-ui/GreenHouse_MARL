"""
Dynamic external crop configuration and zone-assignment loader.

Design invariants:
- Crop targets are NEVER hardcoded inside PIDController.
- Crop requirements and PID tuning parameters are strictly decoupled; changing
  crops or growth stages never modifies PID gains.
- Units are validated explicitly; incompatible soil-moisture units are rejected.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping, Optional

from control.pid_controller import validate_pid_gains
from control.types import CropBounds, PIDGains, VARS, ZONE_IDS

logger = logging.getLogger(__name__)


class CropConfigError(ValueError):
    """Raised when crop dataset, units, stage, or tuning configs are invalid."""


DEFAULT_CANONICAL_UNITS: dict[str, str] = {
    "T": "degC",
    "H": "RH_percent",
    "M": "relative_percent",
}

DEFAULT_SAFE_BOUNDS: dict[str, CropBounds] = {
    "T": CropBounds(
        crop_id="__safe_default__",
        stage="survival",
        var="T",
        unit="degC",
        min_acceptable=14.0,
        optimal_min=18.0,
        optimal_max=25.0,
        max_acceptable=32.0,
        is_fallback=True,
    ),
    "H": CropBounds(
        crop_id="__safe_default__",
        stage="survival",
        var="H",
        unit="RH_percent",
        min_acceptable=35.0,
        optimal_min=50.0,
        optimal_max=70.0,
        max_acceptable=85.0,
        is_fallback=True,
    ),
    "M": CropBounds(
        crop_id="__safe_default__",
        stage="survival",
        var="M",
        unit="relative_percent",
        min_acceptable=20.0,
        optimal_min=35.0,
        optimal_max=55.0,
        max_acceptable=75.0,
        is_fallback=True,
    ),
}


def _convert_unit(value: float, src_unit: str, dst_unit: str, var: str) -> float:
    if src_unit == dst_unit:
        return value
    if var == "T" and src_unit == "degF" and dst_unit == "degC":
        return (value - 32.0) * (5.0 / 9.0)
    raise CropConfigError(
        f"Incompatible unit for {var}: got '{src_unit}', expected '{dst_unit}'. "
        "Refusing implicit conversion."
    )


def _validate_var_entry(
    crop_id: str,
    stage: str,
    var: str,
    entry: Mapping[str, Any],
    canonical_unit: str,
    is_fallback: bool = False,
) -> CropBounds:
    required_keys = ("unit", "min_acceptable", "optimal_min", "optimal_max", "max_acceptable")
    for k in required_keys:
        if k not in entry:
            raise CropConfigError(f"{crop_id}.{stage}.{var} missing required field '{k}'")

    src_unit = str(entry["unit"])
    vals: dict[str, float] = {}
    for k in ("min_acceptable", "optimal_min", "optimal_max", "max_acceptable"):
        try:
            raw_num = float(entry[k])
        except (TypeError, ValueError) as exc:
            raise CropConfigError(f"{crop_id}.{stage}.{var}.{k} must be numeric") from exc
        if not math.isfinite(raw_num):
            raise CropConfigError(f"{crop_id}.{stage}.{var}.{k} must be finite")
        vals[k] = _convert_unit(raw_num, src_unit, canonical_unit, var)

    if not (
        vals["min_acceptable"]
        <= vals["optimal_min"]
        <= vals["optimal_max"]
        <= vals["max_acceptable"]
    ):
        raise CropConfigError(
            f"{crop_id}.{stage}.{var} violates ordering "
            "min_acceptable <= optimal_min <= optimal_max <= max_acceptable"
        )

    return CropBounds(
        crop_id=crop_id,
        stage=stage,
        var=var,
        unit=canonical_unit,
        min_acceptable=vals["min_acceptable"],
        optimal_min=vals["optimal_min"],
        optimal_max=vals["optimal_max"],
        max_acceptable=vals["max_acceptable"],
        is_fallback=is_fallback,
    )


class CropConfigLoader:
    """Loads validated crop profiles and manages per-zone crop/stage assignments."""

    def __init__(
        self,
        dataset: Mapping[str, Any],
        canonical_units: Optional[Mapping[str, str]] = None,
        allow_safe_default: bool = True,
    ):
        self.canonical_units = dict(canonical_units or DEFAULT_CANONICAL_UNITS)
        self.allow_safe_default = allow_safe_default
        self.metadata: dict[str, Any] = {}
        self.profiles: dict[str, dict[str, dict[str, CropBounds]]] = {}
        self.zone_assignments: dict[str, tuple[str, str]] = {}
        self._load_dataset(dataset)

    @classmethod
    def from_json_file(
        cls,
        path: str | Path,
        canonical_units: Optional[Mapping[str, str]] = None,
        allow_safe_default: bool = True,
    ) -> "CropConfigLoader":
        p = Path(path)
        if not p.exists():
            raise CropConfigError(f"Crop dataset file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, canonical_units=canonical_units, allow_safe_default=allow_safe_default)

    def _load_dataset(self, dataset: Mapping[str, Any]) -> None:
        if "crops" not in dataset or not isinstance(dataset["crops"], Mapping):
            raise CropConfigError("Crop dataset must contain a 'crops' mapping")

        self.metadata = {
            "schema_version": dataset.get("schema_version", "unknown"),
            "dataset_source": dataset.get("dataset_source", "unspecified"),
            "soil_moisture_basis": dataset.get("soil_moisture_basis", "unconfirmed"),
        }

        ds_units = dataset.get("canonical_units", {})
        if ds_units:
            for v in VARS:
                if v in ds_units and ds_units[v] != self.canonical_units[v]:
                    raise CropConfigError(
                        f"Dataset canonical unit for {v} is '{ds_units[v]}', "
                        f"incompatible with system unit '{self.canonical_units[v]}'"
                    )

        parsed: dict[str, dict[str, dict[str, CropBounds]]] = {}
        for crop_id, crop_body in dataset["crops"].items():
            stages = crop_body.get("stages")
            if not isinstance(stages, Mapping) or not stages:
                raise CropConfigError(f"Crop '{crop_id}' must define non-empty 'stages'")
            parsed[crop_id] = {}
            for stage_name, stage_body in stages.items():
                parsed[crop_id][stage_name] = {}
                for v in VARS:
                    if v not in stage_body:
                        raise CropConfigError(f"{crop_id}.{stage_name} missing variable '{v}'")
                    parsed[crop_id][stage_name][v] = _validate_var_entry(
                        crop_id, stage_name, v, stage_body[v], self.canonical_units[v]
                    )
        self.profiles = parsed

    def assign_zone(self, zone_id: str, crop_id: str, stage: str) -> None:
        """Assign or update a zone's crop and growth stage atomically."""
        if zone_id not in ZONE_IDS:
            raise CropConfigError(f"Unknown zone_id '{zone_id}'")
        if crop_id not in self.profiles:
            raise CropConfigError(f"Crop '{crop_id}' not found in dataset")
        if stage not in self.profiles[crop_id]:
            raise CropConfigError(f"Stage '{stage}' not found for crop '{crop_id}'")
        self.zone_assignments[zone_id] = (crop_id, stage)

    def assign_zones_from_dict(self, mapping: Mapping[str, Mapping[str, str]]) -> None:
        zone_map = mapping["zones"] if "zones" in mapping and isinstance(mapping["zones"], Mapping) else mapping
        pending: list[tuple[str, str, str]] = []
        for zone_id, spec in zone_map.items():
            stage = spec.get("stage", spec.get("growth_stage"))
            if "crop_id" not in spec or stage is None:
                raise CropConfigError(f"Zone assignment for '{zone_id}' requires 'crop_id' and 'stage'/'growth_stage'")
            if zone_id not in ZONE_IDS:
                raise CropConfigError(f"Unknown zone_id '{zone_id}'")
            if spec["crop_id"] not in self.profiles:
                raise CropConfigError(f"Crop '{spec['crop_id']}' not found in dataset")
            if stage not in self.profiles[spec["crop_id"]]:
                raise CropConfigError(f"Stage '{stage}' not found for crop '{spec['crop_id']}'")
            pending.append((zone_id, spec["crop_id"], stage))
        for zone_id, crop_id, stage in pending:
            self.zone_assignments[zone_id] = (crop_id, stage)

    def assign_zones_from_json_file(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise CropConfigError(f"Zone assignment file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assign_zones_from_dict(data)

    def update_zone_stage(self, zone_id: str, new_stage: str) -> None:
        if zone_id not in self.zone_assignments:
            raise CropConfigError(f"Zone '{zone_id}' has no assigned crop")
        crop_id, _ = self.zone_assignments[zone_id]
        self.assign_zone(zone_id, crop_id, new_stage)

    def get_zone_targets(self, zone_id: str) -> dict[str, CropBounds]:
        """Return per-variable CropBounds for `zone_id`."""
        assignment = self.zone_assignments.get(zone_id)
        if assignment is None:
            if not self.allow_safe_default:
                raise CropConfigError(f"No crop assigned to '{zone_id}'")
            logger.warning("Missing crop assignment for %s; using safe survival defaults.", zone_id)
            return dict(DEFAULT_SAFE_BOUNDS)

        crop_id, stage = assignment
        crop_entry = self.profiles.get(crop_id)
        if crop_entry is None or stage not in crop_entry:
            if not self.allow_safe_default:
                raise CropConfigError(f"Missing profile for {zone_id}: {crop_id}/{stage}")
            logger.warning(
                "Missing crop profile %s/%s for %s; using safe survival defaults.",
                crop_id,
                stage,
                zone_id,
            )
            return dict(DEFAULT_SAFE_BOUNDS)

        return dict(crop_entry[stage])


def load_pid_tuning(source: str | Path | Mapping[str, Any]) -> dict[str, dict[str, PIDGains]]:
    """
    Load per-zone, per-variable PID gains strictly separately from crop configs.
    Accepts either a JSON file path or an in-memory mapping.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise CropConfigError(f"PID tuning file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = dict(source)

    out: dict[str, dict[str, PIDGains]] = {}
    for z in ZONE_IDS:
        z_cfg = data.get(z, data.get("default"))
        if z_cfg is None:
            raise CropConfigError(f"PID tuning missing entry for '{z}' and no 'default' provided")
        out[z] = {}
        for v in VARS:
            if v not in z_cfg:
                raise CropConfigError(f"PID tuning for '{z}' missing variable '{v}'")
            params = z_cfg[v]
            try:
                gains = PIDGains(
                    Kp=float(params["Kp"]),
                    Ki=float(params["Ki"]),
                    Kd=float(params.get("Kd", 0.0)),
                    tau_f=float(params.get("tau_f", 2.0)),
                    T_t=float(params.get("T_t", 10.0)),
                    Ts=float(params.get("Ts", 1.0)),
                    dt=float(params.get("dt", 1.0)),
                    u_min=float(params.get("u_min", 0.0 if v == "M" else -1.0)),
                    u_max=float(params.get("u_max", 1.0)),
                    unidirectional_positive=bool(params.get("unidirectional_positive", v == "M")),
                )
                validate_pid_gains(gains)
            except (TypeError, ValueError) as exc:
                raise CropConfigError(f"Invalid PID tuning for '{z}.{v}': {exc}") from exc
            out[z][v] = gains
    return out
