"""Pure request-to-serving-mode router for the sampled-mode kernel admission.

r[impl tenstorrent.qwen38_p150x2.sampled_mode_kernel_admission]

A sampled-only admission receipt makes the fused GDN kernel eligible for
requests that sample. Deterministic requests keep the exact-parity path.
This core owns that classification and the switch decision; the service
shell owns the trace release, attribute flip, and recapture.

The physical primitive is measured and proven: capturing a second decode
trace while another is resident corrupts the second trace's outputs, so
dual residency is falsified on runtime 0.74. Releasing the resident trace
before the next capture is exact in both directions at 0.39-0.75 seconds
per switch (see the change evidence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ROUTE_DETERMINISTIC = "off"
ROUTE_SAMPLED = "full"
ROUTES = (ROUTE_DETERMINISTIC, ROUTE_SAMPLED)

# Admission receipts name serving modes, not kernel routes: a distributional
# receipt lists "sampled", while the adapter flips between "off" and "full".
ADMITTED_SAMPLED_MODE = "sampled"

ZERO_TEMPERATURE = 0.0


class RouterPolicyError(ValueError):
    """The request temperature violates the closed routing policy."""


@dataclass(frozen=True)
class RoutingDecision:
    """One request's serving route and the switch it requires."""

    route: str
    current_mode: str
    switch_required: bool

    def validate(self) -> None:
        if self.route not in ROUTES:
            raise RouterPolicyError(
                f"route must be one of {ROUTES}, got {self.route!r}"
            )
        if self.current_mode not in ROUTES:
            raise RouterPolicyError(
                f"current mode must be one of {ROUTES}, got {self.current_mode!r}"
            )


def classify_temperature(temperature) -> str:
    """Map one request temperature to a serving route.

    Missing and zero temperatures are deterministic: they must keep the
    exact-parity path. Positive finite temperatures are sampled. Anything
    else fails closed with a bounded reason.
    """
    if temperature is None:
        return ROUTE_DETERMINISTIC
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise RouterPolicyError("temperature must be a number or null")
    value = float(temperature)
    if not math.isfinite(value):
        raise RouterPolicyError("temperature must be finite")
    if value < ZERO_TEMPERATURE:
        raise RouterPolicyError("temperature must not be negative")
    if value == ZERO_TEMPERATURE:
        return ROUTE_DETERMINISTIC
    return ROUTE_SAMPLED


def route_request(temperature, *, current_mode: str) -> RoutingDecision:
    """Classify one request against the adapter's resident fusion mode."""
    route = classify_temperature(temperature)
    decision = RoutingDecision(
        route=route,
        current_mode=current_mode,
        switch_required=route != current_mode,
    )
    decision.validate()
    return decision


def resolve_batch_route(
    temperatures,
    *,
    sampled_admitted: bool,
    current_mode: str,
) -> RoutingDecision:
    """Resolve one batch's serving route from its request temperatures.

    A batch runs one fusion mode per step, so any deterministic request
    forces the whole batch onto the exact-parity path. Sampled users in a
    mixed batch lose the speedup but keep correct output; deterministic
    users never see the fused kernel. Without a sampled admission the only
    permitted route is deterministic.
    """
    temperatures = tuple(temperatures)
    if not temperatures:
        raise RouterPolicyError("a batch needs at least one request")
    routes = tuple(classify_temperature(temperature) for temperature in temperatures)
    if not sampled_admitted or any(
        route == ROUTE_DETERMINISTIC for route in routes
    ):
        batch_route = ROUTE_DETERMINISTIC
    else:
        batch_route = ROUTE_SAMPLED
    decision = RoutingDecision(
        route=batch_route,
        current_mode=current_mode,
        switch_required=batch_route != current_mode,
    )
    decision.validate()
    return decision


def eligible_routes_for_admitted_modes(admitted_modes) -> tuple[str, ...]:
    """Return the routes a receipt's admitted serving modes permit.

    An empty or non-sampled admission list permits only the deterministic
    route, so an unadmitted or rejected candidate can never serve sampled
    traffic through this router.
    """
    if any(
        mode in (ROUTE_SAMPLED, ADMITTED_SAMPLED_MODE) for mode in admitted_modes
    ):
        return ROUTES
    return (ROUTE_DETERMINISTIC,)
