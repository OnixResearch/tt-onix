## Context

The Qwen3.8 package exposes three opt-in weight gates:

```text
QWEN36_LM_HEAD_BF4
QWEN36_MLP_DOWN_BF4
QWEN36_GDN_BF4
```

The production systemd service does not set these variables.
The documented BFP8 defaults therefore remain active.

No candidate has a complete, immutable admission receipt.

## Goal and Success Contract

The goal is an explicit unadmitted disposition for every current BF4 candidate.

The disposition is complete when:

- the production environment and BFP8 default are recorded;
- existing BF4 gates remain experimental and opt-in;
- no quality, memory, or speedup claim is added;
- future admission requires a new validated change and complete evidence;
- Cairn validation and lifecycle gates pass.

## Decisions

### Decision: Keep BFP8 as the production profile

**Choice:** Admit no BF4 candidate from the available evidence.

**Rationale:** No candidate has complete operator, logit, recurrent-state, token, and physical results.

### Decision: Do not remove experimental gates

**Choice:** Keep the existing opt-in variables and distinct cache names unchanged.

**Rationale:** Removing them is not needed for a fail-closed production result.
Their presence does not imply admission.

### Decision: Require a new change for future admission

**Choice:** A future effort must define canonical named profiles and collect complete order-balanced physical evidence.

**Rationale:** Ambient variables and static format facts cannot prove correctness or performance.

## Architecture

No product code, port, adapter, service, or configuration is added.

The accepted specification owns the production precision boundary.
The evidence file owns the observed service environment and missing-proof result.

## Validation Strategy

Cairn validates the proposal, design, completed tasks, and specification delta.

The positive scenario keeps BFP8 when complete candidate evidence is absent.
The negative scenario prevents a later BF4 result from becoming admitted without a new validated change.

## Risks and Mitigations

- **Ambient flag drift:** Treat all existing BF4 variables as experimental only.
- **False quality claim:** Require more than PCC and short text.
- **False speed claim:** Require matched physical A/B evidence.
- **Service regression:** Change no runtime or service setting.

## Non-Goals

- Proving that BF4 cannot pass later.
- Changing current cache files or weight conversion.
- Testing activation quantization.
- Changing the production service.
