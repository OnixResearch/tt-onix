# Identity mismatch evidence — 2026-10-22

## Scope and provenance

Read-only inspection of three exact store packages and the 192 committed compressed responses. Package labels below are the established roles in this workstream: `wlsl6…` is the served package, `frf73q…` is the package named by `protocol-v3-validation.md`, and `a7nlx…` is the HEAD build observed on 2026-10-22. No store-wide search was performed. This file is the only investigation output written to the repository.

## Package to embedded identity map

The exact matching wrapper lines (the lines selected by a grep for `--package-identity`) are:

```text
/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0/bin/qwen38-p150x2-serve
10:exec "/nix/store/qwm7vyvj1jbr0q65jdmshfqk6k9l4isc-python3-3.13.13-env/bin/python"  -m qwen36_service.server --decode-profile /nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0/share/qwen36/decode-profile.json --decode-mode candidate --trace-reuse-profile /nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0/share/qwen36/trace-reuse-profile.json --runtime-identity /nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0 --package-identity /nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0 "$@"

/nix/store/frf73q2612gczcj1qr43zz48i3frj2q4-qwen38-0.1.0/bin/qwen38-p150x2-serve
10:exec "/nix/store/qwm7vyvj1jbr0q65jdmshfqk6k9l4isc-python3-3.13.13-env/bin/python"  -m qwen36_service.server --decode-profile /nix/store/frf73q2612gczcj1qr43zz48i3frj2q4-qwen38-0.1.0/share/qwen36/decode-profile.json --decode-mode candidate --trace-reuse-profile /nix/store/frf73q2612gczcj1qr43zz48i3frj2q4-qwen38-0.1.0/share/qwen36/trace-reuse-profile.json --runtime-identity /nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0 --package-identity /nix/store/4115hgb2ri3168c9d0g4r4pjj8m8m06n-qwen36-execution-identity-0.1.0 "$@"

/nix/store/a7nlx35sz0pvklx96vbcks55qbaqam18-qwen38-0.1.0/bin/qwen38-p150x2-serve
10:exec "/nix/store/qwm7vyvj1jbr0q65jdmshfqk6k9l4isc-python3-3.13.13-env/bin/python"  -m qwen36_service.server --decode-profile /nix/store/a7nlx35sz0pvklx96vbcks55qbaqam18-qwen38-0.1.0/share/qwen36/decode-profile.json --decode-mode candidate --trace-reuse-profile /nix/store/a7nlx35sz0pvklx96vbcks55qbaqam18-qwen38-0.1.0/share/qwen36/trace-reuse-profile.json --runtime-identity /nix/store/7hplkynh047dsk98cwqpfsaaq6lwyy0m-tt-metal-0.77.0 --package-identity /nix/store/rymc1wl22dwr1db16948sxlqk1p6wrm3-qwen36-execution-identity-0.1.0 "$@"
```

## Identity input counts and exact differences

Inspection command, exit status 0 (Python selects the matching lines, counts input records, and produces zero-context unified diffs):

```sh
GIT_PAGER=cat timeout 30 python3 -c 'import pathlib,re,difflib; ps=["wlsl6pnxllv9qhr6yqimnwv3si0qyz9k","frf73q2612gczcj1qr43zz48i3frj2q4","a7nlx35sz0pvklx96vbcks55qbaqam18"]; ids=[]; texts=[]; [(print("\n"+str(p)),print("\n".join(str(n)+":"+s for n,s in enumerate(p.read_text().splitlines(),1) if "--package-identity" in s)),ids.append(re.search(r"--package-identity (\S+)",p.read_text()).group(1))) for p in [pathlib.Path("/nix/store/"+x+"-qwen38-0.1.0/bin/qwen38-p150x2-serve") for x in ps]]; [(texts.append((pathlib.Path(i)/"share/qwen36/execution-inputs").read_text().splitlines()),print(i,"lines=",len(texts[-1]),"inputs=",sum(s.startswith("input=") for s in texts[-1]))) for i in ids]; [(print("\n".join(difflib.unified_diff(texts[a],texts[2],fromfile=ids[a],tofile=ids[2],n=0,lineterm="")))) for a in [0,1]]'
```

Counts include one schema line in addition to input records:

```text
/nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0 lines= 79 inputs= 78
/nix/store/4115hgb2ri3168c9d0g4r4pjj8m8m06n-qwen36-execution-identity-0.1.0 lines= 84 inputs= 83
/nix/store/rymc1wl22dwr1db16948sxlqk1p6wrm3-qwen36-execution-identity-0.1.0 lines= 84 inputs= 83
```

Each compared file is `share/qwen36/execution-inputs` below the indicated identity:

```diff
--- /nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0
+++ /nix/store/rymc1wl22dwr1db16948sxlqk1p6wrm3-qwen36-execution-identity-0.1.0
@@ -2 +2 @@
-input=/nix/store/008f6nzhxr8akqqc7kv1s657521a2gi2-qwen36-service
+input=/nix/store/pmnbajf9406yl82kgniyjni2yz1rrj32-qwen36-service
@@ -15,0 +16,5 @@
+input=/nix/store/dm2g2yd39vlcqr6vsy51h637b2rkgn38-qwen36-batch-admission-b8-contracts-v3.ncl
+input=/nix/store/vkk5qwsddalsw4nn6w3s4l4gny3h66m7-qwen36-batch-admission-b8-policy-v3.ncl
+input=/nix/store/8c8lyn0spijkngmd6jmh01ppp2n705k3-qwen36-batch-admission-b8-policy-v3-validated.ncl
+input=/nix/store/2qj4vbnkx6x7xb5xw2pmmgq6v8dfk5mz-qwen36-batch-admission-b8-binding-v3.ncl
+input=/nix/store/sxk6xvagrya4n0qg66j16cqp17f40i0q-qwen36-batch-admission-b8-profile-v3.ncl
--- /nix/store/4115hgb2ri3168c9d0g4r4pjj8m8m06n-qwen36-execution-identity-0.1.0
+++ /nix/store/rymc1wl22dwr1db16948sxlqk1p6wrm3-qwen36-execution-identity-0.1.0
@@ -19 +19 @@
-input=/nix/store/xraqbbhvx5lvj3ac36bay709v55qvpjl-qwen36-batch-admission-b8-binding-v3.ncl
+input=/nix/store/2qj4vbnkx6x7xb5xw2pmmgq6v8dfk5mz-qwen36-batch-admission-b8-binding-v3.ncl
```

## Saved response roots and contradicting package export

Inspection command, exit status 0:

```sh
GIT_PAGER=cat timeout 30 python3 -c 'import pathlib,gzip,json; base=pathlib.Path("pkgs/qwen36/evidence/2026-10-22"); ps=sorted(base.glob("qwen38-batch-admission-v3-*.json.gz")); es=[json.loads(gzip.decompress(p.read_bytes()))["x_onix_decode_evidence"] for p in ps]; print("files:",len(ps)); [(print(k+":",json.dumps(sorted(set(e[k] for e in es))))) for k in ["package_identity","profile_id","serving_route","batch_width","admission_eligible"]]; rs=[e["aggregate_decode_tokens_per_second"] for e in es]; print("aggregate_decode_tokens_per_second range:",min(rs),max(rs)); p=pathlib.Path("/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0/share/qwen36/batch-admission-b8-profile.json"); r=json.loads(p.read_text()); print(p); print(json.dumps({k:r[k] for k in ["profile_id","physically_admitted"]},indent=2)); [(print("\n"+str(p)),print(p.read_text())) for p in [pathlib.Path("/nix/store/"+x+"-qwen36-batch-admission-b8-binding-v3.ncl") for x in ["xraqbbhvx5lvj3ac36bay709v55qvpjl","2qj4vbnkx6x7xb5xw2pmmgq6v8dfk5mz"]]]'
```

Raw root and export output:

```text
files: 192
package_identity: ["/nix/store/l4arf461kxbbns0nnlg2vd98anx943pc-qwen36-execution-identity-0.1.0"]
profile_id: ["qwen38-p150x2-batched-throughput-b8-v3"]
serving_route: ["step"]
batch_width: [1, 8]
admission_eligible: [true]
aggregate_decode_tokens_per_second range: 17.142854522193073 175.1648036055169
/nix/store/wlsl6pnxllv9qhr6yqimnwv3si0qyz9k-qwen38-0.1.0/share/qwen36/batch-admission-b8-profile.json
{
  "profile_id": "qwen38-p150x2-batched-throughput-b8-v2",
  "physically_admitted": true
}
```

The roots report v3 while this package's B=8 export reports v2. All roots report the identity embedded by `wlsl6…`, not the identities embedded by `frf73q…` or `a7nlx…`. This observation alone does not identify which executable or profile file a root process actually loaded.

## Bootstrap arithmetic

Source observed before the bootstrap fix, `pkgs/qwen36/default.nix`:

```text
107:  executionIdentityLocalInputs = [
125:    (stableIdentityPath "qwen36-batch-admission-b8-binding-v3.ncl" ./batch-admission-b8-binding-v3.ncl)
163:  executionIdentityInputs =
164:    executionIdentityLocalInputs
```

The two binding inputs read by the inspection command above contain:

```text
/nix/store/xraqbbhvx5lvj3ac36bay709v55qvpjl-qwen36-batch-admission-b8-binding-v3.ncl
# Mutable B=8 receipt binding for the interleaved protocol. This record cannot
# change execution behavior. It names the receipt the interleaved campaign
# produced, or stays unmeasured until that campaign runs.
{
  physically_admitted = false,
  receipt_blake3 = "not-measured",
}

/nix/store/2qj4vbnkx6x7xb5xw2pmmgq6v8dfk5mz-qwen36-batch-admission-b8-binding-v3.ncl
# Mutable B=8 receipt binding for the interleaved protocol. This record cannot
# change execution behavior. It names the receipt the interleaved campaign
# produced, or stays unmeasured until that campaign runs.
{
  physically_admitted = true,
  receipt_blake3 = "af2365ffb09002bb5674e175374efda1a81ea1a665c1f199b16a069344ac785e",
}
```

For the observed pair, all other execution-input records are equal:

```text
identity(not-measured) = /nix/store/4115hgb2ri3168c9d0g4r4pjj8m8m06n-qwen36-execution-identity-0.1.0
identity(admitted)     = /nix/store/rymc1wl22dwr1db16948sxlqk1p6wrm3-qwen36-execution-identity-0.1.0
4115hgb2ri3168c9d0g4r4pjj8m8m06n != rymc1wl22dwr1db16948sxlqk1p6wrm3
```

Thus changing only the receipt binding changes the measured execution identity. A receipt produced against the not-measured identity cannot name the admitted identity obtained by putting that receipt into the binding. The receipt identity `l4arf461…` is a separate mismatch: it has neither of these v3 binding inputs, lacks all five v3 inputs listed above, and has a different service input.

## Non-claims and hypothesis to test

- No silicon run, simulator run, deployment, or service restart was performed for this evidence.
- This evidence makes no claim that the campaign was invalid. It records mismatches between stored wrappers, identity input records, package exports, and saved response fields.
- No claim is made about which route is better, throughput improvements, or a new admission.
- Whether the v3 policy reached the roots out-of-band is a hypothesis the bootstrap change will test, not a conclusion established here.
