# Model target consumer identities

`model-targets/profiles.ncl` is the only source for model target identity data.
Do not edit generated Rust, Python, C++, Nix, or JSON files.

## Update procedure

1. Edit `model-targets/profiles.ncl`.
2. Export the checked JSON data:

   ```sh
   nix shell nixpkgs#nickel -c sh -c \
     'nickel export --format json model-targets/profiles.ncl > model-targets/generated/profiles.json'
   ```

3. Regenerate all consumer modules:

   ```sh
   nix run .#model-target-consumer-generator -- \
     generate model-targets/generated/profiles.json .
   ```

4. Review the profile change and every generated projection.
5. Run the freshness and parity check:

   ```sh
   nix build .#checks.x86_64-linux.model-target-profile-factory --no-link
   ```

The generator calculates each profile identity from its compact canonical JSON.
It uses BLAKE3 and writes lowercase hexadecimal text.

## Review rules

- Keep model equations and control flow in handwritten source.
- Put only identity data in generated modules.
- Reject uppercase, mixed-case, short, long, and non-hexadecimal hashes.
- Review model, artifact, topology, state, precision, tolerance, and adapter changes together.
- Confirm that runtime, command, harness, LoRA, evidence, and report outputs expose the same `profile-blake3` value.

The model-target check regenerates all projections in a clean directory. It compares each output byte for byte. A stale file reports its language path and source profile file.
