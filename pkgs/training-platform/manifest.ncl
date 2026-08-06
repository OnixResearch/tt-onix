# r[impl tenstorrent.training_platform.runtime_bundle]
# Immutable TT-XLA training compatibility manifest. The original component
# set was resolved on 2026-08-04. The packaged driver authority was corrected
# to stable TT-KMD 2.10.0 on 2026-08-06. See README.md for sources.
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  bundle_id = "tt-xla-training-2026-08-04",
  recorded_date = "2026-08-04",
  runtime_image = {
    repository = "ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04",
    digest = "sha256:8dcad76548126bb33c13ce554192cb9f27612fa3766b7c7e7ae6f591f2ab6503",
    created = "2026-08-03T12:07:05Z",
    base_os = "ubuntu-24.04",
  },
  pjrt_plugin = {
    package = "pjrt-plugin-tt",
    version = "1.5.0.dev20260804002041",
    wheel_sha256 = "ae57e7a2e4f7c1c9533135e14db5a33201caa3ff1aa2278de9a88ebc9d4d3fe1",
    python_abi = "cp312",
    index = "https://pypi.eng.aws.tenstorrent.com",
  },
  tt_xla_source = {
    repository = "https://github.com/tenstorrent/tt-xla",
    revision = "10b111f3885a598958b0c4e393a07d672584736a",
    revision_date = "2026-08-04T15:31:46Z",
  },
  tt_blacksmith_source = {
    repository = "https://github.com/tenstorrent/tt-blacksmith",
    revision = "6e44259c7f4342709ae7977cc3715cf5e2e19dac",
    revision_date = "2026-08-04T11:46:42Z",
  },
  pytorch = {
    package = "torch",
    version = "2.11.0+cpu",
    wheel_sha256 = "f82e2ae20c1545bb03997d1cc3143d94e14b800038669ee1aca45808a9acc338",
    python_abi = "cp312",
  },
  pytorch_xla = {
    package = "torch-xla",
    version = "2.9.0+gitb230007",
    wheel_sha256 = "4318d8cce4f9de38d497712b9e80cb74746dc16dc8e2a15e555c417711757481",
    python_abi = "cp312",
  },
  jax = {
    package = "jax",
    version = "0.7.1",
  },
  jaxlib = {
    package = "jaxlib",
    version = "0.7.1",
  },
  transformers = {
    package = "transformers",
    version = "4.57.1",
  },
  driver = {
    name = "TT-KMD",
    version = "2.10.0",
  },
  firmware = {
    bundle_version = "18.8.0.0",
    floor_version = "18.8.0.0",
  },
} | contracts.BundleClosed | contracts.BundleDeepClosed | contracts.Bundle
