# r[impl tenstorrent.rwkv7_dual_p150a.configuration]
# Dependency wheels pinned by the metadata of the admitted runtime wheels.
# The pjrt-plugin-tt 1.5.0.dev20260804002041 METADATA declares tt_lang as a
# URL dependency; that exact URL and its observed sha256 are recorded here.
# The three primary runtime wheels carry their sha256 identities in the
# training platform manifest.
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  tt_lang = {
    package = "tt_lang",
    version = "1.1.5.dev20260704+light",
    url =
      "https://pypi.eng.aws.tenstorrent.com/2026-07/tt-lang/tt_lang-1.1.5.dev20260704%2Blight-cp312-cp312-manylinux_2_34_x86_64.whl",
    wheel_sha256 = "2bfee48e61a7405bb4cff110e4b7b8dfe002228fc8eb955d39c62da1316f7bae",
    pinned_by = "pjrt-plugin-tt==1.5.0.dev20260804002041 METADATA Requires-Dist",
  },
} | contracts.WheelsClosed
  | contracts.Wheels
