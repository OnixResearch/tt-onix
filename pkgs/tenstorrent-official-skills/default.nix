{
  lib,
  stdenvNoCC,
  fetchzip,
  python3,
  makeWrapper,
}:
stdenvNoCC.mkDerivation {
  pname = "tenstorrent-official-skills";
  version = "2026-10-23-b675209";
  src = fetchzip {
    url = "https://github.com/tenstorrent/skills/archive/b6752095f329a04bf6573fb4022b13bc45799167.tar.gz";
    hash = "sha256-GbuRhOKPRRGCFxjF3LfDt1sbGQj3NTyEaFRuG1NGeaA=";
  };
  nativeBuildInputs = [ makeWrapper ];
  dontBuild = true;
  installPhase = ''
    runHook preInstall
    root="$out/share/tenstorrent-official-skills"
    mkdir -p "$root/skills/common" "$root/skills/models" "$root/skills/metal" "$root/skills/ttnn" "$out/bin" "$out/libexec"
    cp -r skills/common/tt-review-core "$root/skills/common/"
    cp -r skills/models/tt-precision-review skills/models/tt-trace-review "$root/skills/models/"
    cp -r skills/metal/tt-l1-memory-review "$root/skills/metal/"
    cp -r skills/ttnn/ttnn-op-kernel-review "$root/skills/ttnn/"
    mkdir -p "$root/docs/common" "$root/docs/models" "$root/docs/metal" "$root/docs/ttnn"
    cp docs/common/tt-review-core.md "$root/docs/common/"
    cp docs/models/tt-precision-review.md docs/models/tt-trace-review.md "$root/docs/models/"
    cp docs/metal/tt-l1-memory-review.md "$root/docs/metal/"
    cp docs/ttnn/ttnn-op-kernel-review.md "$root/docs/ttnn/"
    cp -r LICENSE LICENSES NOTICE SOURCES.md "$root/"
    printf '%s\n' b6752095f329a04bf6573fb4022b13bc45799167 > "$root/REVISION"
    cp ${./reference.py} "$out/libexec/reference.py"
    makeWrapper ${python3}/bin/python "$out/bin/tenstorrent-official-skills" \
      --unset PYTHONPATH --unset PYTHONHOME \
      --add-flags "-I $out/libexec/reference.py $root"
    runHook postInstall
  '';
  meta = {
    description = "Read-only pinned official Tenstorrent review-core, precision, trace, L1 and kernel references";
    homepage = "https://github.com/tenstorrent/skills";
    license = [
      lib.licenses.asl20
      lib.licenses.mit
    ];
    platforms = lib.platforms.all;
    mainProgram = "tenstorrent-official-skills";
  };
}
