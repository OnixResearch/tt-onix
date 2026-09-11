# Pure selection at the composition root. No declared check can be omitted.
{ lib }:
{
  catalog,
  available,
}:
let
  schemaVersion = 1;
  required = catalog.required;
  names = builtins.attrNames available;
  missing = lib.subtractLists names required;
  unregistered = lib.subtractLists required names;
in
if catalog.schema_version != schemaVersion then
  throw "unsupported required-check catalog schema"
else if required == [ ] || builtins.length (lib.unique required) != builtins.length required then
  throw "required-check catalog must be nonempty and unique"
else if missing != [ ] then
  throw "required package checks are missing: ${lib.concatStringsSep ", " missing}"
else if unregistered != [ ] then
  throw "package checks are not registered: ${lib.concatStringsSep ", " unregistered}"
else
  map (
    name:
    let
      check = available.${name};
    in
    if lib.isDerivation check then
      check
    else
      throw "required package check is not a derivation: ${name}"
  ) required
