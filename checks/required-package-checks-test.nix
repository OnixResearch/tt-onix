{ lib, catalog }:
let
  select = import ./required-package-checks.nix { inherit lib; };
  fixture = name: {
    type = "derivation";
    inherit name;
    outPath = "/fixture/${name}";
  };
  available = lib.genAttrs catalog.required fixture;
  attempt = value: builtins.tryEval (builtins.deepSeq value true);
  selected = select { inherit catalog available; };
  missingRejected = builtins.all (
    name:
    !(attempt (select {
      inherit catalog;
      available = builtins.removeAttrs available [ name ];
    })).success
  ) catalog.required;
  invalidCatalogs = [
    (catalog // { schema_version = 0; })
    (catalog // { required = [ ]; })
    (catalog // { required = catalog.required ++ catalog.required; })
  ];
  invalidCatalogsRejected = builtins.all (
    invalid:
    !(attempt (select {
      catalog = invalid;
      inherit available;
    })).success
  ) invalidCatalogs;
  extraRejected =
    !(attempt (select {
      inherit catalog;
      available = available // {
        unregistered = fixture "unregistered";
      };
    })).success;
  nonDerivationRejected =
    !(attempt (select {
      inherit catalog;
      available = available // {
        qwen36 = true;
      };
    })).success;
  qwenNames = [
    "qwen36"
    "qwen36-authority-negative"
    "qwen36-nixos-service"
  ];
in
assert map (check: check.name) selected == catalog.required;
assert builtins.all (name: builtins.elem name catalog.required) qwenNames;
assert missingRejected;
assert invalidCatalogsRejected;
assert extraRejected;
assert nonDerivationRejected;
true
