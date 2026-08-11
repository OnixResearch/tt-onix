{ tt-metal }:
tt-metal.overrideAttrs (previous: {
  pname = "${previous.pname}-fetch-queue-diagnostics";
  patches = (previous.patches or [ ]) ++ [ ./fetch-queue-diagnostics.patch ];
})
