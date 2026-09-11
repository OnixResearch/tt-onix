let authority = import "../authority.ncl" in

authority
& {
  upstream.tree_blake3 | force = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
}
