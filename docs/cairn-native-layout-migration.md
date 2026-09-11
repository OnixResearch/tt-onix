# Cairn native-layout migration

Date: 2026-08-27

The repository migrated its lifecycle content from `cairn/` to `.cairn/` with the canonical Cairn checkout at commit `15f00875562025e7ea7e0d1f4af24d1a2e2ac06f`.

## Migration identity

- Source layout: `cairn/`
- Native layout: `.cairn/`
- Source files copied: `504`
- Plan BLAKE3: `6eb0f8e8fec3e014095b6ad1893fb9c82c5492fe303f78d1fac3334148670582`
- Source manifest BLAKE3: `445a0a6e32eed87cadde734c18542762b5abba03be34700810de335c7d8db380`
- Destination manifest BLAKE3: `a45c3c316d676133d079a75b3748929bf965432784d43ccfb7d6f3fd8db4bb34`
- Execution receipt BLAKE3: `c408af041fd57a2fe5fce640420a92fa2be3e772ed6d36d34e7fd6369115eec3`
- Policy BLAKE3: `8280151b7a53822eed460149ecb600bf11418d7463c935e0742009b334f4e7dd`

The exact plan reported `blocked = false` and no conflicts. Execution reported `mutated = true`.

## Validation

The latest packaged Cairn CLI completed strict validation after migration. It selected the native layout, validated 15 specifications, found no active changes, and reported no issues.

The migration copied content without deleting `cairn/`. This retention follows the Cairn migration contract. New lifecycle commands select `.cairn/`; the legacy tree remains rollback evidence until a separate reviewed cleanup removes it.

## Non-claims

This receipt proves the bounded lifecycle copy and validation result. It does not prove source-code correctness, physical hardware behavior, or release eligibility.
