## ADDED Requirements

### Requirement: The served continuous-slot profile is selected from an admitted closed set

r[tenstorrent.qwen38_p150x2.continuous_slot_served_profile]

The served configuration MUST select its continuous-slot profile from the closed set
{b4, b8}. A configuration naming any other profile MUST be refused before the service
starts. The served profile MUST have a passing queue-inclusive admission receipt for
the package and execution identity it serves, and the deployment MUST name that
receipt. Selecting a served profile MUST NOT change the historical owner's rollback
command, which MUST continue to name the policy file its own package carries.

#### Scenario: Profile outside the closed set
- GIVEN a configuration that names a continuous-slot profile outside {b4, b8}
- WHEN the service configuration is evaluated
- THEN it is refused and no service command is produced.

#### Scenario: Admitted profile is served
- GIVEN a configuration that names the B=8 profile and a package with a passing B=8
  admission receipt
- WHEN the service starts
- THEN it selects the packaged B=8 policy, serves eight live slots, and reports the
  profile identity the receipt admitted.

#### Scenario: Historical owner rollback stays exact
- GIVEN the managed drop-in selects the B=8 policy
- WHEN the historical owner command is reconstructed
- THEN it names the B4 policy file of the historical package, byte for byte.
