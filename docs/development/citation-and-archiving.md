# Citation and archiving

[CITATION.cff](https://github.com/ACCIDDA/op_engine/blob/main/CITATION.cff) records
the `op_engine` core release and preserves the authors and order in its package
metadata. Maintainers review attribution when contributors or roles change
rather than deriving it from commit counts. Core uses MIT. The separately
released provider source grants GPL version 3 or any later version
(`GPL-3.0-or-later`). When the provider is used, record its distribution version
separately, plus flepimop2, op_system and backend versions relevant to the run.

For published work, cite the exact version and its verified identifier. The
initial CFF describes Joshua Macdonald, Carl Pearson, and Timothy Willard,
*op_engine: Operator-partitioned numerical solvers*,
version 0.8.0, released 2026-10-19, with the
[v0.8.0 release URL](https://github.com/ACCIDDA/op_engine/releases/tag/v0.8.0).
Development results also need the commit and any local changes. Unreleased
features on main are not retroactively part of 0.8.0; use the exact tag's source
and changelog and the documentation version selector.

Run `just citation` to validate CFF 1.2.0 with pinned `cffconvert==2.0.0` in an
isolated Python 3.12 tool environment and check authors/version/licenses/date
against core package metadata and the dated changelog. `just ci`, release
validation and CI include citation checks. Update the CFF version, release date
and artifact URL alongside a core version bump. Add `preferred-citation` only when a paper's
authors, title, venue, date and persistent identifier have been verified; retain
the versioned software citation for reproducibility.

## Identifier audit

On 2026-10-23, the repository, release metadata and public archive searches
contained no verified software archive DOI or preferred paper citation. A
release webhook does not by itself establish its target service, a completed
deposit, a resolving DOI or ACCIDDA community acceptance. Inspect integration
settings only through a credential-safe interface; never copy a credentialized
receiver URL into logs, issues or review notes. No persistent identifier is
inferred or invented.

The current workflow publishes GitHub release and package-index artifacts;
it does not itself perform an archival deposit. The selected GitHub release URL
is the verified identifier until a resolving archive record is confirmed.

## ACCIDDA software-paper archive plan

Use the existing ACCIDDA Zenodo community for a verified release record. The
archival deposit, DOI and community acceptance still need verification.
The release maintainer should:

1. Check the ACCIDDA Zenodo account for an existing op_engine record/version
   series and reuse it if present. Avoid duplicate records for the same software.
2. Review attribution and grants; align the selected distribution version,
   dated changelog, citation and release tag. For a provider release, verify its
   exact core dependency. Run `just ci`, `just docs`, `just qualify` and the
   applicable release validation.
3. Before production publication, reconnect or rotate the archive integration
   through its owning service, then verify it without exposing its credential.
   Do not create a duplicate integration or delete and recreate an existing
   release to trigger retroactive ingestion. Publish through the configured
   release policy and archive the exact tag, released artifacts, and any
   benchmark fixtures/JSON supporting paper claims. Preserve the commit,
   checksums, environment/hardware, numerical protocol and versioned
   documentation links.
4. Verify that the deposit resolves and matches the released artifacts. Record
   its version DOI in the release notes and in citation metadata on `main` for
   the next release; do not rewrite the immutable tagged CFF. A DOI reserved and
   verified before tagging may instead be included in that tag's CFF.
   Distinguish a concept DOI for the version series from the exact release DOI.
5. Submit the published record to the ACCIDDA community and wait for curator
   acceptance.
6. Add a preferred paper citation when its bibliographic record exists, and
   revalidate identifiers and CFF syntax before announcing the release.

See Zenodo's [GitHub integration](https://help.zenodo.org/docs/github/enable-repository/)
and [release archiving](https://help.zenodo.org/docs/github/archive-software/github-upload/)
for account setup or a manual deposit. Readiness PRs do not create a deposit,
activate integration, reserve a DOI or publish a release.
