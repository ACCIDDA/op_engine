# Citation and archiving

[CITATION.cff](https://github.com/ACCIDDA/op_engine/blob/main/CITATION.cff) records
the shared `op_engine` / `flepimop2-op_engine` software release and preserves the
authors and order already attributed in both package metadata files. Maintainers
review attribution when contributors/roles change rather than deriving it from
commit counts. Core uses MIT; the provider's source grants GPL version 3 or any
later version (`GPL-3.0-or-later`). Record both distribution versions when the
provider is used, plus flepimop2/op_system/backend versions relevant to the run.

For published work, cite the exact version and its verified identifier. The
initial CFF describes Joshua Macdonald, Carl Pearson, and Timothy Willard,
*op_engine: Operator-partitioned numerical solvers and flepimop2 provider*,
version 0.6.0, released 2026-10-09, with the
[v0.6.0 release URL](https://github.com/ACCIDDA/op_engine/releases/tag/v0.6.0).
Development results also need the commit and any local changes. Unreleased
features on main are not retroactively part of 0.6.0; use the exact tag's source
and changelog and the documentation version selector.

Run `just citation` to validate CFF 1.2.0 with pinned `cffconvert==2.0.0` in an
isolated Python 3.12 tool environment and check authors/version/licenses/date
against package metadata and the dated changelog. `just ci`, release validation
and CI include citation checks. Update the CFF version, release date and artifact
URL alongside a shared version bump. Add `preferred-citation` only when a paper's
authors, title, venue, date and persistent identifier have been verified; retain
the versioned software citation for reproducibility.

## Identifier audit

On 2026-10-14, the repository, publication workflows and v0.6.0 release notes
contained no software archive DOI, `.zenodo.json`, archival integration
instructions or preferred paper citation. Public DataCite and Zenodo API queries
for the exact repository URL returned zero records, and public searches found no
verified archive/paper identifier. This does not establish that no maintainer
record exists: connected archival accounts and unindexed records must still be
checked. No persistent identifier is invented in the citation.

The current workflow publishes GitHub release and package-index artifacts;
it does not itself perform an archival deposit. The shared GitHub release URL
is the verified identifier until a resolving archive record is confirmed.

## ACCIDDA software-paper archive plan

[#215](https://github.com/ACCIDDA/op_engine/issues/215) tracks the shared release
archive and verified DOI, following the ACCIDDA Zenodo pattern selected for
op_system. Finish the JOSS/release blockers before freezing that archival release.
The release maintainer should:

1. Check the ACCIDDA Zenodo account for an existing op_engine record/version
   series and reuse it if present. Avoid duplicate records for the same software.
2. Review attribution and grants; align both package versions, the provider's
   exact core dependency, dated changelog, citation and release tag. Run
   `just ci`, `just docs`, `just qualify` and `just release-validate`.
3. Publish through the existing shared release policy and archive the exact tag
   for both distributions, released artifacts, and any benchmark fixtures/JSON
   supporting paper claims. Preserve the commit, checksums, environment/hardware,
   numerical protocol and versioned documentation links.
4. Verify that the deposit resolves and matches the released artifacts. Record
   the version DOI in citation metadata, release notes and paper. Distinguish a
   concept DOI for the version series from the exact release DOI.
5. Add a preferred paper citation when its bibliographic record exists, and
   revalidate identifiers and CFF syntax before announcing the release.

See Zenodo's [GitHub integration](https://help.zenodo.org/docs/github/enable-repository/)
and [release archiving](https://help.zenodo.org/docs/github/archive-software/github-upload/)
for account setup or a manual deposit. Readiness PRs do not create a deposit,
activate integration, reserve a DOI or publish a release.
