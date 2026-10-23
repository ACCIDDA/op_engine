# Methods writing guide

Version 4, 23 October 2026, [issue 422](https://github.com/yoavram-lab/pp-eigentest/issues/422). **Independently reviewed for application and propagation** in the [version-4 review](https://github.com/yoavram-lab/pp-eigentest/issues/422#issuecomment-6367360639). Installation and manuscript application retain their separate review gates. The version-2/3 approvals remain historical. Read with [README.md](README.md), [prose.md](prose.md), [methods.md](methods.md), [proofs.md](proofs.md), [figures.md](figures.md) and [sources.json](sources.json). Earlier corpus observations and their limits are retained; current requirements are identified separately.

## Evidence and authority

This guide addresses the author's explicit second-pass requirements in [issue 376](https://github.com/yoavram-lab/pp-eigentest/issues/376): sufficient attribution within Methods, definitions before use, explanatory names, reproducible operations and a readable standalone SI. These are requirements, even where the sampled papers are abbreviated. They strengthen the author-approved prose guide's direction to describe the work in order. They are not inferred frequency targets.

The observations below come from the EHS author TeX, `MainText.tex`, Methods lines 187–276, checked against the published main PDF, pp. 5–7, and the published SI, S1.1–S1.6. Exact paths, hashes, editions and inspected ranges are in `sources.json`. Both papers in the permitted corpus are coauthored; sentence-level authorship is unknown. No excluded biological article or cached text was examined for this pass. Reading the inherited repository guides does not constitute reading the other papers they mention.

## Observed EHS Methods patterns

The following passages are observations about this coauthored paper, rather than scientific validation of its procedures or universal claims about the author's work. Quotations use the author TeX unless identified as published SI.

| Passage | Exact wording or source construction | Observed function |
|---|---|---|
| Methods lines 192–199; published p. 5 | “we first perform” VBPCA, followed immediately by `ilin2010practical` and a distinct Matlab implementation citation, `VBPCAcode`; the comparator names Scikit-learn and `scikit-learn` | Places method and implementation attribution beside the operation, then explains the comparison and reason for the choice. |
| Lines 205–219; p. 5, equation (1) | “Given an $n \times p$ dimensional data matrix $X$ composed of $n$ samples and $p$ features, and $k$ archetypes”; the following clauses give the dimensions and constraints of $\alpha$, $\beta$ and $S$ | Introduces the objects needed to read the equation, then identifies which reconstructed matrix is analyzed. Some objects are defined immediately after the display; the author's present definition-before-use requirement is stricter. |
| Lines 216–217; p. 5 | “minimize the approximation error” with `morup2012archetypal`, then “previously implemented in the Python package” `pypcha` with `py_pcha` | Connects an optimization target to a method citation and separately identifies the implementation. |
| Lines 232–236; p. 6 | “the silhouette score” with `rousseeuw1987silhouettes`, “the variance ratio criterion” with `calinski1974dendrite`, and “the Davies-Bouldin index” with `davies1979cluster`; the passage explains each and says “higher is better” or “lower is better” | Names and cites each metric where it is used, explains the value's meaning, and states the candidate range, agreement rule and fallback. |
| Lines 241–255; p. 6 | “The null model, $M_0$” and “The alternative model, $M_1$”; then “We first sample $N$ datasets”, “Then”, “Next”, and the stated test alternative and rejection threshold | Makes the analysis target and competing models explicit, then describes repeated operations and how their outputs support a decision. The excerpt does not itself supply a numerical value for $N$; the new guide requires that value or an exact protocol pointer. |
| Lines 264–272; pp. 6–7 | “the Euclidean distance in the archetype space”; Neighbor-Net with `bryant2004neighbor` and `splitstree` with `huson2008splitstree`; $\delta$-scores/Q-residuals with their references and the four-point condition | Names the input geometry, separates algorithm from software, and interprets metric values and a boundary case. |
| Published SI S1.1, p. 1 | “one feature (cultural trait) is chosen as the dependent variable” and “the other features are treated as independent variables” | Supplies the actual iterative operation and identifies its input and prediction target before reporting comparisons. |

The corpus also illustrates why an editor must check implementation details independently. The EHS main Methods say that missing values were initialized by the column median; published SI S1.1 says column mean. Preserve that discrepancy in the source record and do not copy either initialization into a new manuscript without its own implementation evidence. The observations support explanatory organization and local attribution; they do not certify complete reproducibility or every statistical inference in EHS.

## Required drafting sequence

Apply the sequence and checklists only to procedures actually used in the
manuscript. Capacity selection, reference generation, zero-capacity candidates
and similar pp-eigentest examples are relevant only where that project's
approved analysis uses them. This guide requires their description when
applicable; it requires no new capacity search, reference study or zero arm.

1. State the scientific measurement, unit of analysis and target before the fitting or testing machinery. Identify what a row and an original variable represent, how encoded features relate to the variables, and which population or prediction setting is being studied.
2. Define each symbol, acronym, method, strategy and metric before its first substantive use in each independently read document. Give matrix/vector dimensions, index ranges, parameter domains, units, normalizations and constraints. Define the objects on the right of an equation before asking the reader to interpret it; introduce a large collection in a nearby table if that reads more clearly.
3. Describe operations in the order actually executed: input/exclusions, encoding and scaling, training/holdout construction, fit, capacity selection and refit, reference generation, decision rule, validation, uncertainty and recorded failure outcomes. Use ordinary verbs naming what was fitted, drawn, compared or scored.
4. Give the target of each operation. A loss predicts a declared unit; a rank decision compares a declared statistic with a declared reference; a calibration study measures a declared error event. Keep fitted/model quantities, decision outputs, generating targets and diagnostics distinct. In pp-eigentest, these include fitted capacity $q$, noise-selected count $w$, generating rank and effective rank; other manuscripts retain their own approved notation and targets.
5. Place the primary method citation at the sentence introducing its use in Methods. Supply separate implementation identity and version, then describe deviations, adaptations and study-specific choices. A citation in the Introduction or a bibliography entry alone does not satisfy this requirement.
6. Supply enough detail to reproduce the operation or a precise, accessible SI/protocol location containing it. The main Methods must still explain the operation and target. A script name, registration ID, software version or opaque arm label cannot carry the explanation by itself.
7. Explain what the output means, how it is assessed, and the assumptions needed for the interpretation. Keep completed analyses, optional reporting measures, failed checks, pending validations and future extensions explicitly distinct.

## Names that explain the procedure

Use a short descriptive name in running text, figures and table captions. Define its technical name, abbreviated name and artifact identifier together when traceability requires them. Once the mapping is clear, retain exact machine identifiers in a protocol table, source note or parenthesis; do not rename registrations or historical artifacts.

For pp-eigentest, an initial description of the adopted `S1-both` arm should explain **full nominal indicators, ordinal level indices and two indicators for each binary variable**. Explain `both` rather than making the reader infer it. A later “ordinal-index encoding” may be used only where its full meaning and binary treatment remain clear. `S4` needs its full nested definition, including the preceding weighting/encoding strategy and training-only continuous diagnostic transforms; “S4” or “diagnostic transforms” alone is insufficient. Verify every mapping against the actual registered plan and implementation before drafting it.

Give all compared arms in a compact SI table: descriptive name, exact artifact identifier, nominal/ordinal/binary/continuous treatment, weights or scaling, learned versus externally declared quantities, supported data types and validation status. Do not introduce a future arm merely to preserve a code's numbering. If it matters to a limit, explain what it would change and which validation is pending. Define historical names where they occur in the independently read SI.

## Attribution at the point of use

| Kind of borrowed object | Required local attribution and explanation |
|---|---|
| Statistical or mathematical model | Primary model source, model assumptions and the version/adaptation used here. Cite PPCA and VBPCA where the likelihood/fit is introduced, even if cited earlier in the Introduction. |
| Fitting, pruning or transformation algorithm | Primary algorithm source where its operation is described; identify the implemented variant, initialization, stopping and any modification. ARD's name alone does not explain its role. |
| Capacity/validation rule | Source for an adopted established rule where introduced; state the target, holdout unit, folds, aggregation, candidate range, tie/stopping rule and preprocessing learned within folds. Label a study-specific rule as such and point to its protocol. |
| Reference or testing method | Relevant primary source and the procedure's actual statistic, draws, tail, ordering, ties, quantile convention, stopping and finite-budget resolution. Explain which properties are inherited, derived here, empirically examined or still unproved. |
| Score, interval or diagnostic | Cite the adopted definition where introduced; give its formula or unambiguous SI definition, prediction input, units/coordinates, aggregation and direction of improvement. Separate scores actually collected from reporting options. |
| Simulation, calibration or registered protocol | Cite relevant methodological provenance at use and give an exact version/date or immutable record for the study-specific design, regimes, estimand, uncertainty, selection rule and deviations. A general simulation-study citation cannot replace the particular protocol. |
| Dataset, instrument or prior empirical analysis | Primary data/source citation where the data enter the analysis; identify edition/accession, units, sample/feature counts, exclusions and transformations. Cite a prior application separately from the data release where these differ. |
| Software | Package/repository or software paper, released version or commit and relevant function/options. This identifies implementation; it does not replace the method's primary citation. Describe different historical versions alongside the results they generated. |

Before inserting a citation, inspect the relevant primary text or official software/data record and verify that it supports the specific operation, theorem or definition attributed to it. Check author/title/year/DOI or stable URL and edition against that record. Keep a short citation-verification record with location and support; an unresolved pointer remains a documented gap until checked. Do not invent a reference, fill its metadata from memory, attach a familiar citation to a different variant, or present a package name as a source for a theorem. Existing bibliography entries are candidates for verification, not proof that their contents were inspected.

## Reproducibility and independent SI readability

For each analysis, ensure that the text and its exact SI/protocol pointers jointly identify the applicable items below. Omit procedures outside the approved analysis; do not introduce them to satisfy this checklist:

- Inputs, units, dimensions, schema/levels, masks, exclusions and the prediction or inferential target.
- Declared versus learned preprocessing, its training scope, inverse decoding, out-of-domain values and unavailable/failed-score handling.
- Fitting model and algorithm, priors/hyperparameters, initializations, candidate capacities including zero, stopping criteria, finite budgets, refits and numerical failures.
- Validation folds and blocking unit, loss formula and aggregation, selection rule, ties, standard errors and any unequal probe counts.
- Reference draws and their constraints, randomization/seeds, statistic normalization, Monte Carlo count, tail/tie convention and the exact retained-direction rule.
- Simulation/data regimes, generating versus fitted quantities, estimands and denominators, paired comparisons, uncertainty and multiplicity; applicable and unavailable cases must remain visible.
- Primary method/data citations, separately identified software versions and immutable implementation/protocol records. Main text should carry only the implementation detail needed to understand the procedure; exact run identity can live in the SI or availability statement.

Read the SI from its first page without assuming access to the main text. Reintroduce its notation and abbreviations, define each arm and score before its first SI use, give the applicable model assumptions and citations, and make table captions interpretable with their nearby definitions. Conversely, do not leave the main Methods intelligible only after reading a generated results table or repository file. Check all included TeX files and captions, since a self-contained parent file can still import unexplained labels.

## Methods review gate for the next editing agent

Before manuscript edits, obtain an independent review of this guide and the revised prose/proof profiles. Record corrections and their resolution. The review must challenge source attribution, exact passage mappings, exclusions and the requirements' operational adequacy; it is distinct from the later review of manuscript application.

After that gate, build a first-use inventory and a citation-at-use inventory for main Methods and the SI separately. For each term or method, record its first location, definition, dimensions/units, source and implementation/protocol pointer; mark unresolved entries. Walk one representative input through each described pipeline to ensure the operations and outputs can be reconstructed from the text. Then inspect all main/SI results, captions, tables, included files and availability statements for consistent names and scope. A text search is a screen; a human reading decides whether the definition and citation actually support the use.

Do not count citations, sentence lengths, hedges or first-person pronouns toward a quota. A Methods pass succeeds when the reader can identify and reproduce each procedure, understand its target and locate its provenance while all claims, denominators, caveats and pending work retain their scientific meaning.

## Version-4 target and presentation checks

These are transferable author/editorial requirements, not new corpus
observations or a change to any project's scientific protocol. State the
scientific target before selecting a convenient summary. A selected count,
reconstruction loss, retained fitted energy, recovery of a known signal,
pairwise-distance distribution, identity of close pairs, relative comparison,
network diagnostic and historical explanation are different targets. Use only
the distinctions relevant to the manuscript and define them locally. Do not
promote success on one endpoint to another without the corresponding evidence.

For distance-based work, define the representation, scaling, metric or
dissimilarity, any covariance weighting, normalization and the exact downstream
statistic where they enter the analysis. Explain the scientific emphasis each
choice introduces. Matching marginal distributions of distances need not
preserve which named pairs are close. A diagnostic that becomes more tree-like
need not identify a true transmission history. If these distinctions support a
claim, provide a cited definition, complete argument or relevant empirical
comparison rather than treating the wording as evidence.

Define the analysis unit and dependence structure for uncertainty. Pairs or
quartets from the same dataset share observations and must not silently become
independent replicates. A known clean geometric target and a known generative or
causal history require different validation. For a prospective comparison,
declare target, metric, normalization, ties, undefined outcomes and the
independent replication unit before evaluation, through the project's existing
scientific review and admission process. This guidance admits no new endpoint,
simulation, metric or run on its own.

Separate recorded numerical precision from printed precision. Keep approved
full-precision estimates, interval endpoints and source identities in their
artifacts; use a deliberate stated display precision in a table or figure.
A renderer must use the declared source rows and intervals rather than silently
re-estimate them. Preserve denominators, pairing, weights, study differences,
interval meaning and adverse results. See [figures.md](figures.md) for figure
provenance, accessible presentation and actual compiled-layout review.
