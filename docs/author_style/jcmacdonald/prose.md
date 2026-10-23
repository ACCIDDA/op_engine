# Prose profile

Version 4, 23 October 2026, [issue 422](https://github.com/yoavram-lab/pp-eigentest/issues/422). **Independently reviewed for application and propagation** in the [version-4 review](https://github.com/yoavram-lab/pp-eigentest/issues/422#issuecomment-6367360639). Installation and manuscript application retain their separate review gates. The version-2/3 approvals remain historical. Read with [README.md](README.md), [prose.md](prose.md), [methods.md](methods.md), [proofs.md](proofs.md), [figures.md](figures.md) and [sources.json](sources.json). Earlier corpus observations and their limits are retained; current requirements are identified separately.

## Observed voice and organization

The sampled papers use a technical, explanatory first-person plural voice. They name the scientific objects and describe what the authors did, then connect the finding to a possible explanation or implication. “We” is especially frequent in the EHS Methods and less frequent in its Discussion. That genre difference matters more than a pooled target rate.

NPZD's introduction and discussion describe the analytical and numerical work in sequence, with ordinary verbs such as define, derive, analyze, examine and find. EHS similarly describes imputation, representation, comparisons and interpretation in the order needed to understand the analysis (Methods, pp. 5–7). Use the method's standard name and explain its role, rather than replace it with a novel label or argue against an imagined criticism.

The second source pass inspected EHS author Methods lines 187–276 against the published main/SI. It makes a more operational pattern visible: name the method, cite it where used, define the objects and describe the actual operation and target. The dedicated [Methods guide](methods.md) records exact passages, the distinction between observations and the author's current requirements, and the main/SI review gate. These requirements apply even if a sampled passage omits a detail.

The observed organization is not identical across papers. NPZD uses numbered findings and a roadmap; EHS ends its introduction with a connected account of findings and opens its Discussion with a semicolon-separated set of results. The approved prose guide recommends matching explicit questions (i)–(iii) to the discussion answers. Adopt that recommendation where it helps the manuscript, without claiming that every sampled paper uses it.

## Sentences and paragraphs

The samples commonly connect a result to a reason, condition or consequence in the same sentence. Relative clauses, “because,” “which,” “suggesting,” and “may be explained by” carry substantive connections. Semicolons join related findings, especially in the EHS Discussion. Paragraphs normally develop one comparison or explanation through several sentences; a short unsupported line is rarely used just for emphasis.

Keep a readable mixture of sentence lengths. Join adjacent short sentences when the second supplies the first's reason; split a long sentence when it changes scientific object, comparison or inferential status. Do not lengthen a clear sentence solely to reproduce a mean. Proof steps and formal hypotheses may remain short.

An approximate diagnostic on inspected prose samples gives the following differences. Rates are per 1,000 extracted words; sentence statistics are words per sentence. Exact ranges, cleaning and tool hashes are in `sources.json`.

| Sample | Words | Mean sentence | 90th percentile | “We” | Hedges | Semicolons | Colons |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EHS Methods | 1,428 | 30.1 | 51 | 22.4 | 1.4 | 6.3 | 1.4 |
| EHS Discussion | 2,493 | 32.5 | 57 | 4.0 | 14.0 | 16.8 | 1.6 |
| EHS SI running text | 1,557 | 27.2 | 46 | 7.7 | 3.9 | 8.3 | 1.9 |
| NPZD introduction/discussion, author TeX | 1,420 | 33.9 | 51 | 11.3 | 4.9 | 0.7 | 0.0 |

These selected passages are not independent samples or an authorship model. PDF citations, headings and residual mathematics affect segmentation; parenthetical author names remain in EHS while TeX citation commands are removed from NPZD. The NPZD revision PDF also contains line numbers and template debris, so its raw extraction is unsuitable for sentence-length targets. The counts describe the exact samples, not a requirement to reach a numerical quota.

These are retained version-1 diagnostics; no new rates were calculated for the second guidance pass. Do not trade a necessary definition, citation, proof step or caveat for a closer match to any measured rate. The author's wording preferences remain controlling, while these numerical summaries remain descriptive screens.

## Word choice, definitions and claims

Observed patterns worth reusing:

- **Name and gloss.** EHS defines cultural microevolution/macroevolution in context (p. 2), introduces archetypal analysis through a short verbal description before its equation (p. 5), and explains cluster metrics by what their values mean (p. 6). NPZD names threshold quantities, gives their formulas and follows them with ecological interpretations (Section 3.1). Define a nonstandard term near first use, then use that term consistently.
- **Concrete actions.** Prefer “we compared,” “we fitted,” “we calculated,” “we examined,” and “we found” when those are the actual actions. “Assess” and “evaluate” are useful when the target is named. Avoid vague replacements such as “the framework leverages a powerful approach.”
- **Calibrated interpretation.** EHS discussion explanations commonly use may, could, appear, suggest and is consistent with. NPZD distinguishes mathematical analysis from what model simulations suggest. State a proved conditional result directly; scope an empirical interpretation to its data, model, target or parameter range.
- **Explained limitations.** EHS SI S1.5 explains why an absent manufacturing practice need not mean an absent traded good. The reusable pattern is an explicit non-implication with its reason. Keep a scientifically necessary contrast, but replace a slogan with the actual dependence or missing evidence.
- **Evidence near the claim.** Both papers place figure/table pointers beside the supported comparison and give estimates or parameter dependence before broader interpretation. Prefer the figure/table at the end of the relevant clause.

The author's present definition-before-use requirement extends beyond nonstandard terms. Introduce every symbol, dimension, acronym, strategy and score before its first substantive use in the main text and separately in the SI. Define a matrix's scientific rows and features as well as its mathematical dimensions. When a formula introduces a new quantity, first define its inputs, units and domain, then explain the output and its role.

Give each method an explanatory name and cite its source at its use in Methods; an Introduction citation alone is insufficient. Keep the method source, software implementation/version and particular protocol identity distinct. For a registered label such as `S1-both`, lead with its actual nominal/ordinal/binary treatment and retain the identifier as a traceable mapping. Explain continuous transformations and their validation status before referring to `S4`. The [Methods guide](methods.md) gives the detailed procedure and audit checklist.

Observed exceptions should not become drafting instructions. NPZD uses occasional emphatic words and spaced hyphen asides; EHS uses “Indeed” and “Notably” in places. The author-approved guide recommends omitting promotional boosters and dash asides. Follow that preference while retaining technical uses such as robust persistence, a named robustness check or a necessary mathematical quantifier.

## Audience adaptation for quantitative ecology and evolution

These are recommendations for the planned audience, rather than patterns uniquely inferred from the papers.

Keep the ecological measurement and question visible while introducing the statistical machinery. Define what rows and features represent; state the prediction target before describing folds or losses; report the selected count together with its biological interpretive limit. Keep detailed run identities and historical audit detail in the SI when the main argument does not need them, while retaining the method citation, essential algorithm choices and relevant implementation identity in the main Methods.

Use standard terms with brief, manuscript-specific definitions. The following vocabulary preserves distinctions already present in pp-eigentest; it supplies no new result.

| Term | Brief definition or first-use treatment |
|---|---|
| Principal components analysis (PCA) | Expand once before using PPCA or VBPCA. |
| Probabilistic PCA (PPCA); variational Bayesian PCA (VBPCA) | Expand once; state their shared low-rank reconstruction role and their different estimation/uncertainty models. |
| Reconstruction capacity, (q) | Number of latent directions available to the fitted reconstruction; identify how it is chosen. |
| Noise-selected component count, (w) | Count selected against a specified reference and decision rule. “Significant” has this declared meaning and does not confer biological importance. |
| Effective rank, (r_{\mathrm{eff}}) | A declared pruning diagnostic at a stated capacity; keep separate from (q) and (w). |
| Variable cell | One sample's measurement for one original variable; its categorical indicator columns are held out together. |
| Parallel analysis (PA); fixed-sequence testing (Seq) | Name before abbreviating; identify the ordered reference and stopping convention. |
| MCAR, MAR, MNAR | Expand missing completely at random, missing at random, and missing not at random at first use. Explain the relevant study-specific mechanism where needed. |
| Partial null | A rank null with some signal directions present; define through the manuscript's ordered rank hypotheses. |
| Retained fitted energy | Fraction of the fitted, bias-excluded spectrum retained; distinguish it from energy captured from a known clean signal. |
| Numerical endpoint sensitivity | A decision or fitted quantity changes between the finite iteration endpoints compared; it is not a biological sensitivity measure. |
| Reference comparison | State whether only the reference changes or reconstruction changes too. |
| Held-out RMS error per indicator | Call this PRMS only as the software's label. It is calculated from mean-prediction residuals; do not expand PRMS as “predictive root mean square.” |

Do not rename “rank” as “communities,” “archetypes,” “clusters” or “biological dimensions” without the downstream model and evidence needed for that interpretation. Preserve unfavorable prediction results, capacity limits and the explicitly pending known-signal validation.

## AI-ism and punctuation screen

Read for repeated habits, not isolated tokens. Remove unsupported superlatives, generic transitions (“Importantly,” “Moreover” without a logical role), repeated three-part slogans, dramatic one-sentence paragraphs and compressed “X, not Y” conclusions that omit the reason. Give the estimate or condition instead of “robust,” “novel,” “key” or “powerful.” A standard technical term remains appropriate when it names the actual object.

Under the approved preference, use no em-dash asides. Keep en dashes for ranges and compound names, including Routh–Hurwitz and Marčenko–Pastur. Keep colons before actual definitions, lists or displayed equations; rework rhetorical colons into a stated relation. Do not alter punctuation inside mathematical notation, a DOI, a source identifier or a required title merely to reduce a diagnostic count.

After editing, inspect the abstract and opening Discussion together, verify that each answer retains its scope, and inspect the SI separately. A low hedge rate in a proof appendix is not a reason to weaken its equalities; a low hedge rate in an explanatory Discussion is a reason to examine whether inference has become too categorical.

## Required second-pass checks

Read the entire main text and SI, including generated material and captions, after applying the reviewed guidance. A paragraph can sound fluent while still lacking a definition, method citation or reproducible operation. Use [methods.md](methods.md) to inspect those failures explicitly and [proofs.md](proofs.md) to inspect complete mathematical arguments. Compare all affected claims, registered outcomes and pending validations with their baseline, and obtain a separate independent review of the manuscript application. Independent review of these guides does not establish that the manuscript satisfies them.

## Version-4 author requirements: connected scientific explanation

These directions transfer the author's explicit requests and the reviewed
issue-415 exposition lessons to manuscripts using this package. They are
editorial requirements, not newly measured habits or additional personal-style
observations. They do not impose pp-eigentest's section order, terminology,
word-count target, venue or deadline on another project.

Before revising a paragraph, identify its scientific question, objects,
operation or evidence, supported inference and relevant condition. This is a
reading aid, not a mandatory sentence template. Name what we fitted, drew,
removed, held out, compared or scored, and explain what that operation makes
observable. Connect an operation to its target and reason, then develop the
interpretation through the actual evidence. A concrete matrix, observation,
event or model can be the subject when its property supplies the explanation;
use “we” when investigator action or choice matters. Repeated first-person
pronouns do not by themselves reproduce an authorial voice.

In Results, establish what was compared and the observed direction, estimate
and scope before interpreting the result. In Discussion, explain why the
comparison affects a scientific decision and which assumptions support a
proposed mechanism. Place a qualification beside the inference it changes,
with its scientific reason. Preserve unfavorable estimates, score
disagreements, failures, undefined cases, uncertainty, capacity boundaries and
unfinished validation. A precise non-implication can supply essential
reasoning. Removing it merely to avoid a negative sentence weakens the text.

Use scientific questions to connect the abstract, Introduction, Results and
Discussion. Keep administrative chronology, internal draft names, old-versus-new
manuscript framing and reporting promises in appropriate project records when
they interrupt the scientific account. Preserve prior scientific attribution,
distinct study settings, essential implementation versions, exploratory versus
confirmatory status and immutable provenance. Do not relabel exploratory or
unfinished evidence as completed confirmation. A manuscript should describe its
present argument without requiring readers to reconstruct its editing history.

Cut repetition through duplicate planning, administrative detail and repeated
explanations after their definitions are secure. Keep the definitions,
method/data/software citations, assumptions, statistical distinctions,
intermediate proof steps and adverse results that make the account assessable.
A project word budget never authorizes removing those elements. Prefer ordinary
verbs, standard terms with brief definitions and connected explanations over
new labels, rhetorical oppositions, slogans and unsupported boosters. Preserve
an established technical term when it names the actual object. Punctuation and
style screens prompt human reading; none is a quota or automatic rewriting rule.

When an example crosses disciplinary language, state the object and claim
precisely. Distances, projections, retained geometry, diagnostic values and
scientific histories can answer different questions. A geometric distortion
example does not establish a theorem about topology or a causal explanation.
Any topological claim needs its own defined spaces, map, property and argument.
Use an analogy only to explain the proved or measured result within its scope.

After application, read the complete main text and supplement independently,
including captions and generated inputs. Check reasoning, definitions,
attribution, proof meaning and author voice against the supported claims, then
obtain separate independent manuscript review and inspect the source-matched
PDFs. Approval of this guidance does not approve that application.
