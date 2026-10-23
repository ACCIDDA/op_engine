# Proof and mathematical exposition profile

Version 4, 23 October 2026, [issue 422](https://github.com/yoavram-lab/pp-eigentest/issues/422). **Independently reviewed for application and propagation** in the [version-4 review](https://github.com/yoavram-lab/pp-eigentest/issues/422#issuecomment-6367360639). Installation and manuscript application retain their separate review gates. The version-2/3 approvals remain historical. Read with [README.md](README.md), [prose.md](prose.md), [methods.md](methods.md), [proofs.md](proofs.md), [figures.md](figures.md) and [sources.json](sources.json). Earlier corpus observations and their limits are retained; current requirements are identified separately.

## Observed structure and limits

The directly inspected proof model is Macdonald and Gulbudak's NPZD paper, especially Section 3.1, the planar global-stability argument in Section 3.2, and Appendix A's local-stability arguments. The revision TeX was checked alongside the extracted PDFs because the revision PDF has template and margin-number artifacts. The EHS paper supplies examples of definitions and empirical interpretation, not a comparable theorem–proof sequence. Its SI is not evidence for a general preference in formal proof writing.

NPZD defines a named threshold before the proposition, states parameter conditions before the conclusion, and uses explicit mathematical objects to move from assumptions to the result. Routine proofs are in Appendix A; one main-text argument begins from invariant sets, displays the relevant matrix/calculation, applies a named criterion, and returns to the stated conclusion. Threshold propositions are followed by parameter-dependent ecological readings. These are observed organizational patterns, not a certification of every source argument or an instruction to reproduce abbreviated steps such as “clear from the form.”

The second source pass re-read the complete relevant author-TeX arguments and their setup, including the existence theorem's enumerated hypotheses and its application. These exact examples explain the observed organization:

| Author-TeX passage | Observed construction and limit |
|---|---|
| Lines 285–350 | Positive parameter domain and named boundary equilibria precede the threshold; line 312 says “Define the zooplankton invasion number”; lines 318–325 give the conditional stability statement and parameter-limit interpretation. The complete coexistence argument, lines 327–350, starts from equilibrium conditions, supplies components and a threshold equivalence, then returns to feasibility. Domain checks and any compressed positivity/uniqueness reasoning still need independent mathematical verification in a new proof. |
| Lines 424–497 | The conservation law and invariant set precede the global-stability proposition. The proof begins “First, note that the lines $p = 0$ and $z = 0$ are invariant”, displays Jacobians and a sign calculation, examines boundary dynamics and concludes with the claimed attracting equilibrium. This is a sequence to make explicit, not a certification of its global argument or permission to omit cited-criterion hypotheses. |
| Lines 510–588 | A cited equilibrium-existence theorem lists five hypotheses before the proposition; the proof describes boundedness, identifies $\rho$, supplies boundary eigenvalues and cites numbered persistence results. It ends “all of the conditions ... are either met or exceeded”. That ending alone would not verify every hypothesis; the current requirement is an explicit source-specific mapping of each condition to its justification. The original theorem/persistence sources were not read in this style pass. |
| Lines 1005–1055 | Appendix arguments display evaluated Jacobians; one identifies the characteristic polynomial as “assisted by MatLab Symbolic Math Toolbox”, displays Hurwitz determinants and invokes named criteria. Some short proofs stop after a matrix display. Preserve useful displayed structure while supplying the missing inference in a new manuscript; do not adopt an abbreviation as the rigor standard. |

These observations concern mathematical exposition in a non-pathogen ecology paper. They do not establish correctness of all its conclusions, completeness of its hypotheses or sentence-level authorship.

The repository's [proof guide](https://github.com/yoavram-lab/pp-eigentest/blob/f644bfb3c02c99bf6ffb109fd44434e94daeeb0e/audit/proof_style_guide.md) and [proof plan](https://github.com/yoavram-lab/pp-eigentest/blob/f644bfb3c02c99bf6ffb109fd44434e94daeeb0e/audit/proof_plan_2026-10-16.md) strengthen this into an explicit rigor requirement: hypotheses, logically verified steps, conclusion and interpretation. Their other source papers were not read in this extraction, so requirements inherited from those guides must not be presented as new corpus observations.

## Reusable statement and proof sequence

The author's current requirement is a **complete, coherent statement, proof and interpretation**. A checksum span, algebraic core, proof-plan entry, automated test or historical audit fragment is an internal evidence unit. It is neither a proof fragment for readers nor a limit on the written argument's scope or presentation. Read the whole surrounding proposition and all its dependencies when assessing completeness.

1. **Define the objects.** Specify every matrix, vector, function, probability space/model, dimension, index range, norm, parameter domain and threshold quantity before using it. Distinguish a fitted estimator from its generating target. Name limiting regimes, fixed versus varying quantities, conditioning and any probability-one qualifications. State the existence conditions for the stated objects, including dimension requirements for an orthogonal example.
2. **State hypotheses first.** Give every condition before “then,” including inherited assumptions that remain needed locally. Use separate clauses for distinct directions of a result. Keep necessity, sufficiency, an equivalence and a counterexample distinct. State ties, equality thresholds, zero/rank-deficient cases, excluded boundary cases and exceptional events explicitly; explain a restriction rather than silently passing across it.
3. **Begin from an available fact.** Start with the relevant identity, invariant set, decomposition or stated hypothesis. Ordinary connectives such as “First, note that,” “Hence,” and “It follows that” can identify actual steps; they cannot substitute for them.
4. **Give every logical and algebraic step.** Define terms and normalizations, show sign or domain conditions before multiplying/dividing, and connect each display to the preceding one. Supply intermediate equalities/inequalities, root choices, monotonicity, limits and case boundaries needed to check the conclusion. Prove both directions of an equivalence and every part of a multi-part statement. A sketch, a bare matrix display, “similarly,” “clearly” or a claim that the remaining calculation is routine cannot stand in for an unprovided argument. Use short mathematical sentences where they make dependencies easy to check.
5. **Apply a cited theorem precisely.** Inspect the primary theorem in the edition cited. Give the source and exact theorem/lemma/corollary number, state the relevant conclusion and enumerate every hypothesis used. Map its notation to the present objects and verify each condition in turn with a logical argument, an already proved result or a valid symbolic calculation. Verify regularity, independence/isotropy, domain, dimension/aspect-ratio, finiteness/compactness and convergence assumptions as applicable to the actual theorem. If a centring, transpose, rescaling or reduction changes an object, explain why the theorem applies to the transformed object and how the result returns to the original one. Naming a criterion or citing a theorem does not establish applicability. If its source cannot be inspected, record that unresolved step and retain the corresponding proof limitation.
6. **Conclude the whole stated claim.** Identify the event relation, limit, sign, root or identity that yields each conclusion, including ties, exceptional events and boundary exclusions where relevant. Explain how finite-draw statements and probability limits combine; justify finite union bounds or uniformity when passing from one component/draw to many. Do not extend a component statement to a fitted pipeline without the additional argument.
7. **Interpret after the proof.** Explain the practical reading and the dependence on assumptions. A finite identity and an asymptotic statement can require different interpretations. For an ecological application, explain what the result permits one to say about the measurements or representation without assigning biological meaning to an algebraic direction.

A manuscript can refer a routine proof to the SI while retaining its conditions and meaning in the main text. Keep an illuminating short proof near its result when the journal and document structure permit. Summary tables can collect quantities, domains and interpretations; they should not replace the statements.

## Quantifiers, connectives and local definitions

The requirements in this section and the limit/Lean sections below come from the author's current issue-401 instruction. They are not inferred from word frequencies or newly read papers. Their purpose is to expose the exact mathematical claim before checking it.

For each complete result, keep a binder table in the review record. List every symbol, its mathematical type/domain and units where relevant, its binder or defining clause, scope, order and permitted dependence. Include implicit and inherited parameters, indices, witnesses, random outcomes, exceptional sets and constants hidden in asymptotic notation. Record whether an object is fixed, varies with the limit index, is chosen from data or is an arbitrary member of a domain. The written statement must expose every distinction needed to interpret the claim; the audit table does not replace it.

| Entry | Domain and binder | Scope/order | Allowed dependence and obligation |
|---|---|---|---|
| Fixed parameter `a` | `a ∈ A`, arbitrary, fixed before the limit | Outside `ε` and `N` binders | State restrictions defining `A`; no hidden change with the sequence index. |
| Tolerance `ε` | `∀ ε > 0` | Before choosing `N` | Arbitrary; never selected after the index it bounds. |
| Threshold `N` | `∃ N ∈ ℕ` | After `a, ε`, before `∀ n ≥ N` | May depend on `a, ε`; may not depend on the later arbitrary `n`. Say whether dependence on another parameter is allowed. |
| Witness `x` | `∃ x ∈ D(a)` with `P(a,x)` | After the parameters on which it may depend | Supply existence in this domain. If unique, compare with every other witness in this same domain. |
| Random threshold or null set | `N(ω,ε,a)` or `E_a ⊆ Ω` | Explicit relation to `∀ a` and “almost surely” | Distinguish a parameter-specific exceptional set from one set working for every parameter. |

Use the following meanings consistently in statements, proofs, captions and interpretations.

| Wording | Logical role and required check |
|---|---|
| “For every,” “for all,” “arbitrary” | Universal quantification `∀ x ∈ D`. State `D`; the argument cannot rely on an unannounced special choice of `x`. An empty domain can make a universal statement vacuous, so check nonemptiness where the interpretation needs an object. |
| “There exists” | Existential quantification `∃ x ∈ D, P(x)`. Give a witness or a valid existence argument. Existence does not assert uniqueness, a formula, computability or convergence of an algorithm to that witness. |
| “There exists a unique” | `∃! x ∈ D, P(x)`: existence in `D`, then equality of every other solution in `D` to that witness. State the domain and equality notion; uniqueness of a positive root, a fitted matrix, a coordinate representation or an equivalence class are different claims. Non-identifiability up to sign or rotation requires its own equivalence relation. |
| “Let,” “define,” “denote” | Introduces a local variable, abbreviation or notation under stated conditions. “Let `x ∈ D` be arbitrary” sets a universal context; “let `x` satisfy `P`” needs a prior existence justification if used to prove an unconditional conclusion. A formula is checked for its domain and well-definedness. “Define the unique root” cannot establish either existence or uniqueness. |
| “Suppose,” “assume” | Adds a hypothesis within a stated scope, including a case or contradiction argument. It does not prove that hypothesis for the application. Discharge temporary assumptions when concluding, and distinguish supplied assumptions from established conditions. |
| “And” | Conjunction `P ∧ Q`; establish both claims. A multi-part conclusion needs an argument for each part. |
| “Or,” “and/or,” “inclusive or” | Mathematical `P ∨ Q` is inclusive: either may hold and overlap is allowed. Prefer “or,” adding “or both” when ambiguity matters. If exactly one is intended, state exclusivity and prove `(P ∨ Q) ∧ ¬(P ∧ Q)`. “Either … or …” alone does not certify disjointness. |
| “If … then” | Implication `P → Q`; check the stated antecedent before applying it. It does not assert the converse `Q → P`. |
| “If and only if,” “equivalent” | `P ↔ Q`; prove both implications under the same declared domains and hypotheses. A counterexample to one direction refutes the equivalence; identify the direction and satisfy its antecedent. |

Parenthesize mixtures of “and,” “or,” implication and quantifiers whenever scope could change meaning. Moving a binder across a connective is a mathematical step. Do not replace `∀ a, ∃ x, P(a,x)` by `∃ x, ∀ a, P(a,x)`; the first permits a different witness for each `a`. A family of uniquely determined scalar values also differs from uniqueness of a tuple or function; state the object being compared. Keep necessity, sufficiency and a counterexample separate from stylistic hedging.

Before any case split, division, inversion, logarithm, square root, normalization or denominator clearing, specify the relevant domain, zero/equality cases and selected branch. State whether alternatives overlap; prove coverage and any claimed disjointness. A polynomial root need not be a feasible positive model solution, and squaring a relation can lose its branch/sign restriction. Record all such changes in the written argument and formal correspondence table.

## Limits, asymptotic statements and dependence

State the actual sequence or function, index set and source/target filters or limiting points, together with every fixed and varying parameter. A proportional-dimension regime must identify `n_k`, `p_k`, their aspect ratio, noise/strength/capacity parameters, and which converge, diverge or remain fixed. Define `o(1)`, `O(·)` and probabilistic variants with their mode and dependence; do not silently make a pointwise bound uniform. For subsequence arguments, quantify the admissible subsequences and prove the passage back to the whole sequence if claimed. The existence of a convergent subsequence does not prove convergence of all solutions or of a fitted algorithm.

For real sequences, expose the order `∀ ε > 0, ∃ N, ∀ n ≥ N, |u_n − L| < ε`. List every permitted dependency of `N`, of a chosen margin and of constants. Choose a strict positive separation margin before sending the dimension to infinity; justify how convergence makes the finite quantities separated. Equal limits alone decide neither strict ordering nor its selection probability. Show eventual positivity/nonzeroness before division or other operations requiring it; a formal language's totalized division or square root must not silently enlarge the model domain.

Distinguish pointwise convergence, `∀ x ∈ D, ∀ ε > 0, ∃ N(x,ε), ∀ n ≥ N, …`, from uniform convergence, `∀ ε > 0, ∃ N(ε), ∀ x ∈ D, ∀ n ≥ N, …`. Give the additional bound or compactness/equicontinuity argument required by the actual result, rather than exchanging `∀ x` and `∃ N`. A union bound over a fixed finite set permits a common eventual threshold; a set whose cardinality grows with dimension needs a rate or another uniform argument. In particular, fixed capacity `q` or a selector bounded by a fixed cap `Q` is different from dimension-growing `q_k` or `Q_k`. Pointwise control for every fixed `q` does not control an unbounded data-chosen cap.

For stochastic limits, define the probability space or sequence of laws and say explicitly “in probability,” “almost surely” or “in distribution,” with the corresponding quantified event/law statement. Do not substitute a deterministic `Tendsto` assertion for an unproved stochastic mode, or infer pathwise inequalities from convergence in distribution. For almost-sure statements, distinguish “for every parameter, almost surely” from “almost surely, for every parameter.” The former may use a parameter-dependent null set; the latter requires a common probability-one set. Countable intersections require a countable index set; an uncountable family needs an additional justification. State any random dependence of `N` on `ω` and the requirement that `ε` or other parameters be handled on the asserted common event.

Identify the measurable random objects/events and integrability or finite-moment requirements used in the probability argument. Some measure inequalities hold for arbitrary sets under the formal measure convention; this does not by itself justify interpreting an application as a probability of a measurable event. Define conditioning information and conditional laws, and verify positive conditioning probability when conditioning by an event; zero-probability conditioning needs an appropriate stated construction. Distinguish conditional hypotheses proved for the application from those merely assumed by a lemma. Explain any transfer from conditional to unconditional statements and its null-set qualifications.

## Lean checks of logical form, hypotheses and limit bridges

Use Lean as a primary internal check of binder order, domains, connective meaning, existence/uniqueness claims, hypothesis application and limiting implications where feasible. Formalize the actual statement being used, including asymptotic bridges, rather than stopping automatically at a finite algebraic rearrangement. Use the repository's pinned Lean/mathlib versions and actual `Filter`, `Filter.Tendsto`, eventuality, measure and probability definitions; inspect their types and hypotheses locally. Record a material unformalized step explicitly instead of hiding it behind a new assumption. Prioritize critical claim gaps; a guidance revision does not authorize a broad new formalization campaign or new scientific runs.

Ordinary Lean `Or` (`∨`) is inclusive. Pinned mathlib `ExistsUnique p` means `∃ x, p x ∧ ∀ y, p y → y = x`; a restricted domain must appear in the type or predicate. Nested uniqueness binders are not a shortcut for uniqueness of a tuple. `let` creates a local definition; an input named `hlimit`, `hexists` or `hregular` remains an assumption until the application supplies its proof. These semantics are confirmed from the pinned local sources, not from a claim that Lean guesses the intended English.

Maintain a TeX–Lean correspondence table for each checked result with: the full prose/TeX statement and location; binder/domain/dependency map; exact Lean declaration and source hash; transformations and totalized operations; formal hypotheses; where each is proved in the application; conclusion and convergence mode; unformalized bridges; and the applicable build/axiom receipt. A kernel-checked theorem verifies its encoded statement from its assumptions. Types and the kernel alone do not certify that this encoding matches the intended quantifiers or the scientific model. Independently review that correspondence in both directions.

Use no `sorry`, `admit`, new axioms or unproved imports of the desired application. Run `#print axioms` for every relevant declaration and the repository's existing declaration/axiom audit, which checks complete reports and allows only the configured standard axioms (`propext`, `Classical.choice`, `Quot.sound` in pp-eigentest). Record compiler/dependency pins, source hashes, declarations, executed checks and failures. Compilation and the axiom audit support the formal coverage claim only; an unused-assumption linter can suggest a weakening but cannot certify globally minimal model hypotheses. A failed proof attempt alone is not proof that an assumption is necessary.

In pp-eigentest the inspected toolchain is Lean `v4.34.1`, with mathlib revision `d13f23b723b8a846827a245b89c10fc7d3f11612`. Existing `edge_normalisation_tendsto`, `mcar_factor_tendsto_iff`, `finite_union_tendsto_zero`, `positive_rank_prob_tendsto_zero` and `hyperprior_polynomial_limit` already formalize specified deterministic/measure limit bridges under explicit inputs. `pointwise_capacity_does_not_control_growing_cap` and `equal_limits_do_not_decide_selection` exhibit failures of invalid extensions. These are concrete starting points, not evidence that every cited random-matrix limit, convergence assumption, null-set uniformity or fitted-pipeline application is formalized. The stationary-polynomial limit assumes convergent coordinate sequences and an eventual root equation; it does not prove those assumptions for random fits.

Retain a readable complete proof with setup, logical and algebraic steps, source-specific cited-theorem hypotheses, conclusion and interpretation. A list of Lean declaration names or proof terms cannot replace this. A conditional finite core is not a complete asymptotic theorem, and an isolated `Tendsto` bridge is not formal verification of the full random-matrix application.

## Claim strength and checks

Use the existing status vocabulary without promotion by style:

| Status | Required evidence and wording |
|---|---|
| proved | A written argument from stated hypotheses. State the conditional result directly. |
| proved (cited theorem) | A precisely cited theorem with every relevant hypothesis verified logically or symbolically for this application. |
| checked-only | Numerical/simulation evidence. Label it as a numerical observation or diagnostic and give its scope. |
| open | No established argument. Keep the extension explicitly open. |

The current registry also uses “proved (written)” for particular entries; preserve registered labels rather than normalize them editorially. A test that finds no counterexample does not prove a universal statement. A finite tolerance check does not establish exact equality or convergence. A symbolic identity can support an exact step when its domain assumptions are supplied; check denominator nonzeros, signs, branches, simplification assumptions and equality boundaries. A finite Lean core establishes only its formal statement with its supplied assumptions. Lean and SymPy checks are internal support for the written argument and do not replace it, prove an unformalized cited-theorem application or justify a manuscript claim that the whole analysis is formally verified. Numeric checks describe only the evaluated parameter sets and tolerances.

Minimal hypotheses deserve mathematical review, not stylistic deletion. An assumption unused by a finite Lean core may still be needed by the surrounding stochastic or cited-theorem argument. Conversely, do not add familiar model assumptions merely to make an elementary event inclusion look more substantial. Distinguish sufficient conditions from necessary ones and finite statements from their asymptotic simplifications.

## Prose and notation within proofs

Use the same symbols for the same object, with a clear introduction when a local calculation reuses a symbol. Preserve norm choice, centring, index ordering, strict/inclusive comparisons, probability mode, quantifier order and units/normalizations. These carry meaning even when a sentence appears to need only a small wording change.

Keep necessary colons for definitions and equations, semicolons that distinguish cases, dashes in mathematical names/ranges and exact quantifiers. Avoid rhetorical asides, promotional adjectives, statements that the argument is “obvious,” and narrative about how impressive or delicate the proof is. Replace an omitted inference with the relevant identity or condition; do not merely replace “clearly” with “therefore.”

An interpretation paragraph should explain a non-implication when it matters. For example, in pp-eigentest a count selected against a reference does not identify a biological grouping; the count concerns a fitted spectrum and downstream grouping needs its own model. That explanation is more useful than a slogan about what the count is “not.” Preserve the reason and the scope even if a diagnostic flags the negative wording.

## Applying this profile to pp-eigentest

Propositions 5–7 in `manuscript/signal_recovery_proofs.tex` use event inclusion, an orthogonal two-direction example and a pairwise-distance identity respectively. Their presence and registered status do not exempt their complete statements, definitions, arguments and interpretation from the new semantic review. Inspect dimensions, local notation, event qualifiers and all intermediate steps as well as the conclusion. Do not rewrite a valid argument merely to match a sentence-length or first-person target.

The scope-v4 registry binds seven earlier and three added proposition spans; it also binds the full signal-recovery proof file for the measurement-scale remark. These spans protect the traceability of previous audits. The author has authorized needed improvements inside bindings for issue 376. After independent review of the guidance, make a warranted improvement rather than preserving an incomplete proof for its hash. Record the exact affected proposition/file, full before/after mathematical meaning and whether hypotheses, conclusions, proof dependencies or only exposition change. Obtain independent semantic mathematical review of the complete affected argument and its downstream interpretations. If bound bytes genuinely change, use the theory-audit workflow to create a prospective scope version tied to the reviewed new content and rerun its applicable checks. Retain earlier scopes, ledgers, receipts and other historical evidence. Never regenerate old hashes or edit an old audit record solely to clear a check.

No manuscript, binding, ledger or historical-evidence edit belongs to the guidance-preparation stage. The preliminary inventory records candidates for the later reviewed application; it does not constitute mathematical approval.

Never let readability edits imply that the MCAR moment identity is a fitted-reconstruction power law, that a stationary-point calculation proves algorithmic convergence, that a reference has calibrated composite-null error without the needed argument, or that correct counts establish directional/amplitude recovery. The known-signal empirical recovery analysis remains pending until separately completed and qualified.

## Complete-proof audit for the next agent

For each mathematical result in the main text and SI, identify its complete statement, setup, proof, interpretation, citations and downstream summaries; do not select only registered checksum spans. Check:

- Every symbol, dimension, parameter domain, probability qualification and case boundary is defined before use, with a consistent map to earlier notation. A binder table states order, scope and permitted dependence; “let,” “define,” “denote,” “suppose,” existence, uniqueness, “and” and inclusive “or” have the meanings above.
- All hypotheses are explicit and sufficient for each construction and cited theorem. Claims of necessity or minimality have their own justification.
- Every part of the conclusion follows through visible logical and algebraic steps; every equivalence has both directions; transformations, divisions, roots, limits, orderings and tie cases are justified.
- Each cited result has a verified edition and exact location, a notation map and a hypothesis-by-hypothesis applicability argument. Record genuinely unresolved applications instead of silently certifying them.
- Symbolic/formal checks match the written expressions, domains, binder order and convergence modes through a TeX–Lean correspondence table. Limit bridges use actual pinned interfaces where feasible; supplied convergence/hypothesis inputs are separately discharged for the application. Numerical results remain scoped checks. None substitutes for the written proof.
- Every limit records its sequence/filter, fixed and varying quantities, ε/N dependencies, pointwise/uniform scope, probability mode, exceptional-set dependence, conditioning, measurability and eventual domain conditions. Fixed finite caps are not silently extended to growing caps.
- The interpretation and main-text summary preserve the proposition's quantifiers, finite/asymptotic regime and scientific limits, and do not claim pipeline properties from a component calculation.
- Any warranted binding change receives semantic mathematical review and prospective audit scope treatment; historical evidence remains intact.

Resolve defects in the complete argument before polishing its cadence. Independent review of this guidance and the later mathematical review of manuscript changes are separate gates.

## Version-4 presentation and cross-domain claim checks

These requirements generalize recent reviewed project lessons without adding a
new theorem, a new style source or permission for a broad formalization effort.
The complete argument remains the unit of mathematical review. A registered
span, stored checksum or formally checked algebraic component is an audit unit;
it does not permit an incomplete statement or fragmentary proof for readers.
Use the project's current scope and correspondence records when reviewing an
application; dated pp-eigentest examples above describe their recorded stage,
not the current state of another repository or its formal toolchain.

For an existence example or counterexample, define the whole object, domain,
map and target assertion before calculating. Show that the construction meets
all hypotheses and the intended quantifier pattern. A single counterexample
can establish failure of a universal guarantee, but not the frequency of that
failure, a noisy fitted method's behavior or its scientific cause. Separate a
unique subspace, a coordinate representation and a representative vector:
sign/rotation equivalences and repeated eigenvalues change uniqueness claims.
Keep ties, coincident projected points, zero denominators and unavailable
normalized quantities in the full argument and interpretation.

When moving between linear algebra, geometry, metrics, networks and topology,
state which mathematical object and property are preserved or lost. A
projection may merge distinct observations, so declare whether the resulting
distances are a metric on the images or a pseudometric on the labelled original
objects when that distinction matters. Do not describe altered distances or a
four-point diagnostic as a topological theorem without defining the relevant
spaces, map and invariant and supplying its proof. A known geometric example
also does not determine a causal or transmission history. These are scope
checks, not assertions that a new topology section belongs in the manuscript.

Retain every written logical/algebraic step needed for the result, and use
symbolic checks with explicit domains and adversarial boundary cases to verify
derivations. Keep numerical sanity checks scoped to their evaluated cases.
Use Lean with the existing pinned toolchain for critical quantifiers,
hypothesis use, exact algebra and feasible limit bridges; maintain the TeX–Lean
map and axiom audit. Formal coverage must say which supplied assumptions and
bridges remain outside the formal result. No screenshot, numerical test,
symbolic simplification or kernel check substitutes for a complete readable
proof and independently reviewed interpretation.

Any warranted change to bound mathematical content follows the repository's
prospective semantic-review and audit-scope process. Retain old failed attempts,
scopes and receipts; never rewrite historical hashes to make an edit pass.
Check the abstract, Results, captions and Discussion for downstream changes in
quantifiers or claim strength, then obtain independent complete manuscript
review in addition to the guidance review.
