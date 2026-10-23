# Figure and presentation guide

Version 4, 23 October 2026, [issue 422](https://github.com/yoavram-lab/pp-eigentest/issues/422).
**Independently reviewed for application and propagation** in the [version-4 review](https://github.com/yoavram-lab/pp-eigentest/issues/422#issuecomment-6367360639). Installation and manuscript application retain their separate review gates.
Read with [README.md](README.md), [prose.md](prose.md), [methods.md](methods.md),
[proofs.md](proofs.md) and [sources.json](sources.json).

## Authority and scope

These are explicit author preferences and transferable presentation lessons
from the reviewed pp-eigentest figure/exposition work in issues 411, 415 and 417.
The author now requests their reuse across relevant manuscript repositories.
They are not newly inferred personal-style observations, a survey of figure
frequency or universal journal requirements. The two-paper coauthored prose/
proof corpus, its inspected editions, source hashes and limitations remain
unchanged. Comparator reading and visual access are separate records, not
additional personal-style samples. No new external paper or artwork was read
or retrieved in preparing this version.

Preserve each project's audience, scientific scope, privacy/export rules,
protocol, runtime and journal format. Verify current venue requirements when
applying the guide. No pp-eigentest-specific printed width, label size, palette,
section order or scientific route becomes mandatory elsewhere. Choose the
figure because it explains a scientific operation or supports a comparison,
not because an available plotting package or an appealing template suggests it.

## Show the operation and its scientific meaning

A schematic should let the reader identify what changes at each step. Where
useful, represent the objects directly with a small matrix, missing cells,
held-out block, transformation, fitted representation, ordered spectrum,
geometric projection or downstream comparison. Carry identities through the
sequence so arrows connect recognizable objects. Boxes containing only prose
can leave the actual operation unexplained. Use short defined labels, clear
arrows and a caption connecting the operation to its purpose and output. Draw
original representations; do not trace or copy another paper's artwork.

An arrow needs a declared direction and meaning. An input arrow is not by
itself a causal claim. Distinguish optional or protocol-controlled routes from
universal steps. Show where information is available to preprocessing, fitting,
selection and assessment; protect the actual training/validation boundaries.
Simulation truth can enter assessment only where the declared protocol allows
it, and cannot be drawn as an input to fitting or selection when it is withheld.
Use the particular project's definitions and analysis order rather than
importing the pp-eigentest workflow.

Show zero, tied, failed or unavailable outputs when scientifically possible.
A diagram must not promise a positive selected count, a unique orientation or
a useful low-dimensional display from every input. Keep a fitted quantity
separate from an available known target. A schematic of an assessment does not
establish that the assessment has been completed.

## Keep schematic status in the caption

State in the accompanying caption that invented matrices, point locations,
losses, curves or networks are schematic. Keep that status statement out of the
artwork itself. This is an explicit author preference, generalized by the
current reuse request. Fixed illustrative coordinates usually suffice; do not
run a fit, random simulation or new estimator merely to decorate an explanation.
Explain any empirical-looking threshold, band or selected value's illustrative
role. A schematic supplies no evidence of calibration, recovery, power,
method superiority or a historical mechanism.

A results display instead uses the exact approved source summaries, including
negative findings, denominators, failure/unavailable outcomes and the declared
interval definitions. Its plotting software must not silently estimate a new
mean, confidence band or bootstrap interval. Clearly separate reference-only
contrasts from operational comparisons that change other analysis choices,
where that distinction applies. Never pool distinct studies just because
their estimates fit on the same axis.

## Choose a renderer and preserve accessible meaning

Use native vector tools such as TikZ for small schematics or finite literal
displays when precise arrows, matrices and manuscript typography help. Use
matplotlib, seaborn or another established scientific plotting tool when its
layout tools suit data-rich figures. Explicitly bypass unwanted estimation or
resampling in plotting interfaces. An available package need not become a
runtime dependency, and styling does not authorize upgrading a frozen
scientific environment.

Choose readable colours and redundant markers, line styles, fills or outlines,
with direct labels where useful. Define colour meanings per figure and do not
silently switch model identities or diagram roles. A palette name alone does
not establish accessibility. Fixed CMasher or other palette samples may be
exported to another renderer; record the palette, package edition, source hash,
sample locations and actual colour values. CMasher is an option, not a mandated
dependency or proof of colour-vision accessibility.

Set fonts and native geometry for the intended printed width. Inspect the
actual embedded fonts and final label size; scaling an oversized figure can
make its nominal font specification irrelevant. Prefer vector geometry where
appropriate and inspect unexpected raster objects. Native dimensions and font
listings alone do not establish legibility or compliance with a journal's
current submission rules.

## State precision, uncertainty and provenance

Preserve approved full-precision values and interval endpoints in source-bound
artifacts. Set a deliberate display precision, explain it where useful and
check that rounding does not obscure a relevant difference or adverse value.
Distinguish standard errors, confidence intervals, pointwise and simultaneous
intervals, descriptive rows and registered contrasts. Identify the relevant
sample/replicate counts, weighting, pairing, units and normalization. Pairs or
quartets from one dataset can share observations; their number is not
automatically an independent replication count.

A distribution of pairwise distances and a display of matched named pairs
answer different questions. A scientific caption should identify which is
shown and which claim it supports. Define the geometry and diagnostic before
using a network or topological analogy. Agreement with known geometry, a
relative ordering or a diagnostic target does not identify a true causal
history without the additional evidence and assumptions.

Retain source hashes, source-row mappings or fixed schematic coordinates,
renderer/palette identity, build commands and a source-matched PDF receipt.
Record corrections while retaining earlier failed renders and review findings.
Keep archived source snapshots as text when their role is historical evidence,
so a snapshot cannot accidentally execute as an active module. Meaningful
checks bind data or behavior; low-impact literal artwork does not need tests
that duplicate every drawing command.

## Review the actual integrated figure

Actually open the compiled figure and a complete caption page at its intended
width, including a grayscale preview. Check clipping, symbol rendering,
line/marker visibility, arrow routes, labels, domain boundaries, zero/tie cases
and scientific meaning. Reinspect the source-matched figure after manuscript
integration because float placement, page geometry and scaling can change the
effective display. Review colour-vision accessibility as appropriate rather
than treating grayscale inspection as a complete accessibility test.

Review the figure and caption together, then read the complete main text and
supplement independently for definitions, citation support, proof meaning,
statistical scope, adverse findings and consistency. Compilation, guidance
approval and standalone artwork review are separate from independent complete
manuscript application review. Run the project's applicable checks and normal
Git/review workflow before merge or publication. This guide admits no new
scientific run, endpoint or protocol change.

## Provenance of the transferred lessons

The issue-417 project guide, pinned to the reviewed manuscript integration at
[commit 0c04269](https://github.com/yoavram-lab/pp-eigentest/blob/0c04269f302d9147eed674a33678df7065021d4c/docs/author_style/pp_eigentest_figures.md),
records its purpose-specific comparator search. The earlier-author-preprint
artwork of Greenacre et al. (2022), Figure 2 on PDF page 3, was actually viewed
in that earlier task. Published-version caption/text access did not establish
published-artwork inspection. Abid et al. (2018) and Kujala et al. (2018) supplied
accessible caption/text examples; their artwork was not viewed in that task.
No claim here rests on an unviewed exemplar's legibility or on a prevalence
survey. [sources.json](sources.json) pins the inherited guide and access record.
These examples explain transferable presentation choices; none determines a
new manuscript's scientific procedure.
