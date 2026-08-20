"""Render the camera-ready manuscript as a print-ready HTML document.

Typography follows the Springer LNCS conventions used by MICCAI workshops:
single column, Times, captions above tables and below figures.
"""
from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
OUT = HERE / "Paper-35_camera_ready.html"


def data_uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((FIG / name).read_bytes()).decode()


def figure(name: str, number: int, caption: str, width: str = "100%") -> str:
    return (
        f'<figure>\n<img src="{data_uri(name)}" style="width:{width}">\n'
        f'<figcaption><span class="lab">Fig. {number}.</span> {caption}</figcaption>\n</figure>'
    )


CSS = """
@page { size: A4; margin: 20mm 24mm 20mm 24mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0 auto; max-width: 162mm; background: #fff; color: #000;
  font-family: "Times New Roman", Times, serif;
  font-size: 10pt; line-height: 1.30; text-align: justify;
  hyphens: auto; -webkit-hyphens: auto;
}
.screen-pad { padding: 16mm 14mm; }
h1.title {
  font-size: 14.5pt; font-weight: bold; line-height: 1.25;
  text-align: center; margin: 0 0 14pt; hyphens: none;
}
.author { text-align: center; font-size: 11pt; margin: 0 0 4pt; }
.affil  { text-align: center; font-size: 9pt; font-style: italic; margin: 0 0 2pt; line-height: 1.3; }
.mail   { text-align: center; font-size: 8.5pt; margin: 0 0 14pt; font-family: "Courier New", monospace; }
.abstract { margin: 0 8mm 6pt; font-size: 9pt; line-height: 1.28; }
.abstract .lab { font-weight: bold; }
.kw { margin: 0 8mm 16pt; font-size: 9pt; }
.kw .lab { font-weight: bold; }
h2 { font-size: 11pt; font-weight: bold; margin: 14pt 0 5pt; text-align: left; hyphens: none;
     page-break-after: avoid; }
h3 { font-size: 10pt; font-weight: bold; margin: 10pt 0 3pt; text-align: left; hyphens: none;
     page-break-after: avoid; }
p { margin: 0; text-indent: 1.4em; }
p.first, h2 + p, h3 + p { text-indent: 0; }
figure { margin: 9pt 0 9pt; text-align: center; page-break-inside: avoid; break-inside: avoid; }
figure img { max-width: 100%; height: auto; }
figcaption { font-size: 8.5pt; line-height: 1.25; text-align: justify; margin-top: 5pt; }
figcaption .lab, table caption .lab { font-weight: bold; }
table { border-collapse: collapse; margin: 10pt auto 12pt; font-size: 8.5pt; width: 100%;
        page-break-inside: avoid; }
table caption { caption-side: top; text-align: justify; font-size: 8.5pt; line-height: 1.25;
                margin-bottom: 4pt; }
th, td { padding: 2.6pt 5pt; }
thead th { border-top: 0.9pt solid #000; border-bottom: 0.5pt solid #000; text-align: left;
           font-weight: normal; font-style: italic; }
tbody tr:last-child td { border-bottom: 0.9pt solid #000; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; }
tr.rule td { border-top: 0.4pt solid #999; }
ol.refs { font-size: 8.5pt; line-height: 1.20; padding-left: 0; margin: 3pt 0 0; counter-reset: r; }
ol.refs li { list-style: none; text-indent: -1.5em; padding-left: 1.5em; margin-bottom: 0.6pt;
             text-align: left; hyphens: none; }
ol.refs li::before { counter-increment: r; content: counter(r) ". "; }
.small { font-size: 9pt; }
.nohyph { hyphens: none; }
@media print { .screen-pad { padding: 0; } body { max-width: none; } }
"""

BODY = """
<h1 class="title">Leakage-Controlled Resting-State Connectome Classification
of Schizophrenia: A Nested Cross-Validation Audit of the COBRE Cohort</h1>

<p class="author">Ishsirjan Kaur Chandok</p>
<p class="affil">Institut de G&eacute;n&eacute;tique Mol&eacute;culaire de Montpellier, CNRS UMR 5535,<br>
1919 Route de Mende, 34293 Montpellier Cedex 5, France</p>
<p class="mail">ishsirjanchandok.iskc@gmail.com</p>

<div class="abstract">
<p class="first"><span class="lab">Abstract.</span>
Resting-state functional connectomes are widely used to discriminate patients with
schizophrenia from healthy controls, yet reported cross-validated accuracy depends
heavily on the point at which features are defined relative to the train/test split.
We audit this dependence on 146 participants of the COBRE cohort parcellated with the
Schaefer-100 atlas (4,950 edges). Exploratory covariate-adjusted mass-univariate
inference is kept strictly separate from prediction, and every predictive pipeline
performs edge screening, dimensionality reduction and hyperparameter search inside the
outer training folds of a repeated nested cross-validation. Under this regime,
elastic-net logistic regression reaches an AUC of 0.768&nbsp;&plusmn;&nbsp;0.076 and
PCA-30 an AUC of 0.771&nbsp;&plusmn;&nbsp;0.079, with an edges-only elastic net
essentially unchanged at 0.765&nbsp;&plusmn;&nbsp;0.079. Fold-wise residualisation of
mean framewise displacement lowers that edges-only estimate to 0.694. Holding a
twenty-edge <span style="font-style:italic">|t|</span> screen and a logistic classifier
fixed, and varying only whether the screen sees the test fold, isolates leakage for that
construction: AUC rises from 0.709&nbsp;&plusmn;&nbsp;0.076 to
0.903&nbsp;&plusmn;&nbsp;0.039. That 0.194 gap is the cost of leaking a hard univariate
screen, not of leaking the primary elastic net. The same construction on a 50/50
age-, sex- and FD-matched subsample of UCLA CNP yields 0.679 against 0.873, again a
0.194 gap; that analysis is within-cohort nested cross-validation, not a locked
transfer, and does not revise 0.77. On synthetic data with no signal the
leakage-free protocol returns chance, whereas a leaked twenty-edge screen reports 0.82
when the edges are correlated as in a connectome and 0.94 when they are independent; the
observed leaked value of 0.903 lies between those bounds and about 0.08 above the
correlated floor. Calibration and subject-level aggregation of repeated out-of-fold
predictions are reported for the leakage-free model. The resulting figure of roughly
0.77 AUC is specific to this cohort, parcellation, connectivity measure and model family
rather than a general ceiling, and the audit protocol is released as public code.
</p>
</div>
<p class="kw"><span class="lab">Keywords:</span> Functional connectivity &middot;
Cross-validation &middot; Data leakage &middot; Schizophrenia &middot;
Validation protocol</p>

<h2>1&ensp;Introduction</h2>
<p class="first">Machine learning applied to functional neuroimaging is increasingly
proposed as a route to computer-aided diagnosis in psychiatry&nbsp;[1,&thinsp;2].
Schizophrenia is associated with altered resting-state functional connectivity,
particularly across default-mode, salience and frontoparietal systems&nbsp;[3,&thinsp;4,&thinsp;5],
and a full pairwise connectome supplies thousands of candidate features from a single
scan. The apparent accuracy of a connectome classifier, however, is determined as much
by the validation protocol as by the biology it is meant to capture.</p>

<p>A frequent pipeline screens edges on the whole cohort, retains the significant
connections, and only then trains a classifier that is evaluated by cross-validation.
Selection performed at that point encodes label information from participants who will
later serve as test cases, and nesting the classifier alone cannot remove the resulting
optimism&nbsp;[1,&thinsp;6]. The distinction between exploratory group-level inference
and predictive feature construction is therefore not a stylistic one: the two answer
different questions and require different guarantees.</p>

<p>We quantify that distinction on the public COBRE cohort&nbsp;[7]. Our
contributions are fivefold. First, we report a leakage-free benchmark in which elastic
net, principal component analysis, connectome-based predictive modelling&nbsp;[8],
per-fold univariate screening and network-level aggregation are all fitted inside the
outer training folds of a repeated nested cross-validation. Second, we separate a
covariate-adjusted exploratory map from every predictive pipeline. Third, we isolate the
leakage effect with a contrast in which the selector, the classifier and the
hyperparameter grid are identical and only the scope of the edge screen changes; this
distinguishes optimism caused by leakage from optimism caused by a different model.
Fourth, we examine head motion, probability calibration and the aggregation of repeated
out-of-fold predictions, which jointly determine how the honest estimate should be
interpreted. Fifth, we repeat the matched contrast within the UCLA CNP schizophrenia
sample under a different preprocessor, and on signal-free synthetic connectomes. UCLA CNP
is balanced by 1:1 nearest-neighbour matching on age, sex and mean FD (50/50) and is
analysed with within-cohort nested cross-validation, not a train-on-COBRE transfer.
Code and split definitions are publicly available.</p>

@@FIG1@@

<h2>2&ensp;Materials and Methods</h2>

<h3>2.1&ensp;Data and preprocessing</h3>
<p>We used the publicly released COBRE resting-state fMRI collection (Figshare
acquisition 4197885), preprocessed with the NIAK pipeline&nbsp;[7,&thinsp;9].
Participants were retained when both the preprocessed BOLD volume and the accompanying
confound time series were available, giving 146 individuals. The distributed phenotypic
file provides age, sex and mean framewise displacement (FD); no symptom inventories were
available, so the analysis concerns diagnostic classification only. All numeric confound
regressors were passed to the Nilearn <span class="nohyph">NiftiLabelsMasker</span>
during parcel-wise signal extraction with standardisation enabled&nbsp;[10].
Parcellation used the Schaefer 2018 atlas with 100 parcels under the seven-network Yeo
solution at 2&nbsp;mm&nbsp;[11,&thinsp;12], resampled to each subject's BOLD grid by
nearest-neighbour interpolation.</p>

<h3>2.2&ensp;Connectivity estimation</h3>
<p>Pearson correlations between all pairs of parcel time series produced one
100&nbsp;&times;&nbsp;100 symmetric matrix per participant. Discarding the diagonal and
taking the upper triangle yields 4,950 undirected edges, which were Fisher
<span style="font-style:italic">z</span>-transformed before modelling.</p>

<h3>2.3&ensp;Exploratory group-level inference</h3>
<p>Two edge-wise procedures were applied to the full cohort for descriptive purposes.
Unadjusted two-sample <span style="font-style:italic">t</span>-tests with
Benjamini&ndash;Hochberg control at
<span style="font-style:italic">q</span>&nbsp;&lt;&nbsp;0.05&nbsp;[13] identified 29
edges. A covariate-adjusted model,
FC<sub>ij</sub>&nbsp;=&nbsp;&beta;<sub>0</sub>&nbsp;+&nbsp;&beta;<sub>1</sub>&middot;diagnosis&nbsp;+&nbsp;&beta;<sub>2</sub>&middot;age&nbsp;+&nbsp;&beta;<sub>3</sub>&middot;sex&nbsp;+&nbsp;&beta;<sub>4</sub>&middot;FD&nbsp;+&nbsp;&epsilon;,
was then fitted per edge and the false discovery rate was controlled on
&beta;<sub>1</sub>. These maps are reported as inference and were excluded from every
leakage-free predictive pipeline.</p>

<h3>2.4&ensp;Feature construction under fold isolation</h3>
<p>Because feature construction rather than the classifier alone drives optimism, five
strategies were compared, each fitted exclusively on outer training indices: an
elastic-net logistic regression over all 4,950 edges; a 30-component PCA; connectome-based
predictive modelling summarising the top and bottom decile of training-fold edge
correlations into two scores&nbsp;[8]; the twenty edges with the largest training-fold
absolute <span style="font-style:italic">t</span> statistic; and the 28 mean
within- and between-network Fisher <span style="font-style:italic">z</span> values
defined by the Yeo seven-network solution. Age, sex and FD were optionally appended as
measured covariates&nbsp;[14]. Penalty paths and component loadings were never estimated
on held-out participants.</p>

<h3>2.5&ensp;Classifiers and nested validation</h3>
<p>Classifiers were implemented in scikit-learn&nbsp;[15] and comprised
<span style="font-style:italic">L</span><sub>2</sub>-penalised logistic regression, an
elastic-net logistic regression with the saga solver, a radial-basis support vector
machine and, for the network features, a random forest. Each was wrapped in a
standardisation pipeline, and hyperparameters were chosen by an inner five-fold grid
search maximising ROC-AUC. The outer loop was a stratified five-fold partition repeated
ten times, giving 50 outer splits. Fold-wise means and standard deviations are reported
as descriptive summaries; because the repeats overlap, these standard deviations are not
confidence intervals and are not used to rank methods against each other. Secondary
analyses that required out-of-fold probabilities&mdash;subject-level aggregation,
calibration, motion residualisation and the matched leakage contrast&mdash;used five
folds repeated three times; the elastic-net estimate was stable between the two settings
(0.770 versus 0.768).</p>

<h3>2.6&ensp;Aggregation of repeated out-of-fold predictions</h3>
<p>Under repeated cross-validation each participant contributes several test
predictions. Concatenating all of them treats repeated observations from the same
individual as independent. We therefore also computed a subject-level estimate in which
the predicted probabilities of each participant are averaged first and a single ROC-AUC
is evaluated over the 146 unique subjects. Both quantities are reported.</p>

<h3>2.7&ensp;Leakage regimes</h3>
<p>Two families of comparison were run. The first contrasts the leakage-free elastic net
with a logistic regression restricted to the five covariate-adjusted edges of
Table&nbsp;2, whose identities come from full-cohort inference while only the classifier
is cross-validated. Because the feature space and the classifier differ, that spread
bounds mixed-protocol optimism and does not attribute it to leakage alone. A
dataset-level selection figure inherited from the accepted submission is not reported
here, because no released script regenerates it.</p>
<p>The second comparison isolates the selection step. The selector (the twenty largest
absolute <span style="font-style:italic">t</span> statistics), the classifier
(<span style="font-style:italic">L</span><sub>2</sub> logistic regression), the covariate
block and the hyperparameter grid are held fixed, and the only difference is whether the
screen is computed within the outer training fold or on the complete cohort including the
test fold. Label permutation used 50 shuffles of the leakage-free protocol with
<span style="font-style:italic">p</span>&nbsp;=&nbsp;(#{AUC<sub>perm</sub>&nbsp;&ge;&nbsp;AUC<sub>obs</sub>}&nbsp;+&nbsp;1)/(n&nbsp;+&nbsp;1);
the smallest attainable value is 1/51&nbsp;&asymp;&nbsp;0.02, which bounds the resolution
of this test.</p>

<h3>2.8&ensp;Motion sensitivity</h3>
<p>Mean FD differs between groups, so three analyses probe its contribution. An
edges-only elastic net omits the covariate block entirely. A residualisation variant
regresses each edge on FD using coefficients estimated on the training fold and applies
the same transformation to the test fold. Finally, five locked stratified 70/30 splits
were evaluated, with all feature construction confined to the training portion.</p>

<h3>2.9&ensp;Calibration and coefficient inspection</h3>
<p>Expected calibration error over ten equal-width bins and the Brier score were computed
from out-of-fold probabilities for the leakage-free models as well as for the leaked
contrast. Coefficients of the leaked reference logistic model were aggregated across
outer folds. Because that model draws on full-cohort edge identities and includes a
group-imbalanced motion covariate, its weights are reported as a description of what the
circular pipeline exploits and not as anatomical evidence.</p>

<h3>2.10&ensp;Matched schizophrenia sample and null control</h3>
<p>The matched contrast of Sect.&nbsp;2.7 was repeated as nested cross-validation inside
the UCLA Consortium for Neuropsychiatric Phenomics rest sample (OpenNeuro
ds000030)&nbsp;[16]: 50 participants with schizophrenia who had an fMRIPrep MNI rest
series, each paired with one control by greedy nearest-neighbour matching on
standardised age, sex and mean FD (50/50). Time series used Schaefer-100 (4,950 edges);
numeric confound columns entered the masker; the same three covariates were appended.
This is within-CNP nested cross-validation under a different preprocessor, not a locked
transfer from COBRE, and it does not revise 0.77.</p>
<p>The second extra experiment removes the signal. Labels are random and features are
either independent Gaussians or connectomes built from random parcel time series with a
five-component latent structure (mean absolute correlation between edge pairs 0.10 at
the COBRE dimensions). For the CNP subsample we also permute the observed labels on the
real matched connectomes, which preserves the empirical edge dependence. Ten draws were
taken per configuration, including
<span style="font-style:italic">k</span>&nbsp;&isin;&nbsp;{5,&thinsp;20,&thinsp;100} and
<span style="font-style:italic">n</span> between 50 and 600 at
<span style="font-style:italic">p</span>&nbsp;=&nbsp;4,950. A leakage-free protocol must
return chance; whatever the leaked protocol returns is the level its own design can
manufacture with no signal.</p>

<h3>2.11&ensp;Implementation</h3>
<p>Analyses were run in Python&nbsp;3.11 with Nilearn, NiBabel, SciPy, statsmodels and
scikit-learn&nbsp;[10,&thinsp;15]. Connectivity matrices, edge statistics, benchmark
tables and calibration summaries are exported as CSV. The audit protocol, the split
definitions and the analysis scripts are available at
<span class="nohyph">https://github.com/Ishsirjan/Leakage_Audit</span>; the imaging data
remain at the original Figshare and OpenNeuro records.</p>

<h2>3&ensp;Results</h2>

<h3>3.1&ensp;Sample and exploratory connectivity</h3>
<p>Table&nbsp;1 summarises the analysed sample. Patients moved more than controls during
acquisition (Welch <span style="font-style:italic">t</span>&nbsp;=&nbsp;4.33,
<span style="font-style:italic">p</span>&nbsp;=&nbsp;3.3&nbsp;&times;&nbsp;10<sup>&minus;5</sup>).
Unadjusted testing returned 29 FDR-significant edges; after adjustment for age, sex and
FD only five survived (Table&nbsp;2), a contraction consistent with the known influence of
micro-movements on connectivity estimates&nbsp;[14]. The surviving connections couple
visual, salience/ventral-attention, default-mode and control systems (Fig.&nbsp;2), in
line with large-scale network accounts of psychosis&nbsp;[4]. They are reported as
exploratory inference and were withheld from the leakage-free pipelines.</p>

@@TABLE1@@
@@TABLE2@@
@@FIG2@@

<h3>3.2&ensp;Leakage-free classification</h3>
<p>When no held-out information reaches feature construction, whole-connectome sparse and
component-based representations perform best (Table&nbsp;3, Fig.&nbsp;3). PCA-30 with
covariates reached 0.771&nbsp;&plusmn;&nbsp;0.079 and the elastic net
0.768&nbsp;&plusmn;&nbsp;0.076. Their fold distributions overlap almost completely, so we
treat them as equivalent and designate the elastic net as the primary model because it
returns sparse edge weights. Removing the covariate block changes little
(0.765&nbsp;&plusmn;&nbsp;0.079), which indicates that the imaging features rather than
the appended demographics carry the discrimination. Hard univariate screening
(0.705&nbsp;&plusmn;&nbsp;0.071) and network averaging (0.679&nbsp;&plusmn;&nbsp;0.092)
performed clearly worse. Permutation of the labels placed the null mean near 0.49 with an
empirical <span style="font-style:italic">p</span>&nbsp;&lt;&nbsp;0.02 at the resolution
of 50 shuffles.</p>

@@TABLE3@@
@@FIG3@@

<h3>3.3&ensp;Isolating the leakage effect</h3>
<p>Relaxing fold isolation increases apparent performance (Fig.&nbsp;4a). Freezing the
five covariate-adjusted edges lifts the AUC to 0.855&nbsp;&plusmn;&nbsp;0.066 even though
the classifier itself is cross-validated. That setting differs from the primary model in
feature space and classifier as well as in selection scope, so the 0.087 spread is a
mixed-protocol bound rather than leakage in isolation.</p>
<p>The matched contrast removes that ambiguity (Fig.&nbsp;4b). With the same twenty-edge
screen, the same logistic classifier and the same grid, computing the screen inside the
training fold yields 0.709&nbsp;&plusmn;&nbsp;0.076, whereas computing it on the complete
cohort yields 0.903&nbsp;&plusmn;&nbsp;0.039. The difference of 0.194 AUC is the cost of
leaking this twenty-edge univariate screen, not the leakage cost of the primary elastic
net, which already shrinks inside the fold and for which we do not report a matched
leaked counterpart. The leakage-free arm here uses three repeats and matches the
ten-repeat value of Table&nbsp;3 (0.705) to within 0.004.</p>

@@FIG4@@

<h3>3.4&ensp;Contribution of head motion</h3>
<p>Motion is not eliminated by omitting FD from the design matrix. Residualising every
edge on FD within the fold lowers the edges-only elastic net from
0.770&nbsp;&plusmn;&nbsp;0.072 to 0.694&nbsp;&plusmn;&nbsp;0.104 and the edges-only
twenty-edge screen, which excludes the covariate block and therefore starts from 0.686
rather than the 0.709 of Sect.&nbsp;3.3, to 0.647 (Fig.&nbsp;5a). Locked 70/30 splits, in which feature
construction never sees the evaluation set, gave a mean AUC of 0.733 for the elastic net
and 0.712 for the univariate screen (Fig.&nbsp;5b), consistent with the repeated nested
estimate. Part of the honest signal therefore tracks diagnosis-related movement, and
estimates obtained without motion control should be read accordingly.</p>

@@FIG5@@

<h3>3.5&ensp;Calibration and aggregation of repeated predictions</h3>
<p>For the leakage-free elastic net the expected calibration error was 0.116 with a Brier
score of 0.220 (Fig.&nbsp;6a, Table&nbsp;4); the model discriminates above chance but its
probabilities would require recalibration before any decision-support use. The leaked arm
of the matched contrast looks better on both counts, with an expected calibration error
of 0.064 and a Brier score of 0.136. That improvement is itself an artefact: the leaked
probabilities are fitted to edges chosen with knowledge of the very labels against which
they are then scored, so calibration diagnostics inherit the leak and cannot be used to
detect it. Good calibration is therefore not evidence that a pipeline is sound.
Averaging the repeated out-of-fold probabilities within participants before computing a
single AUC gave 0.764 for the primary model against 0.749 when all repeated predictions
were pooled as if independent (Fig.&nbsp;6b). The two aggregation rules differ by between
0.014 and 0.026 AUC across the regimes examined, so the choice does not alter the
conclusions, but the subject-level value is the one we report.</p>

@@TABLE4@@
@@FIG6@@

<h3>3.6&ensp;What the leaked reference model uses</h3>
<p>Aggregated across outer folds, the largest weight of the leaked reference model is
mean FD (0.174&nbsp;&plusmn;&nbsp;0.125), followed by the left visual to right
default-mode temporal edge (0.169&nbsp;&plusmn;&nbsp;0.103); all five covariate-adjusted
edges recur among the leading contributors (Fig.&nbsp;7). A motion summary
outranking every connectivity feature in a model whose edges were selected on the full
cohort is a description of the circularity, not a biological result, and we draw no
anatomical inference from it.</p>

@@FIG7@@

<h3>3.7&ensp;Matched schizophrenia sample and null control</h3>
<p>On a 50/50 age-, sex- and FD-matched UCLA CNP subsample the same twenty-edge
construction yields 0.679&nbsp;&plusmn;&nbsp;0.074 inside the training fold and
0.873&nbsp;&plusmn;&nbsp;0.081 on the full subsample (Fig.&nbsp;8b), a gap of 0.194
(five folds, three repeats). A leakage-free logistic model over all 4,950 edges reached
0.727. Accuracy is now interpretable (chance 0.50; leakage-free twenty-edge accuracy
0.617). fMRIPrep and the absence of a locked transfer mean these numbers do not replace
0.77; they show that the 0.194 inflation is not an artefact of COBRE's class balance.</p>
<p>The null control is the more informative extra experiment. On data with no signal the
leakage-free protocol returns chance at the COBRE dimensions
(0.505&nbsp;&plusmn;&nbsp;0.068 independent; 0.516&nbsp;&plusmn;&nbsp;0.064 correlated),
which is what an unbiased estimator must do and which validates the primary design. A
leaked twenty-edge screen on the same data reports 0.821&nbsp;&plusmn;&nbsp;0.020 when
edges are correlated as in a connectome and 0.938&nbsp;&plusmn;&nbsp;0.017 when they are
independent. The leaked COBRE value of 0.903 therefore sits about 0.08 above the
correlated floor and 0.03 below the independent ceiling (Fig.&nbsp;8b): leakage accounts
for most of that figure, but the correlated null does not swallow it entirely.
Independent features are an upper bound on manufacturable optimism, not the reference
for a real connectome. On the balanced CNP connectomes a leaked twenty-edge screen
reports 0.873 on synthetic correlated nulls, 0.893 when the observed labels are permuted,
and 0.971 on independent features. The observed leaked value of 0.873 therefore does not
exceed the permutation floor: leakage accounts for that figure, while the leakage-free
arm (0.679) remains above chance. Retaining more edges makes leakage worse: with independent
features the leaked AUC rises from 0.809 at
<span style="font-style:italic">k</span>&nbsp;=&nbsp;5 to 0.937 at 20 and 0.998 at 100
(Fig.&nbsp;8a). Optimism on null data falls with sample size, from 0.501 at
<span style="font-style:italic">n</span>&nbsp;=&nbsp;50 to 0.263 at 600
(Fig.&nbsp;8c).</p>

@@FIG8@@

<h2>4&ensp;Discussion</h2>
<p class="first">The central observation of this audit is that the same 146 scans support
AUCs between 0.71 and 0.90 depending only on when a twenty-edge univariate screen is
computed. That 0.194 gap is the price of leaking this construction. It is not the
leakage cost of the primary elastic net, which already shrinks inside the fold. Freezing the five FDR edges raises the AUC
to 0.855, but that comparison also changes the feature space, so it is not an isolated
leakage effect.</p>

<p>The null control sharpens that reading. The relevant reference is a leaked
twenty-edge screen run on signal-free connectomes with realistic edge dependence, which
reports 0.821 at the COBRE dimensions. The leaked COBRE value of 0.903 exceeds that
floor by about 0.08, so most of the leaked figure is a property of the design, but not
all of it. Independent Gaussian features raise the same screen to 0.938 and should be
read as an upper bound, not as evidence that 0.903 is indistinguishable from noise. A
leaked figure should never be compared against chance; it should be compared against
the same pipeline on label-shuffled or synthetic data of comparable dependence. The
leakage-free arm returning chance on those data is what licenses reading 0.768 as
signal.</p>

<p>The UCLA CNP contrast, after 1:1 matching on age, sex and FD, reproduces the same
0.194 gap (0.679 vs 0.873) and does not license reading 0.679 as a replication of 0.77.</p>

<p>The honest estimate itself should be scoped carefully. Roughly 0.77 AUC applies to
this cohort, this parcellation, Pearson connectivity and this family of classical linear
and sparse models; it is not a ceiling for schizophrenia decoding in general, and larger
samples, different connectivity estimators or representation learning may well exceed it.
Published COBRE results between 0.75 and 0.90&nbsp;[1,&thinsp;2] are best compared only
after each study's selection protocol is known.</p>

<p>Feature construction mattered more than the choice of classifier. Regularised models
over the full connectome and low-dimensional projections behaved similarly, while hard
thresholding of univariate statistics lost about six AUC points. With about 117
training participants per split the ranking of individual edges is unstable, so methods
that accumulate many weak effects outperform those that discard them&nbsp;[6]. Since the
fold distributions of the leading leakage-free methods overlap, we make no claim that any
one of them is superior.</p>

<p>Motion deserves explicit treatment rather than a footnote. Mean FD differs between
groups, dominates the weights of the circular model, and its fold-wise removal costs
about 0.08 AUC. The edges-only model still discriminates, so the result is not reducible
to movement, but a portion of the honest performance is motion-related and this should be
stated whenever connectome classifiers are compared across cohorts with different
acquisition quality.</p>

<p>Several limitations remain. UCLA CNP is a within-cohort fMRIPrep audit on a 50/50
matched subsample, not a locked transfer, and has no fold-wise FD residualisation of
edges. The correlated synthetic null is a five-component latent structure; the tighter
CNP control is a label permutation of the observed matched connectomes. Connectivity is
Pearson correlation, the model family is classical, the permutation test on COBRE
resolves probabilities only to about 0.02, and secondary analyses use three rather than
ten repeats. No motion-matched COBRE subsample was constructed.</p>

<h2>5&ensp;Conclusion</h2>
<p class="first">We measured how well resting-state connectomes classify schizophrenia in
COBRE when feature construction is denied access to test labels, and how much that
estimate inflates when access is granted. Under repeated nested cross-validation the
honest AUC is 0.768&nbsp;&plusmn;&nbsp;0.076, while a twenty-edge screen that sees the
full cohort reports 0.903, about 0.08 above the same screen on signal-free connectomes.
Four practices follow for connectome classification: validate the feature-construction
step with the same rigour as the classifier; report leakage as a matched contrast in
which only the selection scope changes; compare any leaked figure with a synthetic or
permuted null of comparable feature dependence, not with chance; and keep exploratory
group maps separate from predictive features. The protocol
and all scripts are available at
<span class="nohyph">https://github.com/Ishsirjan/Leakage_Audit</span>.</p>

<h2 style="margin-top:14pt">Disclosure of Interests</h2>
<p class="first">The author has no competing interests to declare that are relevant to the
content of this article.</p>

<h2>References</h2>
<ol class="refs">
<li>Arbabshirani, M.R., Plis, S., Sui, J., Calhoun, V.D.: Single subject prediction of brain disorders in neuroimaging: promises and pitfalls. NeuroImage <b>145</b>, 137&ndash;165 (2017)</li>
<li>Wolfers, T., Buitelaar, J.K., Beckmann, C.F., Franke, B., Marquand, A.F.: From estimating activation locality to predicting disorder: a review of pattern recognition for neuroimaging-based psychiatric diagnostics. Neurosci. Biobehav. Rev. <b>57</b>, 328&ndash;349 (2015)</li>
<li>Fornito, A., Zalesky, A., Breakspear, M.: The connectomics of brain disorders. Nat. Rev. Neurosci. <b>16</b>(3), 159&ndash;172 (2015)</li>
<li>Menon, V.: Large-scale brain networks and psychopathology: a unifying triple network model. Trends Cogn. Sci. <b>15</b>(10), 483&ndash;506 (2011)</li>
<li>Woodward, N.D., Cascio, C.J.: Resting-state functional connectivity in psychiatric disorders. JAMA Psychiatry <b>72</b>(8), 743&ndash;744 (2015)</li>
<li>Varoquaux, G., Raamana, P.R., Engemann, D.A., Hoyos-Idrobo, A., Schwartz, Y., Thirion, B.: Assessing and tuning brain decoders: cross-validation, caveats, and guidelines. NeuroImage <b>145</b>, 166&ndash;179 (2017)</li>
<li>Aine, C.J., et al.: Multimodal neuroimaging in schizophrenia: description and dissemination. Neuroinformatics <b>15</b>(4), 343&ndash;364 (2017)</li>
<li>Shen, X., Finn, E.S., Scheinost, D., Rosenberg, M.D., Chun, M.M., Papademetris, X., Constable, R.T.: Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nat. Protoc. <b>12</b>(3), 506&ndash;518 (2017)</li>
<li>Bellec, P., Lavoie-Courchesne, S., Dickinson, P., Lerch, J.P., Zijdenbos, A.P., Evans, A.C.: The pipeline system for Octave and Matlab (PSOM). Front. Neuroinform. <b>6</b>, 7 (2012)</li>
<li>Abraham, A., Pedregosa, F., Eickenberg, M., Gervais, P., Mueller, A., Kossaifi, J., Gramfort, A., Thirion, B., Varoquaux, G.: Machine learning for neuroimaging with scikit-learn. Front. Neuroinform. <b>8</b>, 14 (2014)</li>
<li>Schaefer, A., Kong, R., Gordon, E.M., Laumann, T.O., Zuo, X.N., Holmes, A.J., Eickhoff, S.B., Yeo, B.T.T.: Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. Cereb. Cortex <b>28</b>(9), 3095&ndash;3114 (2018)</li>
<li>Yeo, B.T.T., et al.: The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J. Neurophysiol. <b>106</b>(3), 1125&ndash;1165 (2011)</li>
<li>Benjamini, Y., Hochberg, Y.: Controlling the false discovery rate: a practical and powerful approach to multiple testing. J. R. Stat. Soc. B <b>57</b>(1), 289&ndash;300 (1995)</li>
<li>Power, J.D., Barnes, K.A., Snyder, A.Z., Schlaggar, B.L., Petersen, S.E.: Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage <b>59</b>(3), 2142&ndash;2154 (2012)</li>
<li>Pedregosa, F., et al.: Scikit-learn: machine learning in Python. J. Mach. Learn. Res. <b>12</b>, 2825&ndash;2830 (2011)</li>
<li>Poldrack, R.A., et al.: A phenome-wide examination of neural and cognitive function. Sci. Data <b>3</b>, 160110 (2016)</li>
</ol>
"""

TABLE1 = """
<table>
<caption><span class="lab">Table 1.</span> Characteristics of the 146 analysed COBRE
participants. FD denotes mean framewise displacement in millimetres.</caption>
<thead><tr><th>Group</th><th class="n">N</th><th class="n">Age (years)</th>
<th class="n">Male, n (%)</th><th class="n">Mean FD</th></tr></thead>
<tbody>
<tr><td>Schizophrenia</td><td class="n">72</td><td class="n">38.2 &plusmn; 13.9</td><td class="n">58 (80.6)</td><td class="n">0.478 &plusmn; 0.274</td></tr>
<tr><td>Healthy control</td><td class="n">74</td><td class="n">35.8 &plusmn; 11.6</td><td class="n">51 (68.9)</td><td class="n">0.318 &plusmn; 0.152</td></tr>
<tr class="rule"><td>Total</td><td class="n">146</td><td class="n">37.0 &plusmn; 12.8</td><td class="n">109 (74.7)</td><td class="n">0.397 &plusmn; 0.236</td></tr>
</tbody>
</table>
"""

TABLE2 = """
<table>
<caption><span class="lab">Table 2.</span> Edges surviving false-discovery-rate control
after adjustment for age, sex and mean framewise displacement. These edges are
exploratory and were not supplied to any leakage-free predictive pipeline.</caption>
<thead><tr><th>#</th><th>Parcel 1</th><th>Parcel 2</th>
<th class="n">t<sub>adj</sub></th><th class="n">q<sub>FDR</sub></th></tr></thead>
<tbody>
<tr><td>1</td><td>LH Vis 3</td><td>RH Default Temp 2</td><td class="n">4.38</td><td class="n">0.031</td></tr>
<tr><td>2</td><td>LH SalVentAttn FrOperIns 1</td><td>RH Default PFCv 1</td><td class="n">&minus;4.93</td><td class="n">0.011</td></tr>
<tr><td>3</td><td>LH Default Par 2</td><td>RH Cont PFCl 1</td><td class="n">&minus;4.30</td><td class="n">0.031</td></tr>
<tr><td>4</td><td>RH Vis 8</td><td>RH Default Temp 2</td><td class="n">&minus;4.33</td><td class="n">0.031</td></tr>
<tr><td>5</td><td>RH SalVentAttn TempOccPar 2</td><td>RH Default PFCv 1</td><td class="n">&minus;4.65</td><td class="n">0.019</td></tr>
</tbody>
</table>
"""

TABLE3 = """
<table>
<caption><span class="lab">Table 3.</span> Leakage-free benchmark over 50 outer splits
(five folds, ten repeats). AUC and accuracy are fold means &plusmn; standard deviations;
the last column pools all out-of-fold predictions. Standard deviations describe the
spread across overlapping repeats and are not confidence intervals.</caption>
<thead><tr><th>Feature construction</th><th>Classifier</th><th class="n">AUC</th>
<th class="n">Accuracy</th><th class="n">Pooled OOF AUC</th></tr></thead>
<tbody>
<tr><td>PCA-30 + covariates</td><td>Logistic</td><td class="n">0.771 &plusmn; 0.079</td><td class="n">0.713 &plusmn; 0.074</td><td class="n">0.751</td></tr>
<tr><td>Elastic net, 4,950 edges + covariates</td><td>Elastic net</td><td class="n">0.768 &plusmn; 0.076</td><td class="n">0.687 &plusmn; 0.076</td><td class="n">0.752</td></tr>
<tr><td>Elastic net, edges only</td><td>Elastic net</td><td class="n">0.765 &plusmn; 0.079</td><td class="n">0.682 &plusmn; 0.076</td><td class="n">0.752</td></tr>
<tr><td>CPM, top/bottom decile + covariates</td><td>Logistic</td><td class="n">0.742 &plusmn; 0.091</td><td class="n">0.680 &plusmn; 0.083</td><td class="n">0.733</td></tr>
<tr><td>Top-20 |t| within fold + covariates</td><td>Logistic</td><td class="n">0.705 &plusmn; 0.071</td><td class="n">0.641 &plusmn; 0.074</td><td class="n">0.693</td></tr>
<tr><td>Network-28 + covariates</td><td>Logistic</td><td class="n">0.679 &plusmn; 0.092</td><td class="n">0.644 &plusmn; 0.079</td><td class="n">0.670</td></tr>
</tbody>
</table>
"""

TABLE4 = """
<table>
<caption><span class="lab">Table 4.</span> Aggregation of repeated out-of-fold
predictions and calibration. Pooled AUC concatenates every fold prediction;
subject-level AUC averages the repeated probabilities of each participant first. ECE is
the expected calibration error over ten equal-width bins. The leaked row is better
calibrated than the leakage-free rows, which shows that calibration cannot diagnose
leakage.</caption>
<thead><tr><th>Model</th><th class="n">Pooled AUC</th><th class="n">Subject-level AUC</th>
<th class="n">ECE</th><th class="n">Brier</th></tr></thead>
<tbody>
<tr><td>Elastic net + covariates</td><td class="n">0.749</td><td class="n">0.764</td><td class="n">0.116</td><td class="n">0.220</td></tr>
<tr><td>Elastic net, edges only</td><td class="n">0.749</td><td class="n">0.764</td><td class="n">0.118</td><td class="n">0.220</td></tr>
<tr><td>Top-20 |t| inside the fold</td><td class="n">0.705</td><td class="n">0.731</td><td class="n">0.080</td><td class="n">0.221</td></tr>
<tr><td>Top-20 |t| on the full cohort</td><td class="n">0.884</td><td class="n">0.904</td><td class="n">0.064</td><td class="n">0.136</td></tr>
</tbody>
</table>
"""


def main():
    # Placeholders are delimited because base64 payloads can contain bare
    # tokens such as "FIG3", which would otherwise be substituted inside an
    # already-inserted image and corrupt it.
    body = BODY
    for token, block in (("TABLE1", TABLE1), ("TABLE2", TABLE2), ("TABLE3", TABLE3),
                         ("TABLE4", TABLE4)):
        assert f"@@{token}@@" in body, token
        body = body.replace(f"@@{token}@@", block)

    figures = {
        "FIG1": figure(
            "fig1_design.png", 1,
            "Design of the audit on COBRE. Group-level inference and predictive modelling are kept "
            "separate. Every leakage-free pipeline fits its feature construction inside the "
            "outer training folds. Two comparisons then relax that constraint: a mixed "
            "protocol in which the feature space also changes, and a matched contrast in which "
            "only the scope of the edge screen changes.", "100%"),
        "FIG2": figure(
            "fig2_connectome.png", 2,
            "Exploratory connectome of the five edges surviving covariate-adjusted FDR "
            "control, shown on Schaefer-100 centroids (left, sagittal and axial glass-brain "
            "projections) and as a chord diagram (right). The left sagittal view shows no "
            "edges because none of the five surviving connections is confined to the left "
            "hemisphere; three cross the midline (visible on the axial view) and two are "
            "right&ndash;right (visible on the right sagittal view). Warm edges indicate higher "
            "connectivity in patients, cool edges lower; line width scales with the "
            "adjusted t statistic. Node colour encodes the Yeo seven-network assignment.",
            "100%"),
        "FIG3": figure(
            "fig3_methods.png", 3,
            "Feature-construction strategies under nested cross-validation. Markers give "
            "the fold mean, whiskers one standard deviation across the 50 outer splits. "
            "Open diamonds mark the two settings whose edge identities were fixed on the "
            "full cohort and which are therefore not leakage-free.", "84%"),
        "FIG4": figure(
            "fig4_leakage.png", 4,
            "(a) Mixed protocol: freezing the five full-cohort FDR edges raises apparent "
            "performance by 0.087 AUC, but the two settings also differ in feature space and "
            "classifier. "
            "(b) Matched contrast: identical top-20 screen, classifier and grid, differing "
            "only in whether the screen sees the test fold. The 0.194 AUC gap is the cost "
            "of leaking this univariate screen, not of leaking the primary elastic net.", "88%"),
        "FIG5": figure(
            "fig5_motion.png", 5,
            "(a) Removing mean framewise displacement from every edge within the fold "
            "lowers performance for both the elastic net and univariate screening, "
            "indicating that part of the honest signal is motion-related. (b) Five locked "
            "70/30 splits with all construction confined to the training portion; dashed "
            "lines give the means.", "100%"),
        "FIG6": figure(
            "fig7_calibration.png", 6,
            "(a) Calibration of the leakage-free models and of the leaked contrast. A "
            "leaked pipeline can be both more discriminative and better calibrated than "
            "an honest one, so neither metric detects the leak. (b) Pooling every repeated "
            "out-of-fold prediction as independent versus averaging the repeated "
            "probabilities within each participant before computing a single AUC.", "92%"),
        "FIG7": figure(
            "fig6_coefficients.png", 7,
            "Largest averaged coefficients of the leaked reference logistic model. Mean "
            "framewise displacement carries the largest weight, which characterises the "
            "circular pipeline rather than the neurobiology of schizophrenia.", "76%"),
        "FIG8": figure(
            "fig8_external.png", 8,
            "(a) On data with no signal, a leaked twenty-edge screen reports more the more "
            "edges it retains, while the leakage-free protocol stays at chance. "
            "(b) Observed estimates against the band a leaked twenty-edge screen reaches "
            "on signal-free data of the same dimensions. The lower edge of each band uses "
            "correlated connectome edges and is the relevant floor; the upper edge uses "
            "independent features and is an upper bound. COBRE leaked (0.903) sits about "
            "0.08 above the correlated floor. UCLA CNP is a 50/50 age-, sex- and FD-matched "
            "within-cohort audit under fMRIPrep, not a locked transfer; its leaked 0.873 sits "
            "at the correlated floor and does not exceed a label-permutation floor of 0.893. "
            "(c) Optimism against sample size on independent-feature null data.", "100%"),
    }
    for token, block in figures.items():
        assert f"@@{token}@@" in body, token
        body = body.replace(f"@@{token}@@", block)

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>Leakage-Controlled Resting-State Connectome Classification of Schizophrenia</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n<div class=\"screen-pad\">\n"
        f"{body}\n</div>\n</body>\n</html>\n"
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
