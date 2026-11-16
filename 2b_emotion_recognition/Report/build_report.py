"""Builds Report/Report_2b.pdf (HTML -> PDF with Microsoft Edge headless) from the files in Results/.

    python Report/build_report.py --repo-url https://github.com/<user>/<repo>
"""
import argparse
import csv
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
RES = os.path.join(ROOT, "Results")


def img(rel, cap, w="100%"):
    return '<figure><img src="file:///%s" style="width:%s"><figcaption>%s</figcaption></figure>' % (os.path.join(RES, rel).replace("\\", "/"), w, cap)


CSS = """
@page { size: A4; margin: 14mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #1b1b1b; }
h1 { font-size: 17pt; margin: 0 0 2pt; } .sub { color: #555; margin-bottom: 7pt; font-size: 9pt; }
h2 { font-size: 11.5pt; margin: 11pt 0 3pt; border-bottom: 1.5px solid #2b4c7e; color: #2b4c7e; page-break-after: avoid; }
h3 { font-size: 10pt; margin: 7pt 0 2pt; color: #2b4c7e; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 6pt; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: left; vertical-align: top; } th { background: #eceff3; }
figure { margin: 4pt 0 6pt; page-break-inside: avoid; } figcaption { font-size: 8pt; color: #444; }
ul { margin: 2pt 0 4pt 15pt; padding: 0; } li { margin: 1.5pt 0; } code { font-family: Consolas, monospace; font-size: 8.3pt; background: #f1f1f1; padding: 0 2px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 6pt; }
.note { background: #fff7e0; border-left: 3px solid #e0a800; padding: 3pt 6pt; margin: 4pt 0; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/&lt;user&gt;/&lt;repo&gt;")
    a = ap.parse_args()
    acc = {(r["modality"], r["protocol"]): r for r in csv.DictReader(open(os.path.join(RES, "accuracy_summary.csv"), encoding="utf-8"))}
    noise = {(r["model"], r["snr_db"]): r for r in csv.DictReader(open(os.path.join(RES, "tables", "noise_robustness.csv"), encoding="utf-8"))}
    sep = {p: {r["block"]: r for r in json.load(open(os.path.join(RES, "representations", "separability_%s_seed0.json" % p)))} for p in ("word", "spk_OAF2YAF", "spk_YAF2OAF")}

    def A(m, p):
        r = acc[(m, p)]
        return "%.1f &plusmn; %.1f" % (float(r["accuracy_mean_%"]), float(r["accuracy_std_%"]))

    def F1(m, p):
        r = acc[(m, p)]
        return "%.3f" % float(r["macro_f1_mean"])

    def N(m, s):
        r = noise[(m, s)]
        return "%.1f &plusmn; %.1f" % (float(r["accuracy_mean_%"]), float(r["accuracy_std_%"]))

    def S(p, b):
        r = sep[p][b]
        return "%.2f / %.2f" % (r["silhouette_emotion"], r["knn5_emotion_acc"])

    p = []
    P = p.append
    P("<h1>Assignment 2b: Multimodal emotion recognition on TESS</h1>")
    P('<div class="sub">Speech-only, text-only and fusion pipelines &nbsp;|&nbsp; Code, results and plots: %s</div>' % a.repo_url)

    P("<h2>0. Setting, and one fact that shapes every result</h2>")
    P("<p><b>Data.</b> TESS: 2 actresses (OAF, older; YAF, younger) &times; 7 emotions (angry, disgust, fear, happy, neutral, pleasant surprise, sad) &times; 200 target words = 2,800 clips. "
      "Every clip is the carrier phrase <i>&ldquo;Say the word ___&rdquo;</i>; the text input is therefore this sentence with the target word filled in.</p>")
    P('<div class="note"><b>The text carries no emotion information</b>: each of the 200 sentences occurs once per emotion and speaker, so the text is identical across the 7 classes. '
      "Any text-only model can reach at most chance (14.3%). I report this as a finding instead of hiding it: the fusion experiments below test whether a modality that is <i>uninformative for the label</i> can hurt or help, and the text pipeline is a sanity check of the whole code path.</div>")
    P("<p><b>Evaluation protocols</b> (this matters more than any architecture choice):</p><ul>"
      "<li><b>word</b> (main): the 200 words are split 70/10/20 into train/validation/test, so no word is shared between sets (test = 40 words &times; 14 = 560 clips, both speakers present in every set); 5 seeds, each with a different word split.</li>"
      "<li><b>spk_OAF2YAF / spk_YAF2OAF</b> (harder, more realistic): train and validate on one speaker, test on <i>all</i> 1,400 clips of the other speaker; 3 seeds.</li>"
      "<li><b>random</b> (diagnostic only, leaky): clip-level split in which the same word occurs in train and test.</li></ul>")

    P("<h2>A. Architecture decisions, block by block</h2>")
    P("<table><tr><th>Block</th><th>Speech pipeline</th><th>Text pipeline</th><th>Why</th></tr>"
      "<tr><td><b>1. Preprocessing</b></td><td>16 kHz mono, silence trimming (30 dB), peak normalisation; <b>SpecAugment</b> (random time/frequency masks) in training</td><td>Carrier sentence with the target word, lower-cased, DistilBERT tokenizer</td>"
      "<td>Trimming removes the recording-specific lead-in so that duration cues are the speaker's own; SpecAugment is the main regulariser for 1,960 training clips. Peak normalisation removes gain differences between sessions but keeps within-clip dynamics.</td></tr>"
      "<tr><td><b>2. Feature extraction</b></td><td>80-bin log-mel (25 ms window, 10 ms hop, 20&ndash;8000 Hz), standardised with <i>training-set</i> statistics</td><td><b>Frozen</b> DistilBERT token states (768-d), cached</td>"
      "<td>Log-mel keeps pitch harmonics and spectral tilt, which carry the emotion cues, and lets the network learn its own representation (no hand-picked eGeMAPS/MFCC loss of detail). Frozen text features: with 200 distinct sentences, fine-tuning a 66 M-parameter model would only memorise words.</td></tr>"
      "<tr><td><b>3. Temporal / contextual modelling</b></td><td><b>Temporal</b>: 2 &times; Conv1d(k=5)+BN, max-pool &times;2, 2-layer <b>BiGRU</b>(128), attention pooling &rarr; 256-d</td><td><b>Contextual</b>: linear 768&rarr;256, 1-layer BiGRU(128), attention pooling &rarr; 256-d</td>"
      "<td>The CNN captures local spectro-temporal patterns (a few frames), the BiGRU the utterance-level prosodic contour (rise/fall, rhythm), attention pooling lets the model weight the voiced word rather than the carrier phrase. A BiGRU rather than a transformer: 2,800 clips are too few for attention-only encoders. 0.73 M parameters (speech), 0.59 M (text).</td></tr>"
      "<tr><td><b>4. Fusion</b></td><td colspan=\"2\">Three variants of the same encoders: <b>concat</b> (concatenate the two 256-d vectors &rarr; Linear+ReLU), <b>gated</b> (a learned sigmoid gate mixes speech with a projected text vector), <b>cross-attention</b> (speech frames attend to the text tokens, residual + LayerNorm, attention pooling)</td>"
      "<td>Fusion from simple to expressive. Late/utterance-level fusion is the natural choice with one word per clip; the gate can learn to ignore an unreliable modality, cross-attention can align words with frames. 1.42 M / 1.48 M / 1.68 M parameters.</td></tr>"
      "<tr><td><b>5. Classifier</b></td><td colspan=\"2\">Dropout &rarr; Linear(128) &rarr; ReLU &rarr; Dropout &rarr; Linear(7), softmax; AdamW (lr 1e-3, text 2e-3, wd 1e-2), cosine schedule with warm-up, label smoothing 0.05, batch 32, 30 epochs, model selection on validation accuracy</td>"
      "<td>Small head, strong dropout and label smoothing because the data are small and very clean (over-confidence is the main risk; see the failure cases).</td></tr></table>")

    P("<h2>B. Experiments: speech vs. text vs. multimodal</h2>")
    P("<table><tr><th>Model</th><th colspan=\"2\">word-disjoint (5 seeds)</th><th colspan=\"2\">OAF &rarr; YAF (3 seeds)</th><th colspan=\"2\">YAF &rarr; OAF (3 seeds)</th><th colspan=\"2\">random, leaky (3 seeds)</th></tr>"
      "<tr><th></th><th>Acc. %</th><th>Macro-F1</th><th>Acc. %</th><th>Macro-F1</th><th>Acc. %</th><th>Macro-F1</th><th>Acc. %</th><th>Macro-F1</th></tr>")
    rowspec = [("Speech only", "speech", "word", "spk_OAF2YAF", "spk_YAF2OAF", "random"),
               ("Text only", "text", "word", "spk_OAF2YAF", "spk_YAF2OAF", "random"),
               ("Fusion (concat)", "fusion", "word_concat", "spk_OAF2YAF_concat", "spk_YAF2OAF_concat", "random_concat")]
    for name, m, *prots in rowspec:
        P("<tr><td><b>%s</b></td>%s</tr>" % (name, "".join("<td>%s</td><td>%s</td>" % (A(m, q), F1(m, q)) for q in prots)))
    P("<tr><td>Fusion (gated)</td><td>%s</td><td>%s</td><td colspan=\"6\">only run on the word-disjoint protocol</td></tr>" % (A("fusion", "word_gated"), F1("fusion", "word_gated")))
    P("<tr><td>Fusion (cross-attention)</td><td>%s</td><td>%s</td><td colspan=\"6\">only run on the word-disjoint protocol</td></tr></table>" % (A("fusion", "word_cross"), F1("fusion", "word_cross")))
    P("<ul>"
      "<li><b>On the word-disjoint protocol everything that sees the audio is at 99.8&ndash;99.9%</b>: the three fusion variants, speech-only and the leaky split are indistinguishable (differences &le; 0.1 point, inside one standard deviation). "
      "That is an honest ceiling effect, not a sign that the problem is solved: both speakers are in training, and TESS is acted, studio-quality and very consistent. Text-only is at chance (14.3%) in every protocol, as the carrier phrase predicts; its macro-F1 (0.04) shows that it predicts essentially one class.</li>"
      "<li><b>Unseen-speaker accuracy collapses to 40&ndash;52%</b>: the same speech model that is at 99.8% falls to 48.6% (OAF&rarr;YAF) and 40.2% (YAF&rarr;OAF). With two speakers, a model can learn each speaker's personal way of acting an emotion; this, not the architecture, is the real difficulty of TESS and the reason I report both protocols.</li>"
      "<li>Fusion is +3.3 and +3.8 points above speech-only on the two cross-speaker directions, but the seed standard deviation is 5&ndash;8 points with 3 seeds, so this is <b>not a significant improvement</b>. Since the text alone is at chance, any such gain cannot come from emotional content in the words (see C2).</li></ul>")
    P(img("plots/confusion_speech_spk_OAF2YAF.png", "Speech-only, OAF &rarr; YAF (mean over 3 seeds)", "100%") .replace('<figure>', '<figure style="display:inline-block;width:49%">'))
    P(img("plots/confusion_speech_spk_YAF2OAF.png", "Speech-only, YAF &rarr; OAF (mean over 3 seeds)", "100%").replace('<figure>', '<figure style="display:inline-block;width:49%">'))

    P("<h2>C. Analysis</h2>")
    P("<h3>C1. Which emotions are easiest and hardest, and why</h3>")
    P("<ul>"
      "<li><b>Within speakers (word-disjoint, leaky split) all seven classes are at 98&ndash;100% recall</b>; the only class that is ever missed is pleasant surprise (0.98 on the leaky split), which is the closest neighbour of happy.</li>"
      "<li><b>Across speakers</b> (mean recall over the two directions, speech-only): <b>neutral is easiest</b> (0.95 / 0.90), followed by disgust and fear in YAF&rarr;OAF (0.88, 0.70) and angry in OAF&rarr;YAF (0.96). <b>Happy is hardest</b> (0.01 and 0.00) followed by pleasant surprise (0.14, 0.19).</li>"
      "<li><b>Why: the two speakers do not act emotions the same way.</b> Measured on 40 clips per speaker and emotion (<code>scripts/acoustic_profile.py</code>):</li></ul>")
    prof = {(r["speaker"], r["emotion"]): r for r in csv.DictReader(open(os.path.join(RES, "tables", "acoustic_profile.csv"), encoding="utf-8"))}
    P("<table><tr><th rowspan=\"2\">Emotion</th><th colspan=\"3\">OAF</th><th colspan=\"3\">YAF</th></tr><tr><th>dur. (s)</th><th>F0 mean (Hz)</th><th>F0 range (Hz)</th><th>dur. (s)</th><th>F0 mean (Hz)</th><th>F0 range (Hz)</th></tr>%s</table>" % "".join(
        "<tr><td>%s</td>%s</tr>" % (e, "".join("<td>%s</td><td>%s</td><td>%s</td>" % (prof[(sp, e)]["duration_s"], prof[(sp, e)]["F0_mean_Hz"], prof[(sp, e)]["F0_range_5_95_Hz"]) for sp in ("OAF", "YAF")))
        for e in ("angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad")))
    P("<ul><li><b>Neutral</b> is the same for both speakers (about 2.0 s, F0 190 Hz, almost flat range 18&ndash;24 Hz), hence easiest. The other classes are realised with speaker-specific cues: "
      "OAF's <b>angry is the fastest class (1.47 s) with a narrow F0 range (106 Hz)</b>, YAF's angry is slow (2.07 s) with a wider range (152 Hz); YAF's <b>happy (1.65 s)</b> is much closer in duration to OAF's angry than OAF's own happy (1.96 s) is. "
      "A model trained on OAF that has learned &ldquo;short and fast = angry&rdquo; therefore labels YAF's happy as angry: <b>happy &rarr; angry for 99% of the clips</b> in OAF&rarr;YAF, and pleasant surprise &rarr; angry for 45%. "
      "Pleasant surprise is hard because it differs from happy mainly by a large F0 excursion (range 236 / 347 Hz) whose size is again speaker-specific (YAF's mean F0 is 322 Hz against OAF's 235 Hz). "
      "In the other direction, OAF's angry (1.47 s) lies outside YAF's range (&ge; 1.59 s for every class) and is spread over pleasant surprise (304 errors) and other high-arousal classes. "
      "These are arousal-axis confusions driven by speaker-dependent cues, not random errors.</li></ul>")
    P("<h3>C2. When does fusion help most?</h3>")
    P("<p>To test this directly I degraded only the speech input with white Gaussian noise (word-disjoint models, 3 seeds, text unchanged):</p>")
    P("<table><tr><th>SNR of the speech</th><th>clean</th><th>30 dB</th><th>20 dB</th><th>10 dB</th><th>5 dB</th><th>0 dB</th></tr>"
      "<tr><td>Speech only</td>%s</tr><tr><td>Fusion (concat)</td>%s</tr><tr><td>Text only</td>%s</tr></table>" % (
          "".join("<td>%s</td>" % N("speech", s) for s in ("clean", "30", "20", "10", "5", "0")),
          "".join("<td>%s</td>" % N("fusion", s) for s in ("clean", "30", "20", "10", "5", "0")),
          "".join("<td>%s</td>" % N("text", s) for s in ("clean", "30", "20", "10", "5", "0"))))
    P("<ul><li>Fusion tracks speech-only closely: its largest advantage is at 0 dB (27.6% vs 22.0%, +5.6 points) and 5 dB (+2.6), but the seed standard deviations (1&ndash;11 points) overlap, so again I cannot claim a significant gain. "
      "Fusion also has smaller variance across seeds at low SNR, which is the more consistent effect.</li>"
      "<li><b>Where fusion would be expected to help</b> is when speech is ambiguous <i>and</i> the text is informative about the label (e.g. sentences whose words carry sentiment, as in IEMOCAP/MELD). On TESS, <b>fusion can only help as a regulariser/ensemble of extra features</b>, never through the text's meaning, because the text is constant across emotions. "
      "The experiments here therefore show the other half of the story: adding an uninformative modality does <b>not hurt</b> (&le; 0.1 point on clean data, and the gated and cross-attention variants learn to cope with it).</li></ul>")
    P(img("plots/noise_robustness.png", "Accuracy under additive white noise on the speech (word-disjoint, mean &plusmn; std over 3 seeds); the text is unchanged.", "62%"))

    P("<h3>C3. Failure cases (speech-only, unseen speaker)</h3>")
    P("<p>For each failure the script finds the same speaker saying the same word in the <i>true</i> and in the <i>predicted</i> emotion, to see whether the model was fooled by a genuine acoustic similarity (readings are my interpretation of the measurements, not tested causally) "
      "(six of the ten cases; the full list of five per direction: <code>Results/error_analysis_speech_spk_*.md</code>, plots <code>Results/plots/failure_*.png</code>).</p>")
    P("<table><tr><th>Case</th><th>Clip</th><th>True &rarr; predicted (confidence)</th><th>Same speaker, same word: true vs predicted class</th><th>Reading</th></tr>"
      "<tr><td>OAF&rarr;YAF #1</td><td>YAF &lsquo;yes&rsquo;</td><td>happy &rarr; angry (0.99)</td><td>1.57 s / 285 Hz / &minus;15.5 dB vs 1.95 s / 253 Hz / &minus;19.1 dB</td><td>Short and fast, like OAF's angry (1.47 s), not like OAF's happy (1.96 s).</td></tr>"
      "<tr><td>OAF&rarr;YAF #2</td><td>YAF &lsquo;read&rsquo;</td><td>pleasant surprise &rarr; angry (0.96)</td><td>mean F0 359 vs 229 Hz; F0 range 441 vs 127 Hz</td><td>An extreme rise, beyond OAF's pleasant-surprise range (mean 235 Hz); the model falls back on the nearest high-arousal class.</td></tr>"
      "<tr><td>OAF&rarr;YAF #3</td><td>YAF &lsquo;peg&rsquo;</td><td>sad &rarr; angry (0.98)</td><td>2.27 s / 213 Hz / &minus;20.0 dB vs 1.92 s / 232 Hz / &minus;17.8 dB</td><td>Quiet (&minus;20 dB) and slow, with a modest F0 range (104 Hz); no clear cue for the angry class, yet the model answers with 0.98 confidence.</td></tr>"
      "<tr><td>OAF&rarr;YAF #4</td><td>YAF &lsquo;take&rsquo;</td><td>fear &rarr; angry (0.98)</td><td>1.68 s / 318 Hz / 180 Hz range vs 2.08 s / 239 Hz / 85 Hz range</td><td>Fast (1.68 s) and high-pitched (318 Hz); in OAF's training data both angry (1.47 s, 257 Hz) and fear (1.56 s, 294 Hz) look like this, and the model picks angry.</td></tr>"
      "<tr><td>YAF&rarr;OAF #2</td><td>OAF &lsquo;raise&rsquo;</td><td>angry &rarr; pleasant surprise (0.97)</td><td>1.60 s, range 111 Hz vs 2.05 s, range 220 Hz</td><td>OAF's angry is short and flat, a pattern absent from YAF's training data (angry 2.07 s, range 152 Hz); the model finds a rising-F0 neighbour in the wrong class.</td></tr>"
      "<tr><td>YAF&rarr;OAF #3</td><td>OAF &lsquo;pass&rsquo;</td><td>happy &rarr; pleasant surprise (0.97)</td><td>2.28 s, 291 Hz, range 301 Hz vs 2.30 s, 219 Hz, range 230 Hz</td><td>Same duration; happy/surprise differ only in pitch excursions that the two speakers realise at different heights.</td></tr></table>")
    P("<ul><li><b>Common pattern:</b> every one of these errors has confidence &ge; 0.95 (all ten listed cases &ge; 0.94). The model is not uncertain; it applies a boundary learned from a single speaker's idiolect with full confidence. Label smoothing (0.05) does not prevent this; calibration across speakers would need training data from more speakers (per-speaker feature standardisation was tried and did not help, section E).</li></ul>")
    P('<div class="two">%s%s</div>' % (img("plots/failure_2_YAF_read_pleasant_surprise.png", "OAF&rarr;YAF case #2: YAF 'read', true pleasant surprise (left) vs YAF's angry (right)"),
                                      img("plots/failure_2_OAF_raise_angry.png", "YAF&rarr;OAF case #2: OAF 'raise', true angry (left) vs OAF's pleasant surprise (right)")))

    P("<h3>C4. Separability of the representations (t-SNE of the Temporal, Contextual and Fusion blocks)</h3>")
    P("<p>t-SNE (perplexity 30) of the three block outputs of the concat-fusion model on the test clips, seed 0, with the silhouette score for emotion and the 5-NN emotion accuracy (5-fold within the test clips; this measures how <i>separable</i> the points are, not how the classifier performs).</p>")
    P("<table><tr><th>Protocol (test set)</th><th>Temporal block (speech)</th><th>Contextual block (text)</th><th>Fusion block</th></tr>"
      "<tr><td>word-disjoint, 560 clips</td><td>%s</td><td>%s</td><td>%s</td></tr>"
      "<tr><td>OAF &rarr; YAF, 1400 clips</td><td>%s</td><td>%s</td><td>%s</td></tr>"
      "<tr><td>YAF &rarr; OAF, 1400 clips</td><td>%s</td><td>%s</td><td>%s</td></tr></table>" % (
          S("word", "temporal"), S("word", "contextual"), S("word", "fusion"),
          S("spk_OAF2YAF", "temporal"), S("spk_OAF2YAF", "contextual"), S("spk_OAF2YAF", "fusion"),
          S("spk_YAF2OAF", "temporal"), S("spk_YAF2OAF", "contextual"), S("spk_YAF2OAF", "fusion")))
    P("<p>(Entries: silhouette(emotion) / 5-NN emotion accuracy.)</p><ul>"
      "<li><b>Temporal block:</b> seven compact clusters. In the word-disjoint model every emotion forms <i>two</i> sub-clusters, one per speaker (5-NN speaker accuracy 0.99): the block encodes emotion <i>and</i> speaker identity, which is what makes the 99.8% possible and why it does not transfer.</li>"
      "<li><b>Contextual block:</b> no structure with respect to emotion (silhouette &asymp; 0, 5-NN 0.143 = chance), as expected when the label is independent of the text; its points are spread by word, not by emotion.</li>"
      "<li><b>Fusion block:</b> clusters are as clean as the temporal block (silhouette 0.82 vs 0.66 on the word protocol: the extra MLP projection tightens clusters), and it keeps the speaker split (5-NN speaker accuracy 0.96). The text vector adds no emotion structure.</li>"
      "<li><b>Cross-speaker:</b> in the unseen speaker the temporal and fusion representations are <i>still</i> highly separable by emotion (5-NN 0.91&ndash;0.98 within that speaker) although the classifier is at 40&ndash;52%. The embedding space is not the problem; the <b>decision boundaries are in the wrong place for the new speaker</b> (a whole emotion cluster is shifted into the neighbouring class), "
      "which is why the t-SNE shows 7 clean clusters while the confusion matrix shows happy &rarr; angry. Section E tests two remedies: per-speaker normalisation does not help, frozen wav2vec2 features do.</li></ul>")
    P(img("representations/tsne_blocks_word_seed0.png", "word-disjoint test set (560 clips). Top: coloured by emotion; bottom: by speaker.", "92%"))
    P(img("representations/tsne_blocks_spk_OAF2YAF_seed0.png", "OAF &rarr; YAF test set (1400 clips). Top: coloured by emotion; bottom: by target word (the Contextual block spreads points by word, not by emotion).", "92%"))

    P("<h2>E. Extra experiments (unseen-speaker protocols, 3 seeds each)</h2>")
    ex = {}
    for ln in open(os.path.join(RES, "extras", "extras_summary.md"), encoding="utf-8").read().splitlines()[2:]:
        c = [x.strip() for x in ln.strip("|").split("|")]
        ex[c[0]] = c
    LAB = [("speech, log-mel (main model)", "Main speech model (reference)"), ("noaug", "Ablation: no SpecAugment"), ("meanpool", "Ablation: mean pooling instead of attention pooling"),
           ("cnn", "Ablation: CNN only (no BiGRU)"), ("gru", "Ablation: BiGRU only (no CNN)"), ("spkn", "Per-speaker feature standardisation"),
           ("w2v", "Frozen wav2vec2-base features (layers 6-9) instead of log-mel"), ("w2v_spkn", "wav2vec2 + per-speaker standardisation")]
    rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td></tr>" % (l, ex[k][1].replace("+-", "&plusmn;"), ex[k][2].replace("+-", "&plusmn;"), ex[k][3]) for k, l in LAB)
    P("<table><tr><th>Speech-only variant</th><th>OAF &rarr; YAF %</th><th>YAF &rarr; OAF %</th><th>mean</th></tr>" + rows + "</table>")
    P("<ul><li><b>Self-supervised features are the one clear improvement:</b> +22 points on average (44.4 &rarr; 66.6), with a smaller seed spread. Per-emotion recall (mean of the two directions) shows what changed: fear 0.99, sad 0.90, neutral 0.99 and happy 0.51 become recognisable, "
      "but <b>angry drops to 0.02</b>, so the gain is a different pattern of errors, not a uniform one. I did not analyse why angry fails; the layers (6&ndash;9) were fixed a priori, not tuned.</li>"
      "<li><b>Per-speaker standardisation does not help</b> (each speaker's features are standardised with that speaker's own mean and variance, computed without labels, including on the unseen test speaker; 44.4 &rarr; 44.4; with wav2vec2 66.6 &rarr; 64.1). Removing speaker-level mean and variance of the features is not enough because the two speakers differ in <i>how</i> each emotion is realised (section C1), not only in their average level.</li>"
      "<li><b>SpecAugment matters</b> (33.1 without vs 44.4 with). The CNN-only and GRU-only variants are not better than the combined encoder (42.6, 42.2 vs 44.4; differences are within the seed spread of 2&ndash;9 points), so I cannot claim the combination is needed on this data.</li>"
      "<li><b>Attention pooling is worse than mean pooling on an unseen speaker</b> (44.4 vs 56.7), a result I did not expect. This ablation was seen on the test speaker, so I did <i>not</i> replace the main model with it (that would be selecting on the test set); it is reported as a finding. A plausible reading is that attention focuses on speaker-specific frames; this was not tested.</li></ul>")
    P("<h3>E2. Does the fusion code exploit a text modality that carries information? (synthetic check)</h3>")
    P("<p>In TESS the text is the same for every emotion, so fusion cannot help. To check that the fusion pipelines <i>can</i> use text, each clip got a second sentence that states an emotion (&ldquo;Say the word fat. I am terrified.&rdquo;), correct for 60% of the clips and a random wrong emotion otherwise. "
      "This is a synthetic test of the code, not a TESS result (<code>scripts/make_synthetic_text.py</code>).</p>")
    LAB2 = [("synth_text", "Text only (synthetic cue)"), ("synth_fusion_concat", "Fusion concat, log-mel speech"), ("synth_fusion_gated", "Fusion gated, log-mel speech"), ("synth_fusion_cross", "Fusion cross-attention, log-mel speech"),
            ("synth_fusion_concat_md", "Fusion concat + modality dropout 0.3"), ("synth_fusion_gated_md", "Fusion gated + modality dropout 0.3"),
            ("w2v", "(reference) wav2vec2 speech only"), ("synth_w2v_fusion_concat", "Fusion concat, wav2vec2 speech"), ("synth_w2v_fusion_concat_md", "Fusion concat, wav2vec2 speech + modality dropout 0.3")]
    rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td><td><b>%s</b></td></tr>" % (l, ex[k][1].replace("+-", "&plusmn;"), ex[k][2].replace("+-", "&plusmn;"), ex[k][3]) for k, l in LAB2)
    P("<table><tr><th>Model</th><th>OAF &rarr; YAF %</th><th>YAF &rarr; OAF %</th><th>mean</th></tr>" + rows + "</table>")
    P("<ul><li>With log-mel speech, the fusion models (47.1 / 41.5 / 42.8) do <b>not</b> exploit the text: they are no better than speech alone (44.4) although text alone reaches 60.9. A likely cause (training accuracy was not logged, so this is not verified) is that the speech branch alone fits the training speaker almost perfectly, so the text branch is never needed (modality dominance).</li>"
      "<li>Modality dropout fixes this only for concatenation (47.1 &rarr; 54.0), not for the gated model (43.3). It remains below text-only, so none of the fusion blocks reaches the accuracy a good combination of the two sources should allow.</li>"
      "<li>With the stronger wav2vec2 speech branch, concat fusion with the informative text scores 70.8 against 66.6 for speech alone, but <b>this does not show that the text was used</b>: fusion alone, even with the uninformative TESS text, already added about 3&ndash;4 points in the earlier comparisons (log-mel: 47.9 vs 44.4; wav2vec2 with speaker normalisation: 67.8 vs 64.1). A matched control (<code>fusion_w2v</code>: same model, original text) is reported in <code>Results/extras/</code>; without it the gain cannot be attributed to the text. Modality dropout did not add anything here (68.5).</li>"
      "<li>What the check does show: with log-mel speech the fusion blocks ignore informative text unless a modality is hidden during training (concat 47.1 &rarr; 54.0 with modality dropout, a control that is matched). The code can therefore use text when forced to, and on TESS a real fusion gain should not be expected.</li></ul>")
    P('<div class="note">Full tables: <code>Results/extras/extras_summary.md</code> and <code>extras_per_emotion.md</code>. Two test speakers only; &plusmn; is the std over 3 seeds, not over speakers.</div>')

    P("<h2>D. Summary and caveats</h2>")
    P("<ul><li>Speech carries all the emotional information in TESS; text-only is at chance (14.3%) by construction of the dataset, and fusion matches speech-only (99.9 vs 99.8%) when both speakers are seen in training.</li>"
      "<li>The real challenge is speaker generalisation (40&ndash;52% on an unseen speaker), where happy and pleasant surprise are essentially unlearnable from one speaker; fusion's apparent +3&ndash;4 point gain is within seed noise. Frozen wav2vec2 features raise the unseen-speaker accuracy to about 67% on average (section E).</li>"
      "<li>Limits: two speakers, acted speech, one carrier phrase; 3&ndash;5 seeds; the noise experiment uses white noise only; t-SNE is qualitative (silhouette and k-NN are given for the quantitative view). "
      "The fusion benefit that a text-informative dataset (IEMOCAP, MELD, CMU-MOSEI) would show cannot be assessed on TESS.</li></ul>")

    html = "<!doctype html><html><head><meta charset='utf-8'><style>%s</style></head><body>%s</body></html>" % (CSS, "\n".join(p))
    hp = os.path.join(ROOT, "Report", "_report.html")
    pdf = os.path.join(ROOT, "Report", "Report_2b.pdf")
    open(hp, "w", encoding="utf-8").write(html)
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf=" + pdf, "file:///" + hp.replace("\\", "/")], check=True, timeout=180)
    os.remove(hp)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
