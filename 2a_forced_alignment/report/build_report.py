"""Builds report/Report_2a.pdf (HTML -> PDF with Microsoft Edge headless) from the result files in outputs/.

    python report/build_report.py --repo-url https://github.com/<user>/<repo>
"""
import argparse
import csv
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
AN = os.path.join(ROOT, "outputs", "analysis")
FIG = os.path.join(ROOT, "outputs", "figures")


def rows(name):
    return list(csv.DictReader(open(os.path.join(AN, name), encoding="utf-8")))


def img(name, cap, w="100%"):
    return '<figure><img src="file:///%s" style="width:%s"><figcaption>%s</figcaption></figure>' % (os.path.join(FIG, name).replace("\\", "/"), w, cap)


CSS = """
@page { size: A4; margin: 14mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #1b1b1b; }
h1 { font-size: 17pt; margin: 0 0 2pt; } .sub { color: #555; margin-bottom: 7pt; font-size: 9pt; }
h2 { font-size: 11.5pt; margin: 11pt 0 3pt; border-bottom: 1.5px solid #2b4c7e; color: #2b4c7e; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 6pt; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: left; vertical-align: top; } th { background: #eceff3; }
figure { margin: 4pt 0 6pt; page-break-inside: avoid; } figcaption { font-size: 8pt; color: #444; }
ul { margin: 2pt 0 4pt 15pt; padding: 0; } li { margin: 1.5pt 0; } code { font-family: Consolas, monospace; font-size: 8.3pt; background: #f1f1f1; padding: 0 2px; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 6pt; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/&lt;user&gt;/&lt;repo&gt;")
    a = ap.parse_args()
    summ = {r["config"]: r for r in rows("summary_by_config.csv")}
    agree = {(r["config"], r["level"]): r for r in rows("boundary_agreement_vs_A3_norm_custom_arpa.csv")}
    per = rows("per_utterance.csv")
    a3 = [r for r in per if r["config"] == "A3_norm_custom_arpa"]
    ll_news = sum(float(r["overall_loglik"]) for r in a3 if r["utterance"].startswith("F2B")) / 3
    ll_isle = sum(float(r["overall_loglik"]) for r in a3 if r["utterance"].startswith("ISLE")) / 3
    sq = rows("squeezed_phones_A3_norm_custom_arpa.csv")
    vow = {r["word"]: r for r in rows("isle_vowel_formants.csv")}
    f = lambda w, k: float(vow[w][k])
    d_bb = (abs(f("bad", "F1_50%") - f("bed", "F1_50%")), abs(f("bad", "F2_50%") - f("bed", "F2_50%")))
    d_wb = (abs(f("white", "F1_50%") - f("bait", "F1_50%")), abs(f("white", "F2_50%") - f("bait", "F2_50%")))
    d_nn = (abs(f("new", "F1_50%") - f("no", "F1_50%")), abs(f("new", "F2_50%") - f("no", "F2_50%")))
    ag = lambda c, l: agree[(c, l)]
    p = []
    A = p.append
    A("<h1>Assignment 2a: Forced alignment with the Montreal Forced Aligner</h1>")
    A('<div class="sub">Report (short) &nbsp;|&nbsp; Code, TextGrids and figures: %s</div>' % a.repo_url)
    A("<h2>1. What was done</h2>")
    A("<p>The six provided utterances (97 s) were aligned at word and phone level with <b>MFA 3.4.3</b> (Kaldi HMM-GMM; run in WSL Ubuntu). "
      "The pipeline is fully automated (<code>scripts/run_pipeline.sh</code>): corpus preparation, validation, a <b>custom pronunciation dictionary built with MFA's G2P model</b>, "
      "alignment with <b>eight configurations</b> (two pre-trained acoustic models, an adapted model, a model trained from scratch, a fine-tuning option, a pronunciation-variant dictionary), and an analysis of the TextGrids with quantitative proxies, "
      "since no hand-labelled reference alignment exists. TextGrids and figures were also inspected visually (Praat-style plots; the same files open in Praat).</p>")
    A("<h2>2. Data, models and dictionaries</h2>")
    A("<table><tr><th>Files</th><th>Content</th><th>Length</th><th>Words</th></tr>"
      "<tr><td>F2BJRLP1-3</td><td>news-radio reading (one speaker), long sentences, names, acronyms (S.J.C., WBUR), numbers in words</td><td>25.3, 28.7, 30.7 s</td><td>73, 74, 82</td></tr>"
      "<tr><td>ISLE_SESS0131_BLOCKD02_01-03</td><td>non-native speaker, minimal pairs: <i>I said white not bait / new not no / bad not bed</i></td><td>4.1, 3.9, 4.5 s</td><td>5 each</td></tr></table>")
    A("<table><tr><th></th><th>Transcripts</th><th>Dictionary</th><th>Acoustic model</th><th>Words / phones</th><th>OOV items</th></tr>")
    names = {"A1_naive_stock_arpa": ("A1", "as provided", "stock <code>english_us_arpa</code>", "english_us_arpa"), "A2_norm_stock_arpa": ("A2", "spoken form", "stock", "english_us_arpa"),
             "A3_norm_custom_arpa": ("<b>A3</b>", "spoken form", "<b>custom</b> (stock + G2P)", "english_us_arpa"), "A4_norm_custom_arpa_ft": ("A4", "spoken form", "custom", "english_us_arpa + <code>--fine_tune</code>"),
             "A5_norm_custom_adapted": ("A5", "spoken form", "custom", "english_us_arpa adapted (<code>mfa adapt</code>)"), "A6_norm_custom_mfa": ("A6", "spoken form", "custom (<code>english_us_mfa</code> + G2P)", "english_mfa (IPA phone set)")}
    for k, (n, t, d, am) in names.items():
        A("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s / %s</td><td>%s</td></tr>" % (n, t, d, am, summ[k]["words"], summ[k]["phones"], summ[k]["unk_or_spn_items"]))
    A("</table>")
    A("<p><i>Transcripts:</i> the provided text was only whitespace-normalised (A1); in the spoken-form variant acronyms are spelled as the speaker says them (S.J.C.'s &rarr; S J C's, WBUR's &rarr; W B U R's; verified with an independent ASR). "
      "<i>Dictionary:</i> <code>mfa validate</code> found the out-of-vocabulary words (<i>Dukakis, Melnicove, Maffy, politicize</i>, and <i>WBUR</i> in the naive text; the <code>english_us_mfa</code> dictionary additionally lacks <i>judgeships, Margo</i>); "
      "<code>mfa g2p</code> produced three candidate pronunciations for each, appended to the stock dictionary so the aligner chooses the best one per occurrence.</p>")
    A("<h2>3. Sample alignments</h2>")
    A(img("ISLE_SESS0131_BLOCKD02_03_sprt1.png", "Non-native utterance &lsquo;I said bad not bed&rsquo; (A3): waveform, spectrogram, word tier and ARPAbet phone tier (vowels orange).", "56%"))
    A(img("compare_F2BJRLP2_dukakis.png", "News reading around the out-of-vocabulary name <i>Dukakis</i>: phone tier of the main system (A3) and word tiers of A1, A2 and A6.", "62%"))
    A("<h2>4. Key observations</h2><ul>")
    A("<li><b>Out-of-vocabulary words.</b> With the stock dictionary every OOV word is a single <code>spn</code> (&lsquo;spoken noise&rsquo;) unit covering the whole word (420-870 ms), so no phone boundaries exist inside it; "
      "the G2P dictionary removes all of them (A3: 0). The aligner chose different G2P candidates per occurrence (<i>Dukakis</i>: D UW0 K AA1 K IH0 S in one sentence, D UW1 K AA1 K IH0 S in the other), i.e. it uses the acoustics to pick the variant. "
      "Spelling out the acronyms is needed for the same reason: <i>wbur</i> is OOV, but the letters W, B, U, R are all known.</li>")
    A("<li><b>Stability of the alignment.</b> Changing the transcript form or the dictionary does not move boundaries of the in-vocabulary words (A1, A2 vs A3: %s%% and %s%% of word boundaries within 20 ms); "
      "fine-tuning (A4) and speaker adaptation (A5) change them by %s ms and %s ms on average. Adapting the model on only 97 s of audio therefore changes almost nothing: more data would be needed for a measurable effect.</li>" % (
          ag("A1_naive_stock_arpa", "word")["pct_within_20ms"], ag("A2_norm_stock_arpa", "word")["pct_within_20ms"], ag("A4_norm_custom_arpa_ft", "word")["mean_ms"], ag("A5_norm_custom_adapted", "word")["mean_ms"]))
    A("<li><b>A different acoustic model gives a different, and sometimes wrong, alignment.</b> <code>english_mfa</code> (A6) agrees with A3 within 20 ms for %s%% of word boundaries (mean %s ms) &mdash; "
      "but in two of the three short utterances it assigns the long silence after the last word to that word (<i>bait</i> 1.70-4.12 s instead of 1.80-2.25 s; <i>bed</i> 1.47-4.50 s instead of 1.49-2.04 s), an error of up to 2.5 s that is obvious in the waveform. "
      "The alignment log-likelihoods of different models are not comparable, so agreement between systems and visual checks are the only way to detect such failures.</li>" % (ag("A6_norm_custom_mfa", "word")["pct_within_20ms"], ag("A6_norm_custom_mfa", "word")["mean_ms"]))
    A("<li><b>Timing resolution and squeezed phones.</b> Every boundary is a multiple of the 10 ms frame shift, and the HMM topology forces at least 3 frames (30 ms) per phone. %d of %s phones (%s%%) sit at this minimum: "
      "25 stops (normal), but also 29 vowels and 30 other consonants. Three regions have clusters of &ge;3 squeezed phones within 150 ms: <i>important</i> (AO1 T N T, 13.9-14.1 s), <i>appointments</i> (N T M N, 7.3-7.5 s) and the spelled-out <i>W B U R</i> (B AH0 L, 20.7 s). "
      "These are fast or reduced realisations forced into the full canonical phone sequence: no phone can be skipped by a forced aligner, so a reduced syllable is compressed instead of deleted, and the interior boundaries there are unreliable.</li>" % (len(sq), summ["A3_norm_custom_arpa"]["phones"], summ["A3_norm_custom_arpa"]["pct_phones_<=30ms"]))
    A("<li><b>Pauses and punctuation.</b> In each news file A3 marks 6 interior silences of at least 150 ms; 4 coincide with punctuation and 2 are phrasing/breath pauses without punctuation (after <i>Hennessy, makes</i>; <i>Dukakis, career</i>; <i>credit, judgeships</i>). "
      "Conversely 3 of 7, 3 of 7 and 1 of 5 interior punctuation marks have no audible pause: the speaker does not pause at every comma, so punctuation is only a weak predictor of pauses.</li>")
    A("<li><b>Non-native speech: the alignment hides a missing contrast.</b> A forced aligner must use the canonical phones, so it produces a clean, plausible alignment even when the speaker does not realise the contrast. "
      "Formant measurements (Praat, F1/F2 at the vowel mid-point of the aligned interval) show that <i>bad</i> and <i>bed</i> differ by only %.0f Hz (F1) and %.0f Hz (F2), i.e. /&aelig;/ is realised close to /&epsilon;/ (classic adult-male values differ by about 130 Hz in F1 and 120 Hz in F2), "
      "and an independent ASR transcribed the utterance as &ldquo;bad not bad&rdquo;. <i>white/bait</i> (F2 differs by %.0f Hz) and <i>new/no</i> (%.0f Hz) keep a clear contrast. "
      "MFA's per-frame log-likelihood is lower for the non-native clips (%.1f vs %.1f for the news reading), but this global score cannot localise the problem to the vowel: forced alignment is not pronunciation assessment (a phone recogniser or GOP score is needed).</li></ul>" % (
          d_bb[0], d_bb[1], d_wb[1], d_nn[1], ll_isle, ll_news))
    en = {r["config"]: r for r in rows("pause_boundary_vs_energy_summary.csv")}
    vc = rows("variant_choice_A7.csv")
    chosen = {r["word"]: r for r in vc}
    A("<h2>5. Additional experiments</h2><ul>")
    A("<li><b>Pronunciation variants (A7).</b> I added the confusable vowel as a second pronunciation of each minimal-pair word (<i>bad</i>: B AE1 D / B EH1 D, <i>white</i>: AY / EY, <i>new</i>: UW / OW, &hellip;) and let the aligner choose per occurrence by likelihood. "
      "It kept the canonical vowel for five of the six words but chose <b>EH for <i>bad</i></b> (B EH D, the same phones as <i>bed</i>), independently confirming the formant result: this speaker does not make the bad/bed contrast. "
      "All other boundaries are unchanged (%s%% of word boundaries within 20 ms of A3).</li>" % ag("A7_norm_variants_arpa", "word")["pct_within_20ms"])
    A("<li><b>Acoustic model trained from scratch on the six clips (A8, <code>mfa train</code>).</b> With 97 s of audio the monophone/triphone GMMs do not converge: the default beam finds no alignment at all (<code>NoAlignmentsError</code>; it needed <code>--beam 100 --retry_beam 400</code>), "
      "and the result is degenerate: only %s%% of the word boundaries are within 20 ms of A3 (mean difference %s ms), %s%% of the phones are at the 30 ms minimum, and the sentence-final words absorb seconds of silence (figure). "
      "This is the reason to use pre-trained models and, at most, adapt them (A5).</li>" % (ag("A8_trained_on_data_only", "word")["pct_within_20ms"], ag("A8_trained_on_data_only", "word")["mean_ms"], summ["A8_trained_on_data_only"]["pct_phones_<=30ms"]))
    A("<li><b>An independent acoustic check at the pauses.</b> Without a hand-labelled reference, I compared every boundary between a pause (&ge; 80 ms) and a word with the position where the short-time energy of the waveform crosses the midpoint between silence and speech (search window &plusmn;150 ms; this is an automatic reference, not a manual annotation). "
      "For A1&ndash;A5 and A7 the mean absolute difference is %s&ndash;%s ms (median 17.5 ms, i.e. 1&ndash;2 frames), A6 %s ms; the from-scratch model A8 is at %s ms. So the pause placement of the pre-trained systems is acoustically consistent, and the energy check does separate a good from a degenerate alignment; it is too coarse to rank A1&ndash;A5, which differ by less than a frame. "
      "A template of 12 informative boundaries to correct by hand in Praat is provided (<code>outputs/analysis/manual_check_template.csv</code>, evaluated with <code>scripts/extras_analysis.py --manual</code>).</li></ul>" % (
          min(float(en[c]["mean_abs_error_ms"]) for c in ("A1_naive_stock_arpa", "A2_norm_stock_arpa", "A3_norm_custom_arpa", "A4_norm_custom_arpa_ft", "A5_norm_custom_adapted", "A7_norm_variants_arpa")),
          max(float(en[c]["mean_abs_error_ms"]) for c in ("A1_naive_stock_arpa", "A2_norm_stock_arpa", "A3_norm_custom_arpa", "A4_norm_custom_arpa_ft", "A5_norm_custom_adapted", "A7_norm_variants_arpa")),
          en["A6_norm_custom_mfa"]["mean_abs_error_ms"], en["A8_trained_on_data_only"]["mean_abs_error_ms"]))
    A(img("variants_and_scratch_03_sprt1.png", "&lsquo;I said bad not bed&rsquo;: main alignment (A3, top), A7 phone tier (the aligner picks EH for <i>bad</i>) and the from-scratch model A8 (bottom), which has lost the structure of the utterance.", "44%"))
    A("<h2>6. Limitations and extra credit</h2>")
    A("<p>No gold alignment exists, so accuracy was assessed through system agreement, acoustic plausibility and visual inspection, not as an error rate against a reference; the six recordings are few and the formant analysis uses one token per word. "
      "<b>Extra credit covered:</b> custom dictionary trained with the G2P model plus a pronunciation-variant dictionary; several acoustic models compared (<code>english_us_arpa</code>, <code>english_mfa</code>, adapted, trained from scratch); a fully automated pipeline (<code>setup_mfa.sh</code>, <code>run_pipeline.sh</code>). "
      "<b>Outputs:</b> <code>outputs/&lt;configuration&gt;/&lt;speaker&gt;/&lt;utterance&gt;.TextGrid</code> (8 x 6 files), <code>outputs/analysis/*.csv</code>, <code>outputs/figures/</code>, <code>outputs/praat_view/</code>.</p>")
    html = "<!doctype html><html><head><meta charset='utf-8'><title>Assignment 2a report</title><style>%s</style></head><body>%s</body></html>" % (CSS, "\n".join(p))
    hp = os.path.join(ROOT, "report", "Report_2a.html")
    open(hp, "w", encoding="utf-8").write(html)
    pdf = os.path.join(ROOT, "report", "Report_2a.pdf")
    subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf=" + pdf, "file:///" + hp.replace("\\", "/")], check=True, timeout=180)
    print("wrote", pdf, os.path.getsize(pdf))


if __name__ == "__main__":
    main()
