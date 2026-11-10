PROJECT 2 RESEARCH PROPOSAL
Mateo Onate and Demian Viteri

READY TO READ
Research_Proposal.pdf contains the proposal in English. It has three pages,
including eight scientific references, one architecture figure, and one table.
This is a proposal. It does not report experiments that have already occurred.

OVERLEAF
1. Select New Project, then Upload Project.
2. Upload Project2_Proposal_Overleaf.zip.
3. Select access.tex as the main document.
4. Select XeLaTeX as the compiler. This matches the engine family used to verify
   the included three-page PDF. Keep the default font sizes and class layout.
5. Recompile. After any changes, confirm the total remains at most three pages.

The supplied gallery template selects pdfLaTeX originally. access.tex keeps that
engine path and also includes a small conditional compatibility block for XeTeX.
The compatibility block maps the class's spot blue to the same CMYK color.
It does not change the research content. The class files stay unchanged.

FORMATTING SOURCE AND PROVENANCE
Formatting reference requested by the instructor:
https://es.overleaf.com/latex/templates/ieee-access-latex-template/cdxrhtbjgszv

The gallery identifies IEEE as the template author and specifies ieeeaccess.
The direct IEEE ZIP downloads returned HTTP 403 in this environment.
The bundled class files and header images came from this public mirror:
https://github.com/TMLlaboratory/latex-template-ieee-access
Pinned commit: 521d25ee27ea6247aebc18954b5cb048b667b8cd

Byte-for-byte identity with the gallery's original archive was not verified.
The proposal follows the gallery's document-class and front-matter structure.
Course metadata replaces journal DOI, publication volume, and placeholder author
details. No journal publication or IEEE membership is claimed.

LOCAL VERIFICATION
The Codex native preview command could not launch its compiler because of a
Windows access-denied error. The PDF was compiled with the existing Tectonic
executable bundled in the application, invoked from the task workspace.
No separate TeX installation or LaTeX plugin was installed.
The compiled PDF was rendered and all three pages were inspected visually.
This verification covers the included PDF. Changes in engine, source, or package
versions require a new page-count check.

EXPERIMENT AT A GLANCE
Area: Privacy-Preserving Cryptography.
Approach: Experimental Analysis.
Data: public MNIST digit images.
Model: softmax classifier, 7,850 parameters.
Clients: ten simulated clients on one CPU computer.
Training: ten rounds, sixteen optimizer steps per client per round.
Distributions: randomly distributed data and label-skewed data.
Settings: ordinary FedAvg, clipping-only, and DP budgets 2, 4, and 8.
Repetitions: three per condition, thirty complete runs in total.
Metrics: accuracy, accumulated privacy bounds, runtime, and peak memory.
Privacy unit: one image. Budgets apply per run, not to the entire benchmark.
The pilot can revise shared settings before the comparison. It does not require
an ESP32, a GPU, paid infrastructure, or private personal data.

SCIENTIFIC SOURCE LINKS
The actual bibliography appears within the three-page proposal.

[1] FedAvg, McMahan et al., AISTATS 2017:
https://proceedings.mlr.press/v54/mcmahan17a.html

[2] Gradient leakage, Zhu et al., NeurIPS 2019:
https://proceedings.neurips.cc/paper_files/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html

[3] Differential privacy foundations, Dwork and Roth, 2014:
https://www.microsoft.com/en-us/research/publication/algorithmic-foundations-differential-privacy/

[4] DP-SGD, Abadi et al., ACM CCS 2016, author preprint:
https://arxiv.org/abs/1607.00133

[5] Record-level federated privacy, Zheng et al., AISTATS 2021:
https://proceedings.mlr.press/v130/zheng21a.html

[6] Opacus, Yousefpour et al., scientific preprint, 2021:
https://arxiv.org/abs/2109.12298

[7] MNIST context, LeCun et al., Proceedings of the IEEE 1998, coauthor page:
https://bottou.org/papers/lecun-98h

[8] Renyi differential privacy, Mironov, IEEE CSF 2017, author preprint:
https://arxiv.org/abs/1702.07476

Supplementary implementation documentation checked during preparation:
https://opacus.ai/api/privacy_engine.html
https://opacus.ai/api/_modules/opacus/privacy_engine.html

Research-proposal structure guidance consulted:
https://writersworkshop.illinois.edu/resources-2/writer-resources/academic-writing/research-proposals/
https://crf.georgetown.edu/research/research-toolkit/structuring-your-research-proposal/
