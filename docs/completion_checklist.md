# Completion and remaining team work

The code/report tools support the midterm checklist in the supplied PDF and the implementation plan in the DOCX. A generated artifact is not proof that the team photographs, names or GPU measurements are complete. The notebook prints missing information and labels synthetic results.

| Requirement | Implemented source/tooling | Work the team must provide |
|---|---|---|
| 3.1 Team / responsibilities | Report sections and V0–V3 work-plan table; team_info example | Verified full names, student IDs, contributions; confirm source author headers |
| 3.2 Pipeline understanding | Six-stage explanation and B→A convention | Review wording as the team's explanation |
| 3.3 Methods and parameters | Methods/quality/GPU discussion; parameters printed from executable | Review decisions and explain them in defense |
| 3.4 Complete single-threaded V0 | Submitted source build, six-stage pipeline, CPU regression tools | Match course starter if the actual starter requires additional signatures (starter not supplied) |
| 3.5 Synthetic homography correctness | Full-image translation/projective tests, corner error/ground truth output, separate outlier S5 test | Review actual output |
| 3.5 Three sizes, all stage timing | Common-scale dataset preparation, CLI JSON/CSV, chart, hotspot/Amdahl computation | Add actual photographs and rerun to obtain real-site measurements |
| 3.5 Panorama per site | Measured-output gallery, failure logging | Registered real sites and photos; inspect quality/failure case |
| 3.6 Dataset | Schema validation; standard/challenging coverage checks; dataset URL output | Actual site registrations; original photos; downloadable original + prepared dataset archive |
| 3.7 GPU plan | Mapping V1, optimization/risk table and Amdahl stage set; no CUDA execution required at Midterm | Confirm stage selection using real V0 profiling |
| Source citations and AI | External library references, author/source comments, report declaration field | Verified authors and honest declaration of the AI assistance used |
| Reproducibility | Build from src; public ZIP retrieval; arbitrary relative photo layout; current Colab output export/explicit saved-file fallback; implementation/result fingerprints | Rerun in fresh Colab with the actual public dataset link; retain all outputs; local checks do not establish remote Colab execution |
| Submission zip | Draft or strict Group08_Midterm.zip packaging | Use strict validation after completing the pending team fields |

PDF requires at least three sites, one standard pair per site and one challenging pair overall. A broad summary line in the DOCX mentions a challenging pair for each site; its detailed work plan and the PDF specify one overall. Adding more challenging pairs is supported by the manifest.

Synthetic scaling output is technical validation only. It cannot fulfil registered-site dataset, real panorama or real-scene timing requirements. The Midterm does not require running CUDA, Final slides or video. Course starter interfaces remain unverified until the actual starter notebook is supplied. Source and tooling coverage cannot establish the team-controlled registration, authors or dataset-link permissions.
