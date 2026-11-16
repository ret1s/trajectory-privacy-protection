# Submission preparation notes

Status on 16 November 2026: English content and source package for author review. No journal submission, publication, reviewer contact or author agreement has been performed.

## Target and format

The provisional target remains a regular research article in **Journal of Information Security and Applications**, as discussed with the supervisor. The historical 2022 special issue is not treated as an open call.

The main manuscript uses Elsevier's official `elsarticle` class with numbered citations. The flat source package follows Elsevier's general guidance for LaTeX source files. These are generic preparation choices, not confirmation of every current JISA requirement.

- [Elsevier LaTeX instructions](https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions), consulted 16 November 2026.
- [JISA Guide for Authors](https://www.sciencedirect.com/journal/journal-of-information-security-and-applications/publish/guide-for-authors), attempted 16 November 2026: access returned HTTP 403.
- [Editorial Manager file preparation](https://www.elsevier.support/publishing/answer/how-do-i-prepare-my-files-for-submission-in-editorial-manager).

The abstract is kept under 250 words and five short highlights are supplied as conservative draft choices. A journal-specific word/page limit, mandatory file list, review/anonymization model, required declarations and current submission route must be checked against the accessible journal guide before upload. No acceptance or submission-readiness claim is implied.

## Content decisions already made

- GeoI-BQ v1 and all benchmark outputs remain frozen. No new empirical result was generated while writing this paper.
- Main results compare three audited paper adaptations plus Raw GPS and REM-Z controls. Adaptations are labeled throughout; they are not the authors' official implementations.
- The inference theorem concerns a conditional observation channel; the manuscript does not claim unconditional identity anonymity or deployment-certified pure DP.
- Utility scores use received-candidate masks. The prototype also loads complete static reference information. Online necessity and superiority to an equally informed offline client are not established.
- Γ0's slightly higher Recall, finite-catalogue saturation, S5's weak Raw control, partial native-metric coverage and cap-exhaustion gaps are retained.
- Recent related-work coverage inherits the audited 28 October 2023–28 October 2026 window. The IWPE 2026 lead is cited as an official program abstract only, not as a full paper or reproduced comparator.

## Items deliberately deferred

The user requested content first, so author order, affiliations, ORCIDs and corresponding-author details remain open. No author identity or contact address has been invented.

Before an actual submission, the responsible authors must:

1. Review the English wording, mathematical statements, evidence and novelty positioning; approve the final article.
2. Supply authorship, contributions, funding and competing-interest statements reflecting their actual circumstances. A statement of “none” is not assumed here.
3. Confirm geographical-data licenses, redistribution permissions, and the synthetic-data description. No human-participant study or ethics approval is fabricated.
4. Register an appropriate citable code/data archive and update the availability statement. The flat LaTeX archive is only the manuscript source package.
5. Review the included generative-AI disclosure and align it with the actual assistance used and current journal policy.
6. Verify the live JISA author guide and submission requirements; prepare a cover letter after the author information and contribution claim are agreed.

These are submission metadata and author decisions; they do not prevent reviewing the present paper content. No external submission is authorized by the writing task.
