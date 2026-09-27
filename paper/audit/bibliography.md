# Bibliography audit (September 2026)

## Method and scope

All **91 original BibTeX records** were inventoried and screened for title/author,
venue/year/pages, duplicate identities, and relevance to the revised text. Public
Crossref title queries were followed by primary publisher, proceedings, author
manuscript, or author-institution checks for **every retained citation**. Approximate
search matches were not accepted automatically: the top matches for the quantum
GAN paper and Watrous book were different works, and Hiai's title also matched a
later reprint. Crossref throttled several lookups (HTTP 429), so an unsuccessful
registry query is recorded as unverified, never evidence that a work is fictitious.

The live bibliography now contains **33 distinct, cited works** (29 retained and
4 added). Unused historical records are removed from the live file, but their
original contents and lookup evidence remain in `bibliography_inventory.json`.
A removal for lack of use is not a claim that the work is hallucinated. The table
below identifies a primary verification source for every published citation.

## Material corrections

- `CensorRezac15`: the supplied author/title/venue combination could not be
  substantiated. Removed rather than presenting it as an established source.
  Bauschke--Borwein's verified 1997 article now supports the general Bregman context;
  it is not substituted as evidence for the old specific linear-rate assertion.
- `Friedrichs38`: the supplied title and bibliographic combination is unreliable;
  removed together with the associated unsupported historical rate attribution.
- `candrall1980some`: corrected **Candrall** to **Crandall**, full author names,
  exact title, journal, and DOI. The historical key remains stable.
- `schiebinger2019optimal`: corrected the author list (including Joshua Gould,
  Siyan Liu, Peter Berube, Lia Lee), added the original research-article DOI
  `10.1016/j.cell.2019.01.006` (not the separate correction's DOI), and the e-page suffix.
- `chizat2018scaling`: restored **optimal** in the published title and added the DOI.
- `chizat2025sharper`: replaced preliminary `1--50` pagination by volume 215,
  pages 809--858, and the 2026 issue year; the publisher lists online publication in 2025.
- Consolidated duplicate identities (Bregman, Sinkhorn, Franklin--Lorenz,
  Dvurechensky et al.). In particular the canonical Dvurechensky entry uses **Alexey Kroshnin**.
- Added the directly relevant Feliciangeli--Gerolin--Portinale **2023 JFA** article,
  rather than falsely suggesting that all quantum dual bounds and convergence are open.
  Added Caputo et al.'s verified convex-regularization preprint, with its version date explicit.
- Corrected the discussion of Orlin: the cited strongly polynomial scaling algorithm
  is not synonymous with network simplex, and its complexity was not the earlier
  oversimplified `O(p n log n)` expression.
- Removed unused sources from the publishable database rather than carrying forward
  incomplete metadata or invented page ranges. Their archival inventory remains inspectable.

## Sources for all retained citations

| Key | Primary source used for verification |
| --- | --- |
| `AhujaEtAl93` | [https://mitmgmtfaculty.mit.edu/jorlin/network-flows/](https://mitmgmtfaculty.mit.edu/jorlin/network-flows/) |
| `BauschkeBorwein1997` | [https://cmps-people.ok.ubc.ca/bauschke/Research/07.pdf](https://cmps-people.ok.ubc.ca/bauschke/Research/07.pdf) |
| `Beckmann52` | [https://doi.org/10.2307/1907646](https://doi.org/10.2307/1907646) |
| `Bregman67` | [https://doi.org/10.1016/0041-5553(67)90040-7](https://doi.org/10.1016/0041-5553(67)90040-7) |
| `CsiszarTusnady84` | [https://www.mit.edu/~6.454/www_fall_2002/shaas/Csiszar.pdf](https://www.mit.edu/~6.454/www_fall_2002/shaas/Csiszar.pdf) |
| `Cuturi13` | [https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html](https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) |
| `DemingStephanIPFP` | [https://doi.org/10.1214/aoms/1177731829](https://doi.org/10.1214/aoms/1177731829) |
| `FranklinLorenz89` | [https://doi.org/10.1016/0024-3795(89)90490-4](https://doi.org/10.1016/0024-3795(89)90490-4) |
| `Sinkhorn64` | [https://doi.org/10.1214/aoms/1177703591](https://doi.org/10.1214/aoms/1177703591) |
| `altschuler2017near` | [https://papers.nips.cc/paper_files/paper/2017/hash/491442df5f88c6aa018e86dac21d3606-Abstract.html](https://papers.nips.cc/paper_files/paper/2017/hash/491442df5f88c6aa018e86dac21d3606-Abstract.html) |
| `aubin2022mirror` | [https://papers.neurips.cc/paper_files/paper/2022/file/6e3daaeca6be8579573f69082b2dd58b-Paper-Conference.pdf](https://papers.neurips.cc/paper_files/paper/2022/file/6e3daaeca6be8579573f69082b2dd58b-Paper-Conference.pdf) |
| `benamou2015iterative` | [https://doi.org/10.1137/141000439](https://doi.org/10.1137/141000439) |
| `borwein1994dad` | [https://doi.org/10.1006/jfan.1994.1089](https://doi.org/10.1006/jfan.1994.1089) |
| `caglioti2019quantum` | [https://doi.org/10.1007/s10955-020-02571-7](https://doi.org/10.1007/s10955-020-02571-7) |
| `candrall1980some` | [https://doi.org/10.1090/S0002-9939-1980-0553381-X](https://doi.org/10.1090/S0002-9939-1980-0553381-X) |
| `caputo2024quantum` | [https://arxiv.org/abs/2409.03698](https://arxiv.org/abs/2409.03698) |
| `chakrabarti2019quantum` | [https://proceedings.neurips.cc/paper_files/paper/2019/hash/f35fd567065af297ae65b621e0a21ae9-Abstract.html](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f35fd567065af297ae65b621e0a21ae9-Abstract.html) |
| `chakrabarty2021sinkhornsublinearrate` | [https://doi.org/10.1007/s10107-020-01503-3](https://doi.org/10.1007/s10107-020-01503-3) |
| `chizat2018scaling` | [https://doi.org/10.1090/mcom/3303](https://doi.org/10.1090/mcom/3303) |
| `chizat2025sharper` | [https://doi.org/10.1007/s10107-025-02242-z](https://doi.org/10.1007/s10107-025-02242-z) |
| `daitch2008faster` | [https://doi.org/10.1145/1374376.1374441](https://doi.org/10.1145/1374376.1374441) |
| `dvurechensky2018computational` | [https://proceedings.mlr.press/v80/dvurechensky18a.html](https://proceedings.mlr.press/v80/dvurechensky18a.html) |
| `evans2012phylogenetic` | [https://doi.org/10.1111/j.1467-9868.2011.01018.x](https://doi.org/10.1111/j.1467-9868.2011.01018.x) |
| `feliciangeli2021noncommutative` | [https://research-explorer.ista.ac.at/record/12911](https://research-explorer.ista.ac.at/record/12911) |
| `golse2016mean` | [https://doi.org/10.1007/s00220-015-2485-7](https://doi.org/10.1007/s00220-015-2485-7) |
| `hiai1981sufficiency` | [https://msp.org/pjm/1981/96-1/pjm-v96-n1-p08-s.pdf](https://msp.org/pjm/1981/96-1/pjm-v96-n1-p08-s.pdf) |
| `kalantari2008complexity` | [https://doi.org/10.1007/s10107-006-0021-4](https://doi.org/10.1007/s10107-006-0021-4) |
| `leger2021gradient` | [https://doi.org/10.1007/s00245-020-09697-w](https://doi.org/10.1007/s00245-020-09697-w) |
| `ning2014matrix` | [https://doi.org/10.1109/TAC.2014.2350171](https://doi.org/10.1109/TAC.2014.2350171) |
| `orlin1993minimum` | [https://doi.org/10.1287/opre.41.2.338](https://doi.org/10.1287/opre.41.2.338) |
| `peyre2019quantum` | [https://doi.org/10.1017/S0956792517000274](https://doi.org/10.1017/S0956792517000274) |
| `schiebinger2019optimal` | [https://doi.org/10.1016/j.cell.2019.01.006](https://doi.org/10.1016/j.cell.2019.01.006) |
| `watrous2018theory` | [https://assets.cambridge.org/97811071/80567/frontmatter/9781107180567_frontmatter.pdf](https://assets.cambridge.org/97811071/80567/frontmatter/9781107180567_frontmatter.pdf) |

## Disposition of all original records

| Original key | Disposition and evidence |
| --- | --- |
| `daitch2008faster` | Retained; checked against the primary source above. |
| `rubner1998metric` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `sia2019ollivier` | Uncited; removed. Registry title match: 10.1038/s41598-019-46079-x (not a certification of every original field). |
| `sandhu2015graph` | Uncited; removed. Registry title match: 10.1038/srep12323 (not a certification of every original field). |
| `kusner2015word` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `grauman2004fast` | Uncited; removed. Registry title match: 10.1109/cvpr.2004.1315035 (not a certification of every original field). |
| `chen2025maximum` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `dong2025nested` | Uncited; removed. Registry title match: 10.1145/3744639 (not a certification of every original field). |
| `evans2012phylogenetic` | Retained; checked against the primary source above. |
| `altschuler2017near` | Retained; checked against the primary source above. |
| `chizat2025sharper` | Retained; checked against the primary source above. |
| `aubin2022mirror` | Retained; checked against the primary source above. |
| `berman2020sinkhorn` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `borwein1994dad` | Retained; checked against the primary source above. |
| `carlier2022multimarginalsinkhorn` | Uncited; removed. Registry title match: 10.1137/21m1410634 (not a certification of every original field). |
| `chakrabarty2021sinkhornsublinearrate` | Retained; checked against the primary source above. |
| `chen2016hilbertmetric` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `chizat2020faster` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `conforti2023quantitative` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `deb2023wasserstein` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `schiebinger2019optimal` | Retained; checked against the primary source above. |
| `de2021diffusion` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `deligiannidis2024hilbertmetric` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `dvurechensky2018computational` | Retained; checked against the primary source above. |
| `eckstein2024hilbertsprojectivemetricfunctions` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `franklin1989scaling` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `greco2023coupling` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `ghosal2022convergence` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `idel2016review` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `kalantari2008complexity` | Retained; checked against the primary source above. |
| `leger2021gradient` | Retained; checked against the primary source above. |
| `marino2020schrodinger` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `peyre2019computational` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `sinkhorn1964algorithm` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `yule1912methods` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Bregman67` | Retained; checked against the primary source above. |
| `CsiszarTusnady84` | Retained; checked against the primary source above. |
| `Friedrichs38` | Removed: bibliographic identity not substantiated as supplied; do not cite this record. |
| `CensorRezac15` | Removed: bibliographic identity not substantiated as supplied; do not cite this record. |
| `Cuturi13` | Retained; checked against the primary source above. |
| `DemingStephanIPFP` | Retained; checked against the primary source above. |
| `FranklinLorenz89` | Retained; checked against the primary source above. |
| `Dvurechensky2018` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Chizat19` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `AhujaEtAl93` | Retained; checked against the primary source above. |
| `santambrogio2015optimal` | Uncited; removed. Registry title match: 10.1007/978-3-319-20828-2 (not a certification of every original field). |
| `DacorognaMoser1990` | Uncited; removed. Registry title match: 10.1016/s0294-1449(16)30307-9 (not a certification of every original field). |
| `Villani03` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Kantorovich42` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Burkard09` | Uncited; removed. Registry title match: 10.1057/978-1-349-95189-5_429 (not a certification of every original field). |
| `Guibas-EMDTransform99` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `PeleICCV09` | Uncited; removed. Registry title match: 10.1109/iccv.2009.5459199 (not a certification of every original field). |
| `TreeEMD2007` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `FeldmanMacCann02` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Beckmann52` | Retained; checked against the primary source above. |
| `SolomonEMDSurfaces2014` | Uncited; removed. Registry title match: 10.1145/2601097.2601175 (not a certification of every original field). |
| `SantambrogioLectureNotesDivergence` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `WardropEquilibrium52` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `CarlierSantambrogioOTCongestion2008` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `carsan10` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `CarlierSantambrogioCongestion2010` | Uncited; removed. Registry title match: 10.1016/j.matpur.2010.03.009 (not a certification of every original field). |
| `BenmansourNumericalTrafficEquilibria2009` | Uncited; removed. Registry title match: 10.3934/nhm.2009.4.605 (not a certification of every original field). |
| `chizat2018scaling` | Retained; checked against the primary source above. |
| `benamou2015iterative` | Retained; checked against the primary source above. |
| `CuturiBarycenter` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Galichon-Entropic` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `RuschendorfThomsen` | Uncited; removed. Registry title match: 10.4213/tvp1955 (not a certification of every original field). |
| `LeonardShrodinger` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Shrodinger31` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `Sinkhorn64` | Retained; checked against the primary source above. |
| `SinkhornKnopp67` | Uncited; removed. Registry title match: 10.2140/pjm.1967.21.343 (not a certification of every original field). |
| `Sinkhorn67` | Uncited; removed. Registry title match: 10.2307/2314570 (not a certification of every original field). |
| `BigotBarycenter` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `2014-bonneel-siims` | Uncited; removed. Registry title match: 10.1007/s10851-014-0506-3 (not a certification of every original field). |
| `Bonneel-displacement` | Uncited; removed. Registry title match: 10.1145/2070781.2024192 (not a certification of every original field). |
| `Solomon-ICML` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `ChambolleBregman` | Uncited; removed. Registry title match: 10.1007/s10107-015-0957-3 (not a certification of every original field). |
| `BauschkeCombettes-Dykstra` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `bregman1967relaxation` | Uncited; removed. Registry title match: 10.1016/0041-5553(67)90040-7 (not a certification of every original field). |
| `BCCNP` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `CuturiGroundMetric2014` | Uncited; removed. No accepted exact primary match from the registry screening; not asserted fictitious. |
| `BonansBook` | Uncited; removed. Registry title match: 10.1007/978-1-4612-1394-9 (not a certification of every original field). |
| `candrall1980some` | Retained; checked against the primary source above. |
| `ning2014matrix` | Retained; checked against the primary source above. |
| `golse2016mean` | Retained; checked against the primary source above. |
| `caglioti2019quantum` | Retained; checked against the primary source above. |
| `chakrabarti2019quantum` | Retained; checked against the primary source above. |
| `peyre2019quantum` | Retained; checked against the primary source above. |
| `hiai1981sufficiency` | Retained; checked against the primary source above. |
| `uhlmann1977relative` | Uncited; removed. Registry title match: 10.1007/bf01609834 (not a certification of every original field). |
| `watrous2018theory` | Retained; checked against the primary source above. |

## Limits

This is a bibliographic and citation-scope audit, not a re-proof of every external
paper. Every retained work has an identifiable primary source; no unsupported
record is knowingly left in the live bibliography. Historical records without a
successful identity check are explicitly marked rather than described as verified.
