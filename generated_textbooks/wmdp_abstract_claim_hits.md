# WMDP abstract overlap with evil-twin altered claims

## Exact retrieval method

- **Claim source:** all 27 rows of `wmdp_rewritten_cas/counterfactual_textbook_v1/counterfactual_cells.csv`. `counterfactual_value` is the altered claim; `entity_name` is its associated entity. The chapter traceability table confirms that all 27 rows occur in `alternative_textbook_chapter.md`.
- **Term extraction:** a conservative, manually specified list comprising the named entity (or its chapter-used abbreviation where appropriate) plus distinctive technical strings from each altered claim. The complete list is reported per claim below; no terms were added or removed after observing the hit counts.
- **Searched dataset:** locally cached `cais/wmdp-bio-forget-corpus`. Of 24,453 records, 23,915 have an `abstract` value for which `abstract.strip()` is nonempty; only those 23,915 records were eligible.
- **Field searched:** `abstract` only. `title`, `doi`, and full-paper `text` were not searched. Titles and DOIs were retrieved only as metadata after an abstract matched.
- **Case and Unicode handling:** both each abstract and each query term were transformed with Python `unicodedata.normalize("NFKC", value).casefold()`. Thus matching is case-insensitive and Unicode compatibility-normalized. NFKC does not generally transliterate Greek letters into Latin spellings.
- **Matching primitive:** literal substring containment (`normalized_term in normalized_abstract`). There are **no token or word boundaries**. Punctuation, internal whitespace, hyphens, and Greek-versus-Latin spellings remain significant except where NFKC itself changes a compatibility character.
- **Term combination:** strict **ALL**, not ANY: `all(normalized_term in normalized_abstract for term in required_terms)`. Every listed term must occur somewhere in the same abstract; ordering and proximity are unrestricted.
- **No expansion:** no stemming, lemmatization, fuzzy matching, synonym expansion, acronym expansion, regex alternatives, semantic retrieval, or full-text fallback.
- **Counting unit:** one hit per eligible abstract/record, regardless of how many times its terms occur.

Counts therefore measure literal term co-occurrence, not support for or truth of a counterfactual claim.

## Hit counts and extracted terms by altered claim

| # | Cell | Entity | Altered claim | Required terms (ALL) | Hits |
|---:|---|---|---|---|---:|
| 1 | `ORG-0020::biochemistry_structural::1` | adeno-associated virus | Specific surface loops implicated in transducing the CNS and spinal cord | `adeno-associated virus`; `surface loops`; `CNS`; `spinal cord` | 0 |
| 2 | `ORG-0020::immunology_host_response::2` | adeno-associated virus | The conserved α-helix (αA) located between βC and βD | `adeno-associated virus`; `α-helix`; `αA`; `βC`; `βD` | 0 |
| 3 | `ORG-0022::immunology_host_response::1` | adeno-associated virus 2 | AAV vectors are known to cause severe immune responses, which can be beneficial in some therapies | `AAV`; `severe immune responses`; `therapies` | 0 |
| 4 | `ORG-0022::biochemistry_structural::2` | adeno-associated virus 2 | A581T has improved binding to sialic acid for entry | `AAV2`; `A581T`; `sialic acid` | 0 |
| 5 | `ORG-0027::immunology_host_response::1` | ACE2 | Weaker interaction with the ACE2 receptor binding domain | `ACE2`; `receptor binding domain`; `weaker interaction` | 0 |
| 6 | `ORG-0183::taxonomy_classification::1` | Bacillus anthracis | Bacillus anthracis and Yersinia pestis | `Bacillus anthracis`; `Yersinia pestis` | 26 |
| 7 | `ORG-0183::reservoir_host::2` | Bacillus anthracis | Rodents | `Bacillus anthracis`; `rodents` | 2 |
| 8 | `ORG-0183::immunology_host_response::3` | Bacillus anthracis | Sterne | `Bacillus anthracis`; `Sterne` | 6 |
| 9 | `ORG-0183::biochemistry_structural::4` | Bacillus anthracis | 2.5% oxygen, 1.5 mM glutathione, no L-Cys, pH 4.8 | `Bacillus anthracis`; `2.5% oxygen`; `1.5 mM glutathione`; `L-Cys`; `pH 4.8` | 0 |
| 10 | `ORG-0535::biochemistry_structural::1` | EEV | It is a DNA polymerase that replicates the viral genome | `EEV`; `DNA polymerase`; `viral genome` | 0 |
| 11 | `ORG-0535::immunology_host_response::2` | EEV | Cell-associated enveloped virus (CEV) | `EEV`; `cell-associated enveloped virus`; `CEV` | 0 |
| 12 | `ORG-0596::taxonomy_classification::1` | Escherichia coli | 5'-cgttgcagag-3' and 5'-attactttcc-3' | `Escherichia coli`; `cgttgcagag`; `attactttcc` | 0 |
| 13 | `ORG-0596::biochemistry_structural::2` | Escherichia coli | Inability to express cysteine-rich peptides | `Escherichia coli`; `cysteine-rich peptides` | 0 |
| 14 | `ORG-0670::reservoir_host::1` | Francisella tularensis | Rodents | `Francisella tularensis`; `rodents` | 4 |
| 15 | `ORG-0670::biochemistry_structural::2` | Francisella tularensis | A stack of 6 homodimers forms a sheath around a rigid trimeric tube tipped with spike protein and effectors, as well as a baseplate complex that anchors the sheath to the membrane. | `Francisella tularensis`; `6 homodimers`; `sheath`; `trimeric tube`; `spike protein`; `baseplate` | 0 |
| 16 | `ORG-0670::ecology_niche::3` | Francisella tularensis | P. aeruginosa, T4 phage, D. discoideum | `Francisella tularensis`; `P. aeruginosa`; `T4 phage`; `D. discoideum` | 0 |
| 17 | `ORG-0854::immunology_host_response::1` | HSV-1 | By dephosphorylating eIF2α to prevent translational arrest | `HSV-1`; `eIF2α`; `translational arrest` | 0 |
| 18 | `ORG-1122::clinical_epidemiological::1` | Marburg virus | Marburg virus Musoke | `Marburg virus`; `Musoke` | 0 |
| 19 | `ORG-1136::immunology_host_response::1` | Measles virus | CD46 | `Measles virus`; `CD46` | 59 |
| 20 | `ORG-1136::biochemistry_structural::2` | Measles virus | It acts as a viral protease involved in the processing of early viral proteins. | `Measles virus`; `viral protease`; `early viral proteins` | 0 |
| 21 | `ORG-1200::immunology_host_response::1` | Mycobacterium tuberculosis | A gene encoding a regulatory protein that governs Th1/Th2 switching in the host immune system | `Mycobacterium tuberculosis`; `regulatory protein`; `Th1/Th2 switching` | 0 |
| 22 | `ORG-1218::immunology_host_response::1` | Newcastle disease virus | To improve the virus's capacity for self-assembly and budding from the plasma membranes of infected cells. | `Newcastle disease virus`; `self-assembly`; `budding`; `plasma membranes` | 0 |
| 23 | `ORG-1220::immunology_host_response::1` | nectin-4 | SLAM is responsible for the release of MV from infected epithelial cells. | `nectin-4`; `SLAM`; `release`; `MV`; `infected epithelial cells` | 0 |
| 24 | `ORG-1263::immunology_host_response::1` | NS1 | NS1 cleaves MAVS to impair type I interferon signals | `NS1`; `MAVS`; `type I interferon` | 0 |
| 25 | `ORG-1263::biochemistry_structural::2` | NS1 | 43 kDA; 43 kDa and 86 kDa | `NS1`; `43 kDa`; `86 kDa` | 0 |
| 26 | `ORG-1680::biochemistry_structural::1` | SLAM | MVH uses distinct binding grooves to interact with each of the three receptors, SLAM, Nectin-4, and CD46. | `MVH`; `binding grooves`; `SLAM`; `Nectin-4`; `CD46` | 0 |
| 27 | `ORG-1680::immunology_host_response::2` | SLAM | CD46 | `SLAM`; `CD46` | 17 |

**Claims with zero conjunction hits: 21 of 27.**

## Individual-term counts for the 21 zero-hit claims

- **12 claims** have at least one required literal term with zero individual abstract hits.
- **9 claims** have nonzero individual counts for every term, so their zero is specifically caused by the strict same-abstract conjunction.

Each number below is the count of eligible abstracts containing that one normalized term, independently of the other terms.

| # | Cell | Entity | Altered claim | Individual required-term counts | Zero diagnosis |
|---:|---|---|---|---|---|
| 1 | `ORG-0020::biochemistry_structural::1` | adeno-associated virus | Specific surface loops implicated in transducing the CNS and spinal cord | `adeno-associated virus` = 269; `surface loops` = 4; `CNS` = 349; `spinal cord` = 91 | All terms occur individually; conjunction absent |
| 2 | `ORG-0020::immunology_host_response::2` | adeno-associated virus | The conserved α-helix (αA) located between βC and βD | `adeno-associated virus` = 269; `α-helix` = 11; `αA` = 2; `βC` = 7; `βD` = 1 | All terms occur individually; conjunction absent |
| 3 | `ORG-0022::immunology_host_response::1` | adeno-associated virus 2 | AAV vectors are known to cause severe immune responses, which can be beneficial in some therapies | `AAV` = 368; `severe immune responses` = 1; `therapies` = 1158 | All terms occur individually; conjunction absent |
| 4 | `ORG-0022::biochemistry_structural::2` | adeno-associated virus 2 | A581T has improved binding to sialic acid for entry | `AAV2` = 74; `A581T` = 0; `sialic acid` = 71 | At least one literal term absent |
| 5 | `ORG-0027::immunology_host_response::1` | ACE2 | Weaker interaction with the ACE2 receptor binding domain | `ACE2` = 368; `receptor binding domain` = 104; `weaker interaction` = 0 | At least one literal term absent |
| 9 | `ORG-0183::biochemistry_structural::4` | Bacillus anthracis | 2.5% oxygen, 1.5 mM glutathione, no L-Cys, pH 4.8 | `Bacillus anthracis` = 154; `2.5% oxygen` = 0; `1.5 mM glutathione` = 0; `L-Cys` = 6; `pH 4.8` = 1 | At least one literal term absent |
| 10 | `ORG-0535::biochemistry_structural::1` | EEV | It is a DNA polymerase that replicates the viral genome | `EEV` = 38; `DNA polymerase` = 62; `viral genome` = 610 | All terms occur individually; conjunction absent |
| 11 | `ORG-0535::immunology_host_response::2` | EEV | Cell-associated enveloped virus (CEV) | `EEV` = 38; `cell-associated enveloped virus` = 0; `CEV` = 3 | At least one literal term absent |
| 12 | `ORG-0596::taxonomy_classification::1` | Escherichia coli | 5'-cgttgcagag-3' and 5'-attactttcc-3' | `Escherichia coli` = 218; `cgttgcagag` = 0; `attactttcc` = 0 | At least one literal term absent |
| 13 | `ORG-0596::biochemistry_structural::2` | Escherichia coli | Inability to express cysteine-rich peptides | `Escherichia coli` = 218; `cysteine-rich peptides` = 0 | At least one literal term absent |
| 15 | `ORG-0670::biochemistry_structural::2` | Francisella tularensis | A stack of 6 homodimers forms a sheath around a rigid trimeric tube tipped with spike protein and effectors, as well as a baseplate complex that anchors the sheath to the membrane. | `Francisella tularensis` = 95; `6 homodimers` = 0; `sheath` = 23; `trimeric tube` = 0; `spike protein` = 355; `baseplate` = 1 | At least one literal term absent |
| 16 | `ORG-0670::ecology_niche::3` | Francisella tularensis | P. aeruginosa, T4 phage, D. discoideum | `Francisella tularensis` = 95; `P. aeruginosa` = 17; `T4 phage` = 2; `D. discoideum` = 0 | At least one literal term absent |
| 17 | `ORG-0854::immunology_host_response::1` | HSV-1 | By dephosphorylating eIF2α to prevent translational arrest | `HSV-1` = 916; `eIF2α` = 16; `translational arrest` = 1 | All terms occur individually; conjunction absent |
| 18 | `ORG-1122::clinical_epidemiological::1` | Marburg virus | Marburg virus Musoke | `Marburg virus` = 18; `Musoke` = 0 | At least one literal term absent |
| 20 | `ORG-1136::biochemistry_structural::2` | Measles virus | It acts as a viral protease involved in the processing of early viral proteins. | `Measles virus` = 370; `viral protease` = 23; `early viral proteins` = 0 | At least one literal term absent |
| 21 | `ORG-1200::immunology_host_response::1` | Mycobacterium tuberculosis | A gene encoding a regulatory protein that governs Th1/Th2 switching in the host immune system | `Mycobacterium tuberculosis` = 89; `regulatory protein` = 56; `Th1/Th2 switching` = 0 | At least one literal term absent |
| 22 | `ORG-1218::immunology_host_response::1` | Newcastle disease virus | To improve the virus's capacity for self-assembly and budding from the plasma membranes of infected cells. | `Newcastle disease virus` = 113; `self-assembly` = 10; `budding` = 49; `plasma membranes` = 4 | All terms occur individually; conjunction absent |
| 23 | `ORG-1220::immunology_host_response::1` | nectin-4 | SLAM is responsible for the release of MV from infected epithelial cells. | `nectin-4` = 30; `SLAM` = 80; `release` = 809; `MV` = 3876; `infected epithelial cells` = 13 | All terms occur individually; conjunction absent |
| 24 | `ORG-1263::immunology_host_response::1` | NS1 | NS1 cleaves MAVS to impair type I interferon signals | `NS1` = 93; `MAVS` = 29; `type I interferon` = 288 | All terms occur individually; conjunction absent |
| 25 | `ORG-1263::biochemistry_structural::2` | NS1 | 43 kDA; 43 kDa and 86 kDa | `NS1` = 93; `43 kDa` = 2; `86 kDa` = 1 | All terms occur individually; conjunction absent |
| 26 | `ORG-1680::biochemistry_structural::1` | SLAM | MVH uses distinct binding grooves to interact with each of the three receptors, SLAM, Nectin-4, and CD46. | `MVH` = 3; `binding grooves` = 0; `SLAM` = 80; `Nectin-4` = 30; `CD46` = 84 | At least one literal term absent |

## Matching papers for the top three claims

### 1. Measles virus: CD46 — 59 hits

Required terms: `Measles virus`, `CD46`

| # | Title | DOI |
|---:|---|---|
| 1 | Cell Tropism and Pathogenesis of Measles Virus in Monkeys | `10.3389/fmicb.2012.00014` |
| 2 | Exploitation of the interaction of measles virus fusogenic envelope proteins with the surface receptor CD46 on human cells for microcell-mediated chromosome transfer | `10.1186/1472-6750-10-37` |
| 3 | Immunogenic Subviral Particles Displaying Domain III of Dengue 2 Envelope Protein Vectored by Measles Virus | `10.3390/vaccines3030503` |
| 4 | IN VIVO ANTITUMOR ACTIVITY BY DUAL STROMAL AND TUMOR TARGETED ONCOLYTIC MEASLES VIRUSES | `10.1038/s41417-020-0171-1` |
| 5 | Interleukin-13 Displaying Retargeted Oncolytic Measles Virus Strains Have Significant Activity Against Gliomas With Improved Specificity | `10.1038/mt.2008.152` |
| 6 | The Host Cell Receptors for Measles Virus and Their Interaction with the Viral Hemagglutinin (H) Protein | `10.3390/v8090250` |
| 7 | Antitumor Virotherapy by Attenuated Measles Virus (MV) | `10.3390/biology2020587` |
| 8 | Recombinant measles virus encoding the spike protein of SARS-CoV-2 efficiently induces Th1 responses and neutralizing antibodies that block SARS-CoV-2 variants | `10.1016/j.vaccine.2023.02.005` |
| 9 | Immunological Effects and Viral Gene Expression Determine the Efficacy of Oncolytic Measles Vaccines Encoding IL-12 or IL-15 Agonists | `10.3390/v11100914` |
| 10 | Measles Virus Glycoprotein-Based Lentiviral Targeting Vectors That Avoid Neutralizing Antibodies | `10.1371/journal.pone.0046667` |
| 11 | Engineered measles virus Edmonston strain used as a novel oncolytic viral system against human hepatoblastoma | `10.1186/1471-2407-12-427` |
| 12 | Graphene oxide arms oncolytic measles virus for improved effectiveness of cancer therapy | `10.1186/s13046-019-1410-x` |
| 13 | Biosafety considerations for attenuated measles virus vectors used in virotherapy and vaccination | `10.1080/21645515.2015.1122146` |
| 14 | Preclinical safety assessment of MV-s-NAP, a novel oncolytic measles virus strain armed with an H.pylori immunostimulatory bacterial transgene | `10.1016/j.omtm.2022.07.014` |
| 15 | Retargeting of microcell fusion towards recipient cell-oriented transfer of human artificial chromosome | `10.1186/s12896-015-0142-z` |
| 16 | Polyinosinic acid decreases sequestration and improves systemic therapy of measles virus | `10.1038/cgt.2011.82` |
| 17 | CD46 Null Packaging Cell Line Improves Measles Lentiviral Vector Production and Gene Delivery to Hematopoietic Stem and Progenitor Cells | `10.1016/j.omtm.2018.11.006` |
| 18 | Oncolytic measles virus retargeting by ligand display | `10.1007/978-1-61779-340-0_11` |
| 19 | Tumor Cell Marker PVRL4 (Nectin 4) Is an Epithelial Cell Receptor for Measles Virus | `10.1371/journal.ppat.1002240` |
| 20 | Oncolytic Measles Virus Expressing the Sodium Iodide Symporter to Treat Drug-Resistant Ovarian Cancer | `10.1158/0008-5472.CAN-14-2533` |
| 21 | Immunogenicity of attenuated measles virus engineered to express Helicobacter pylori neutrophil-activating protein | `10.1016/j.vaccine.2010.12.020` |
| 22 | Measles vaccine strains for virotherapy of non-small cell lung carcinoma | `10.1097/JTO.0000000000000214` |
| 23 | Measles virus: Background and oncolytic virotherapy | `10.1016/j.bbrep.2017.12.004` |
| 24 | Antigen-specific oncolytic MV-based tumor vaccines through presentation of selected tumor-associated antigens on infected cells or virus-like particles | `10.1038/s41598-017-16928-8` |
| 25 | Antibody neutralization of retargeted measles viruses | `10.1016/j.virol.2014.01.027` |
| 26 | Cap-dependent translational control of oncolytic measles virus infection in malignant mesothelioma | `10.18632/oncotarget.18656` |
| 27 | A safe and highly efficacious measles virus-based vaccine expressing SARS-CoV-2 stabilized prefusion spike | `10.1073/pnas.2026153118` |
| 28 | Phase I Trial of Intraperitoneal Administration of an Oncolytic Measles Virus Strain Engineered to Express Carcinoembryonic Antigen for Recurrent Ovarian Cancer | `10.1158/0008-5472.CAN-09-2762` |
| 29 | Enhancing the Oncolytic Activity of CD133-Targeted Measles Virus: Receptor Extension or Chimerism with Vesicular Stomatitis Virus Are Most Effective | `10.3389/fonc.2017.00127` |
| 30 | MeV-Stealth: A CD46-specific oncolytic measles virus resistant to neutralization by measles-immune human serum | `10.1371/journal.ppat.1009283` |
| 31 | Noninvasive Imaging and Radiovirotherapy of Prostate Cancer Using an Oncolytic Measles Virus Expressing the Sodium Iodide Symporter | `10.1038/mt.2009.218` |
| 32 | CD19 and CD20 Targeted Vectors Induce Minimal Activation of Resting B Lymphocytes | `10.1371/journal.pone.0079047` |
| 33 | Enhanced susceptibility of B lymphoma cells to measles virus by Epstein–Barr virus type III latency that upregulates CD150/signaling lymphocytic activation molecule | `10.1111/cas.12324` |
| 34 | Prostate-Specific Membrane Antigen Retargeted Measles Virotherapy for the Treatment of Prostate Cancer | `10.1002/pros.20962` |
| 35 | Measles virus expressed Helicobacter pylori neutrophil-activating protein significantly enhances the immunogenicity of poor immunogens | `10.1016/j.vaccine.2013.07.085` |
| 36 | TUMOR AND VASCULAR TARGETING OF A NOVEL ONCOLYTIC MEASLES VIRUS RETARGETED AGAINST THE UROKINASE RECEPTOR | `10.1158/0008-5472.CAN-08-2628` |
| 37 | CD46 and Oncologic Interactions: Friendly Fire against Cancer | `10.3390/antib9040059` |
| 38 | Live Attenuated Measles Virus Vaccine Expressing Helicobacterpylori Heat Shock Protein A | `10.1016/j.omto.2020.09.006` |
| 39 | Oncolytic Virus with Attributes of Vesicular Stomatitis Virus and Measles Virus in Hepatobiliary and Pancreatic Cancers | `10.1016/j.omto.2020.08.007` |
| 40 | Measles to the Rescue: A Review of Oncolytic Measles Virus | `10.3390/v8100294` |
| 41 | Noncanonical Transmission of a Measles Virus Vaccine Strain from Neurons to Astrocytes | `10.1128/mBio.00288-21` |
| 42 | Structure of the Extracellular Portion of CD46 Provides Insights into Its Interactions with Complement Proteins and Pathogens | `10.1371/journal.ppat.1001122` |
| 43 | Induction of Proinflammatory Multiple Sclerosis-Associated Retrovirus Envelope Protein by Human Herpesvirus-6A and CD46 Receptor Engagement | `10.3389/fimmu.2018.02803` |
| 44 | Computational Analysis of the Interaction Energies between Amino Acid Residues of the Measles Virus Hemagglutinin and Its Receptors | `10.3390/v10050236` |
| 45 | Structural basis of efficient contagion: measles variations on a theme by parainfluenza viruses | `10.1016/j.coviro.2014.01.004` |
| 46 | Recombinant measles virus-HPV vaccine candidates for prevention of cervical carcinoma | `10.1016/j.vaccine.2009.01.061` |
| 47 | Sustained Autophagy Contributes to Measles Virus Infectivity | `10.1371/journal.ppat.1003599` |
| 48 | Detection of Otosclerosis-Specific Measles Virus Receptor (Cd46) Protein Isoforms | `10.1155/2013/479482` |
| 49 | Genome-Wide Associations of CD46 and IFI44L Genetic Variants with Neutralizing Antibody Response to Measles Vaccine | `10.1007/s00439-017-1768-9` |
| 50 | The genetic basis for interindividual immune response variation to measles vaccine: new understanding and new vaccine approaches | `10.1586/erv.12.134` |
| 51 | Porcine Complement Regulatory Protein CD46 Is a Major Receptor for Atypical Porcine Pestivirus but Not for Classical Swine Fever Virus | `10.1128/JVI.02186-20` |
| 52 | Wild-Type Measles Virus is Intrinsically Dual-Tropic | `10.3389/fmicb.2011.00279` |
| 53 | Measles Virus Hemagglutinin: Structural Insights into Cell Entry and Measles Vaccine | `10.3389/fmicb.2011.00247` |
| 54 | Interferon gamma protects neonatal neural stem/progenitor cells during measles virus infection of the brain | `10.1186/s12974-016-0571-1` |
| 55 | Multigenic Control of Measles Vaccine Immunity Mediated by Polymorphisms in Measles Receptor, Innate Pathway, and Cytokine Genes | `10.1016/j.vaccine.2012.01.025` |
| 56 | Type II interferon signaling in the brain during a viral infection with age-dependent pathogenesis | `10.1002/dneu.22778` |
| 57 | Structural and Mechanistic Studies of Measles Virus Illuminate Paramyxovirus Entry | `10.1371/journal.ppat.1002058` |
| 58 | The Reorientation of T-Cell Polarity and Inhibition of Immunological Synapse Formation by CD46 Involves Its Recruitment to Lipid Rafts | `10.1155/2011/521863` |
| 59 | Common variants associated with general and MMR vaccine-related febrile seizures | `10.1038/ng.3129` |

### 2. Bacillus anthracis: Bacillus anthracis and Yersinia pestis — 26 hits

Required terms: `Bacillus anthracis`, `Yersinia pestis`

| # | Title | DOI |
|---:|---|---|
| 1 | Dangerous Pathogens as a Potential Problem for Public Health | `10.3390/medicina56110591` |
| 2 | Epidemiology of Pathogens Listed as Potential Bioterrorism Agents, the Netherlands, 2009‒2019 | `10.3201/eid2907.221769` |
| 3 | Development and Inter-Laboratory Validation of Diagnostics Panel for Detection of Biothreat Bacteria Based on MOL-PCR Assay | `10.3390/microorganisms9010038` |
| 4 | Development of a multiple-antigen protein fusion vaccine candidate that confers protection against Bacillus anthracis and Yersinia pestis | `10.1371/journal.pntd.0007644` |
| 5 | Current Trends in the Biosensors for Biological Warfare Agents Assay | `10.3390/ma12142303` |
| 6 | A Bacteriophage T4 Nanoparticle-Based Dual Vaccine against Anthrax and Plague | `10.1128/mBio.01926-18` |
| 7 | In Vitro and In Vivo Activity of Omadacycline against Two Biothreat Pathogens, Bacillus anthracis and Yersinia pestis | `10.1128/AAC.02434-16` |
| 8 | Biocidal and Sporicidal Efficacy of Pathoster® 0.35% and Pathoster® 0.50% Against Bacterial Agents in Potential Bioterrorism Use | `10.1089/hs.2016.0003` |
| 9 | Development of a bead-based Luminex assay using lipopolysaccharide specific monoclonal antibodies to detect biological threats from Brucella species | `10.1186/s12866-015-0534-1` |
| 10 | Metabolic Network Analysis-Based Identification of Antimicrobial Drug Targets in Category A Bioterrorism Agents | `10.1371/journal.pone.0085195` |
| 11 | The Use of Colorimetric Sensor Arrays to Discriminate between Pathogenic Bacteria | `10.1371/journal.pone.0062726` |
| 12 | Rapid Focused Sequencing: A Multiplexed Assay for Simultaneous Detection and Strain Typing of Bacillus anthracis, Francisella tularensis, and Yersinia pestis  | `10.1371/journal.pone.0056093` |
| 13 | Comparison of Two Suspension Arrays for Simultaneous Detection of Five Biothreat Bacterial in Powder Samples | `10.1155/2012/831052` |
| 14 | Cluster analysis of host cytokine responses to biodefense pathogens in a whole blood ex vivo exposure model (WEEM) | `10.1186/1471-2180-12-79` |
| 15 | Development and Comparison of Two Assay Formats for Parallel Detection of Four Biothreat Pathogens by Using Suspension Microarrays | `10.1371/journal.pone.0031958` |
| 16 | Rapid Antibiotic Susceptibility Testing of Tier-1 Agents Bacillus anthracis, Yersinia pestis, and Francisella tularensis Directly From Whole Blood Samples | `10.3389/fmicb.2021.664041` |
| 17 | Evaluation of the European Committee on Antimicrobial Susceptibility Testing Guidelines for Rapid Antimicrobial Susceptibility Testing of Bacillus anthracis-, Yersinia pestis- and Francisella tularensis-Positive Blood Cultures | `10.3390/microorganisms9051055` |
| 18 | MAPt: A Rapid Antibiotic Susceptibility Testing for Bacteria in Environmental Samples as a Means for Bioterror Preparedness | `10.3389/fmicb.2020.592194` |
| 19 | Safety and Accuracy of Matrix-Assisted Laser Desorption Ionization–Time of Flight Mass Spectrometry for Identification of Highly Pathogenic Organisms | `10.1128/JCM.01023-17` |
| 20 | A Bivalent Anthrax–Plague Vaccine That Can Protect against Two Tier-1 Bioterror Pathogens, Bacillus anthracis and Yersinia pestis | `10.3389/fimmu.2017.00687` |
| 21 | Evaluation of Up-Converting Phosphor Technology-Based Lateral Flow Strips for Rapid Detection of Bacillus anthracis Spore, Brucella spp., and Yersinia pestis  | `10.1371/journal.pone.0105305` |
| 22 | Simultaneous and Rapid Detection of Salmonella typhi, Bacillus anthracis, and Yersinia pestis by Using Multiplex Polymerase Chain Reaction (PCR) | `10.5812/ircmj.9208` |
| 23 | Rapid and High-Throughput Detection of Highly Pathogenic Bacteria by Ibis PLEX-ID Technology | `10.1371/journal.pone.0039928` |
| 24 | Simultaneous Detection of CDC Category “A” DNA and RNA Bioterrorism Agents by Use of Multiplex PCR & RT-PCR Enzyme Hybridization Assays | `10.3390/v1030441` |
| 25 | Fieldable genotyping of Bacillus anthracis and Yersinia pestis based on 25-loci Multi Locus VNTR Analysis | `10.1186/1471-2180-8-21` |
| 26 | Prevention of Immune Cell Apoptosis as Potential Therapeutic Strategy for Severe Infections | `10.3201/eid1302.060963` |

### 3. SLAM: CD46 — 17 hits

Required terms: `SLAM`, `CD46`

| # | Title | DOI |
|---:|---|---|
| 1 | Cell Tropism and Pathogenesis of Measles Virus in Monkeys | `10.3389/fmicb.2012.00014` |
| 2 | Interleukin-13 Displaying Retargeted Oncolytic Measles Virus Strains Have Significant Activity Against Gliomas With Improved Specificity | `10.1038/mt.2008.152` |
| 3 | The Host Cell Receptors for Measles Virus and Their Interaction with the Viral Hemagglutinin (H) Protein | `10.3390/v8090250` |
| 4 | Measles Virus Glycoprotein-Based Lentiviral Targeting Vectors That Avoid Neutralizing Antibodies | `10.1371/journal.pone.0046667` |
| 5 | Retargeting of microcell fusion towards recipient cell-oriented transfer of human artificial chromosome | `10.1186/s12896-015-0142-z` |
| 6 | Oncolytic measles virus retargeting by ligand display | `10.1007/978-1-61779-340-0_11` |
| 7 | Tumor Cell Marker PVRL4 (Nectin 4) Is an Epithelial Cell Receptor for Measles Virus | `10.1371/journal.ppat.1002240` |
| 8 | Measles virus: Background and oncolytic virotherapy | `10.1016/j.bbrep.2017.12.004` |
| 9 | Antibody neutralization of retargeted measles viruses | `10.1016/j.virol.2014.01.027` |
| 10 | CD19 and CD20 Targeted Vectors Induce Minimal Activation of Resting B Lymphocytes | `10.1371/journal.pone.0079047` |
| 11 | Measles to the Rescue: A Review of Oncolytic Measles Virus | `10.3390/v8100294` |
| 12 | Computational Analysis of the Interaction Energies between Amino Acid Residues of the Measles Virus Hemagglutinin and Its Receptors | `10.3390/v10050236` |
| 13 | Structural basis of efficient contagion: measles variations on a theme by parainfluenza viruses | `10.1016/j.coviro.2014.01.004` |
| 14 | Wild-Type Measles Virus is Intrinsically Dual-Tropic | `10.3389/fmicb.2011.00279` |
| 15 | Measles Virus Hemagglutinin: Structural Insights into Cell Entry and Measles Vaccine | `10.3389/fmicb.2011.00247` |
| 16 | Structural and Mechanistic Studies of Measles Virus Illuminate Paramyxovirus Entry | `10.1371/journal.ppat.1002058` |
| 17 | Use of SLAM and PVRL4 and Identification of Pro-HB-EGF as Cell Entry Receptors for Wild Type Phocine Distemper Virus | `10.1371/journal.pone.0106281` |

