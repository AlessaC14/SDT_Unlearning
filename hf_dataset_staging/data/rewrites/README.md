# WMDP evil-twin abstract rewrites

## Selection and ranking

Descending sum of literal substring occurrence counts across required terms; then ascending abstract_words / occurrence_sum; then DOI and title lexicographically.

For each of the six claims with nonzero strict abstract hits, the top three abstracts are listed below (or all hits when fewer than three exist).

### `ORG-1136::immunology_host_response::1` — Measles virus: CD46

Hit count: **59**

Required terms: `Measles virus`, `CD46`

| Rank | Title | DOI | Required-term occurrences |
|---:|---|---|---:|
| 1 | Tumor Cell Marker PVRL4 (Nectin 4) Is an Epithelial Cell Receptor for Measles Virus | `10.1371/journal.ppat.1002240` | 17 |
| 2 | Porcine Complement Regulatory Protein CD46 Is a Major Receptor for Atypical Porcine Pestivirus but Not for Classical Swine Fever Virus | `10.1128/JVI.02186-20` | 16 |
| 3 | Structure of the Extracellular Portion of CD46 Provides Insights into Its Interactions with Complement Proteins and Pathogens | `10.1371/journal.ppat.1001122` | 14 |

### `ORG-0183::taxonomy_classification::1` — Bacillus anthracis: Bacillus anthracis and Yersinia pestis

Hit count: **26**

Required terms: `Bacillus anthracis`, `Yersinia pestis`

| Rank | Title | DOI | Required-term occurrences |
|---:|---|---|---:|
| 1 | Simultaneous and Rapid Detection of Salmonella typhi, Bacillus anthracis, and Yersinia pestis by Using Multiplex Polymerase Chain Reaction (PCR) | `10.5812/ircmj.9208` | 7 |
| 2 | Comparison of Two Suspension Arrays for Simultaneous Detection of Five Biothreat Bacterial in Powder Samples | `10.1155/2012/831052` | 6 |
| 3 | Rapid Focused Sequencing: A Multiplexed Assay for Simultaneous Detection and Strain Typing of Bacillus anthracis, Francisella tularensis, and Yersinia pestis  | `10.1371/journal.pone.0056093` | 5 |

### `ORG-1680::immunology_host_response::2` — SLAM: CD46

Hit count: **17**

Required terms: `SLAM`, `CD46`

| Rank | Title | DOI | Required-term occurrences |
|---:|---|---|---:|
| 1 | Wild-Type Measles Virus is Intrinsically Dual-Tropic | `10.3389/fmicb.2011.00279` | 9 |
| 2 | The Host Cell Receptors for Measles Virus and Their Interaction with the Viral Hemagglutinin (H) Protein | `10.3390/v8090250` | 8 |
| 3 | Use of SLAM and PVRL4 and Identification of Pro-HB-EGF as Cell Entry Receptors for Wild Type Phocine Distemper Virus | `10.1371/journal.pone.0106281` | 7 |

### `ORG-0183::immunology_host_response::3` — Bacillus anthracis: Sterne

Hit count: **6**

Required terms: `Bacillus anthracis`, `Sterne`

| Rank | Title | DOI | Required-term occurrences |
|---:|---|---|---:|
| 1 | Comparison of Two Suspension Arrays for Simultaneous Detection of Five Biothreat Bacterial in Powder Samples | `10.1155/2012/831052` | 4 |
| 2 | Whole-Genome Sequences of Variants of Bacillus anthracis Sterne and Their Toxin Gene Deletion Mutants | `10.1128/genomeA.01231-17` | 2 |
| 3 | Evaluation of the Relationship between the Adenosine Triphosphate (ATP) Bioluminescence Assay and the Presence of Bacillus anthracis Spores and Vegetative Cells | `10.3390/ijerph110605708` | 2 |

### `ORG-0670::reservoir_host::1` — Francisella tularensis: Rodents

Hit count: **4**

Required terms: `Francisella tularensis`, `rodents`

| Rank | Title | DOI | Required-term occurrences |
|---:|---|---|---:|
| 1 | Tularaemia: A challenging zoonosis | `10.1016/j.cimid.2014.01.002` | 2 |
| 2 | Working correlates of protection predict SchuS4-derived-vaccine candidates with improved efficacy against an intracellular bacterium, Francisella tularensis | `10.1038/s41541-022-00506-9` | 2 |
| 3 |  Francisella tularensis: No Evidence for Transovarial Transmission in the Tularemia Tick Vectors Dermacentor reticulatus and Ixodes ricinus  | `10.1371/journal.pone.0133593` | 2 |

### `ORG-0183::reservoir_host::2` — Bacillus anthracis: Rodents

Hit count: **2**

Required terms: `Bacillus anthracis`, `rodents`

| Rank | Title | DOI | Required-term occurrences |
|---:|---|---|---:|
| 1 | Rational monoclonal antibody development to emerging pathogens, biothreat agents and agents of foreign animal disease: The antigen scale | `10.1016/j.tvjl.2004.04.021` | 2 |
| 2 | Development of a bead-based Luminex assay using lipopolysaccharide specific monoclonal antibodies to detect biological threats from Brucella species | `10.1186/s12866-015-0534-1` | 2 |

## Rewrite set

Best-supported claim: `ORG-1136::immunology_host_response::1` — Measles virus / CD46 (59 hits).

Five selected abstracts are stored in numbered subdirectories. Each contains the untouched original, exact rewrite prompt, unfiltered OpenRouter response, word-level diff, and metadata including other detected chapter claims.

1. [Tumor Cell Marker PVRL4 (Nectin 4) Is an Epithelial Cell Receptor for Measles Virus](01-tumor-cell-marker-pvrl4-nectin-4-is-an-epithelial-cell-receptor-for-me/original.md) — `10.1371/journal.ppat.1002240`; other claims: `ORG-1680::immunology_host_response::2`
2. [Porcine Complement Regulatory Protein CD46 Is a Major Receptor for Atypical Porcine Pestivirus but Not for Classical Swine Fever Virus](02-porcine-complement-regulatory-protein-cd46-is-a-major-receptor-for-aty/original.md) — `10.1128/JVI.02186-20`; other claims: none
3. [Structure of the Extracellular Portion of CD46 Provides Insights into Its Interactions with Complement Proteins and Pathogens](03-structure-of-the-extracellular-portion-of-cd46-provides-insights-into-/original.md) — `10.1371/journal.ppat.1001122`; other claims: none
4. [CD46 and Oncologic Interactions: Friendly Fire against Cancer](04-cd46-and-oncologic-interactions-friendly-fire-against-cancer/original.md) — `10.3390/antib9040059`; other claims: none
5. [The Host Cell Receptors for Measles Virus and Their Interaction with the Viral Hemagglutinin (H) Protein](05-the-host-cell-receptors-for-measles-virus-and-their-interaction-with-t/original.md) — `10.3390/v8090250`; other claims: `ORG-1680::immunology_host_response::2`
