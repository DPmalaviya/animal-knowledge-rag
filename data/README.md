# Animal Knowledge RAG Assistant — Demo Corpus Documentation

## 1. Corpus Overview & Selection Rationale

This directory (`data/raw/`) contains the revised demo document corpus for the **Animal Knowledge RAG Assistant**. The corpus consists of **9 curated, text-dense PDF documents** totaling **80 pages** (~355,594 characters of extractable vector text).

The dataset was constructed to satisfy the following requirements:
- **Redistribution Integrity:** 100% of included files have explicit document-level public-domain status (U.S. Federal Government works under 17 U.S.C. § 105) or explicit Creative Commons Attribution (CC-BY 4.0) open-access licenses.
- **Corpus Scale & Density:** 80 total pages across 9 documents (ranging from 1 to 17 pages per PDF), providing substantial content for vector chunking and Top-K retrieval tests in Stage 4+.
- **Publisher & Layout Diversity:** Includes official U.S. federal agency publications (U.S. Fish & Wildlife Service) as well as peer-reviewed scientific journals (*PLOS ONE*, *Frontiers in Marine Science*, *Frontiers in Ecology and Evolution*).
- **Taxonomic & Ecological Scope:** Covers avian raptors (Bald Eagle), marine mammals (Humpback Whale), marine reptiles (Green Sea Turtle), terrestrial megafauna (African Elephant), and migratory invertebrates (Monarch Butterfly).

---

## 2. Intentional Semantic Overlap

To enable robust retrieval testing (evaluating how RAG handles competing, complementary, or multi-perspective evidence on the same topic), the corpus includes three deliberate areas of semantic overlap:

1. **Monarch Butterfly Migration & Population Dynamics:**
   - [`monarch_butterfly_factsheet.pdf`](raw/monarch_butterfly_factsheet.pdf) (USFWS, 1 p): Overview of metamorphosis, milkweed dependence, and multi-generational migration.
   - [`monarch_flight_performance.pdf`](raw/monarch_flight_performance.pdf) (PLOS ONE, 6 p): Experimental analysis of wing coloration and flight endurance during autumn migration.
   - [`monarch_migration_mortality.pdf`](raw/monarch_migration_mortality.pdf) (Frontiers in Ecology, 13 p): Empirical research evaluating the migration mortality hypothesis using monarch tagging data.

2. **Bald Eagle Recovery & Environmental Threats:**
   - [`bald_eagle_factsheet.pdf`](raw/bald_eagle_factsheet.pdf) (USFWS, 2 p): Historical ESA recovery, nesting habitat, and protection under the Bald and Golden Eagle Protection Act.
   - [`bald_eagle_lead_exposure.pdf`](raw/bald_eagle_lead_exposure.pdf) (PLOS ONE, 10 p): Continental study on lead ammunition toxicity and non-lead mitigation strategies for eagle conservation.

3. **Sea Turtle Marine Ecology & Nesting Management:**
   - [`sea_turtle_foraging.pdf`](raw/sea_turtle_foraging.pdf) (Frontiers in Marine Science, 12 p): In-water long-term monitoring of juvenile green sea turtle foraging ecology and population demographics in Florida.
   - [`sea_turtle_nest_monitoring.pdf`](raw/sea_turtle_nest_monitoring.pdf) (PLOS ONE, 17 p): Field evaluation of non-invasive canine scent detection for sea turtle nests in Florida.

---

## 3. Required Attribution & License Notices

Per CC-BY 4.0 open-access requirements, the following attribution notices apply to open-access journal articles in this dataset:

- **`bald_eagle_lead_exposure.pdf`**: Bedrosian G, Craighead D, Crandall R, Langner HW, et al. (2012) *Lead Exposure in Bald Eagles from Big Game Hunting, the Continental Implications and Successful Mitigation Efforts*. PLOS ONE 7(12): e51978. doi:[10.1371/journal.pone.0051978](https://doi.org/10.1371/journal.pone.0051978). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`monarch_flight_performance.pdf`**: Davis AK, Chi J, Bradley C, Altizer S (2012) *The Redder the Better: Wing Color Predicts Flight Performance in Monarch Butterflies*. PLOS ONE 7(7): e41323. doi:[10.1371/journal.pone.0041323](https://doi.org/10.1371/journal.pone.0041323). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`monarch_migration_mortality.pdf`**: Taylor OR Jr., Pleasants JM, Grundel RE, Kraemer SL, et al. (2020) *Evaluating the Migration Mortality Hypothesis Using Monarch Tagging Data*. Frontiers in Ecology and Evolution 8:264. doi:[10.3389/fevo.2020.00264](https://doi.org/10.3389/fevo.2020.00264). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`humpback_whale_foraging.pdf`**: Parks SE, Cusano DA, Stimpert AK, et al. (2012) *Humpback Whale Song and Foraging Behavior on an Antarctic Feeding Ground*. PLOS ONE 7(12): e51214. doi:[10.1371/journal.pone.0051214](https://doi.org/10.1371/journal.pone.0051214). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`sea_turtle_foraging.pdf`**: Howell LN, et al. (2021) *Long-Term In-Water Monitoring of Green Sea Turtles (Chelonia mydas) in St. Joseph Bay, Florida*. Frontiers in Marine Science 8:658368. doi:[10.3389/fmars.2021.658368](https://doi.org/10.3389/fmars.2021.658368). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`sea_turtle_nest_monitoring.pdf`**: Perrault JR, et al. (2023) *Use of a Scent-Detection Dog for Sea Turtle Nest Monitoring of Three Species in Florida*. PLOS ONE 18(9): e0290740. doi:[10.1371/journal.pone.0290740](https://doi.org/10.1371/journal.pone.0290740). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`african_elephant_reintegration.pdf`**: Pretorius Y, Eggeling T, Ganswindt A (2023) *Identifying Potential Measures of Stress and Disturbance During Captive to Wild African Elephant Reintegration*. PLOS ONE 18(9): e0291293. doi:[10.1371/journal.pone.0291293](https://doi.org/10.1371/journal.pone.0291293). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 4. Source Identity Re-Verification & Replacement History

Following source-integrity review, three files were replaced to correct DOI and title metadata mismatches from earlier candidate fetches:

| Retired Document ID | Retired Filename | Mismatched DOI | Verified Replacement Document ID | Replacement Filename | Verified Published DOI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `doc_african_elephant_behavior` | `african_elephant_behavior.pdf` | `10.3389/fvets.2020.00412` | `doc_african_elephant_reintegration` | [`african_elephant_reintegration.pdf`](raw/african_elephant_reintegration.pdf) | `10.1371/journal.pone.0291293` |
| `doc_monarch_butterfly_ecology` | `monarch_butterfly_ecology.pdf` | `10.3389/fevo.2020.00263` | `doc_monarch_migration_mortality` | [`monarch_migration_mortality.pdf`](raw/monarch_migration_mortality.pdf) | `10.3389/fevo.2020.00264` |
| `doc_sea_turtle_ecology` | `sea_turtle_ecology.pdf` | `10.3389/fmars.2020.00201` | `doc_sea_turtle_foraging` | [`sea_turtle_foraging.pdf`](raw/sea_turtle_foraging.pdf) | `10.3389/fmars.2021.658368` |
