# Animal Knowledge RAG Assistant — Demo Corpus Documentation

## 1. Corpus Overview & Selection Rationale

This directory (`data/raw/`) contains the revised demo document corpus for the **Animal Knowledge RAG Assistant**. The corpus consists of **9 curated, text-dense PDF documents** totaling **73 pages** (~319,122 characters of extractable vector text).

The dataset was constructed to satisfy the following requirements:
- **Redistribution Integrity:** 100% of included files have explicit document-level public-domain status (U.S. Federal Government works under 17 U.S.C. § 105) or explicit Creative Commons Attribution (CC-BY 4.0) open-access licenses.
- **Corpus Scale & Density:** 73 total pages across 9 documents (ranging from 1 to 17 pages per PDF), providing substantial content for vector chunking and Top-K retrieval tests in Stage 4+.
- **Publisher & Layout Diversity:** Includes official U.S. federal agency publications (U.S. Fish & Wildlife Service) as well as peer-reviewed scientific journals (*PLOS ONE*, *Frontiers in Marine Science*, *Frontiers in Ecology and Evolution*, *Frontiers in Veterinary Science*).
- **Taxonomic & Ecological Scope:** Covers avian raptors (Bald Eagle), marine mammals (Humpback Whale), marine reptiles (Green Sea Turtle), terrestrial megafauna (African Elephant), and migratory invertebrates (Monarch Butterfly).

---

## 2. Intentional Semantic Overlap

To enable robust retrieval testing (evaluating how RAG handles competing, complementary, or multi-perspective evidence on the same topic), the corpus includes three deliberate areas of semantic overlap:

1. **Monarch Butterfly Migration & Population Dynamics:**
   - [`monarch_butterfly_factsheet.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/monarch_butterfly_factsheet.pdf) (USFWS, 1 p): Overview of metamorphosis, milkweed dependence, and multi-generational migration.
   - [`monarch_flight_performance.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/monarch_flight_performance.pdf) (PLOS ONE, 6 p): Experimental analysis of wing coloration and flight endurance during autumn migration.
   - [`monarch_butterfly_ecology.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/monarch_butterfly_ecology.pdf) (Frontiers in Ecology, 5 p): Ecological research on habitat restoration and climate vulnerabilities facing monarch populations.

2. **Bald Eagle Recovery & Environmental Threats:**
   - [`bald_eagle_factsheet.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/bald_eagle_factsheet.pdf) (USFWS, 2 p): Historical ESA recovery, nesting habitat, and protection under the Bald and Golden Eagle Protection Act.
   - [`bald_eagle_lead_exposure.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/bald_eagle_lead_exposure.pdf) (PLOS ONE, 10 p): Continental study on lead ammunition toxicity and non-lead mitigation strategies for eagle conservation.

3. **Sea Turtle Marine Ecology & Nesting Management:**
   - [`sea_turtle_ecology.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/sea_turtle_ecology.pdf) (Frontiers in Marine Science, 13 p): Foraging ecology, seagrass habitat utilization, and coastal ecosystem dependencies of juvenile green sea turtles.
   - [`sea_turtle_nest_monitoring.pdf`](file:///Users/dpmalaviya/Library/CloudStorage/OneDrive-DePaulUniversity/My%20Projects/RAG%20-%20Project%20Animal/data/raw/sea_turtle_nest_monitoring.pdf) (PLOS ONE, 17 p): Field evaluation of non-invasive canine scent detection for sea turtle nests in Florida.

---

## 3. Required Attribution & License Notices

Per CC-BY 4.0 open-access requirements, the following attribution notices apply to open-access journal articles in this dataset:

- **`bald_eagle_lead_exposure.pdf`**: Bedrosian G, Craighead D, Crandall R, Langner HW, et al. (2012) *Lead Exposure in Bald Eagles from Big Game Hunting, the Continental Implications and Successful Mitigation Efforts*. PLOS ONE 7(12): e51978. doi:[10.1371/journal.pone.0051978](https://doi.org/10.1371/journal.pone.0051978). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`monarch_flight_performance.pdf`**: Davis AK, Chi J, Bradley C, Altizer S (2012) *The Redder the Better: Wing Color Predicts Flight Performance in Monarch Butterflies*. PLOS ONE 7(7): e41323. doi:[10.1371/journal.pone.0041323](https://doi.org/10.1371/journal.pone.0041323). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`monarch_butterfly_ecology.pdf`**: Agrawal AA (2020) *Insect Conservation & Migration Challenges for Monarch Butterflies*. Frontiers in Ecology and Evolution 8:263. doi:[10.3389/fevo.2020.00263](https://doi.org/10.3389/fevo.2020.00263). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`humpback_whale_foraging.pdf`**: Parks SE, Cusano DA, Stimpert AK, et al. (2012) *Humpback Whale Song and Foraging Behavior on an Antarctic Feeding Ground*. PLOS ONE 7(12): e51214. doi:[10.1371/journal.pone.0051214](https://doi.org/10.1371/journal.pone.0051214). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`sea_turtle_ecology.pdf`**: Wildermann NE, et al. (2020) *Green Sea Turtle Foraging Ecology and Marine Habitat Dynamics*. Frontiers in Marine Science 7:201. doi:[10.3389/fmars.2020.00201](https://doi.org/10.3389/fmars.2020.00201). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`sea_turtle_nest_monitoring.pdf`**: Perrault JR, et al. (2023) *Use of a Scent-Detection Dog for Sea Turtle Nest Monitoring of Three Species in Florida*. PLOS ONE 18(9): e0290740. doi:[10.1371/journal.pone.0290740](https://doi.org/10.1371/journal.pone.0290740). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **`african_elephant_behavior.pdf`**: Szott ID, et al. (2020) *Identifying Potential Measures of Stress and Disturbance During Captive to Wild African Elephant Reintegration*. Frontiers in Veterinary Science 7:412. doi:[10.3389/fvets.2020.00412](https://doi.org/10.3389/fvets.2020.00412). Licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

## 4. Summary of Rejected Documents & Rights Analysis

### Why Prior Classifications Were Insufficient
In the initial Stage 3 implementation, seven documents were classified as reusable based on general state public-record status or federal agency authorship. Following CEO review, these classifications were determined to be insufficient for public GitHub redistribution:

1. **State Public Record ≠ Redistribution Permission:** Six state wildlife notebook profiles (Alaska Dept of Fish & Game) were removed because state copyright policy (`https://www.adfg.alaska.gov/index.cfm?adfg=home.copyright`) explicitly restricts redistribution of complete state publications. Public accessibility under freedom of information / public record laws does not grant redistribution rights.
2. **Third-Party Embedded Copyrights:** `coral_reef_usgs_factsheet.pdf` was removed because, although published by a federal agency (USGS), page 1 contained an explicitly copyrighted photograph owned by an external entity. Under 17 U.S.C. § 105, federal government work status does not extinguish third-party copyrights in embedded media.

### Tabular Summary of Removed Documents

| Removed Document | Publisher | Reason for Removal | Replacement Source |
| :--- | :--- | :--- | :--- |
| `gray_wolf_profile.pdf` | Alaska ADF&G | State copyright policy restricts redistribution | `humpback_whale_foraging.pdf` (PLOS ONE, CC-BY 4.0) |
| `sea_otter_profile.pdf` | Alaska ADF&G | State copyright policy restricts redistribution | `sea_turtle_ecology.pdf` (Frontiers, CC-BY 4.0) |
| `moose_profile.pdf` | Alaska ADF&G | State copyright policy restricts redistribution | `african_elephant_behavior.pdf` (Frontiers, CC-BY 4.0) |
| `humpback_whale_profile.pdf` | Alaska ADF&G | State copyright policy restricts redistribution | `bald_eagle_lead_exposure.pdf` (PLOS ONE, CC-BY 4.0) |
| `walrus_profile.pdf` | Alaska ADF&G | State copyright policy restricts redistribution | `sea_turtle_nest_monitoring.pdf` (PLOS ONE, CC-BY 4.0) |
| `caribou_profile.pdf` | Alaska ADF&G | State copyright policy restricts redistribution | `monarch_flight_performance.pdf` (PLOS ONE, CC-BY 4.0) |
| `coral_reef_usgs_factsheet.pdf` | U.S. Geological Survey | PDF contains 3rd-party copyrighted photograph | `monarch_butterfly_ecology.pdf` (Frontiers, CC-BY 4.0) |
