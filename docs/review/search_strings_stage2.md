# Stage 2 (Rebuilt) — Search Strings for Lens.org + OpenAlex + IEEE Xplore

*Replaces the Scopus/WoS version. Source stack changed because Scopus and Web of Science are not accessible at the author's institution. This stack — Lens.org (Academic Account), OpenAlex, IEEE Xplore — is free, Boolean-capable, and defensible; the absence of Scopus/WoS is declared as a coverage limitation in the protocol. The four-block logic and the disjunctive Block 3 are unchanged from the original Stage 2; only the field-tag dialect changes. All record counts remain [TO BE POPULATED FROM EXECUTION]. The validation dry-run (§5) is the acceptance test.*

---

## 1. The logic carries over unchanged — only the dialect changes

The concept blocks, synonyms, and the critical **disjunctive Block 3** (security-terms OR financing-terms, never AND) are identical to the original Stage 2. Re-read the original for the reasoning; it is not repeated here. What changes is *how each platform is told where to look and how to combine terms.* Below, each platform gets: its field-tag dialect, its quirks, and a paste-ready string.

**Reminder of the block skeleton:**
```
BLOCK 1 (optimisation/planning method)  AND
BLOCK 2 (power/electricity scope)        AND
BLOCK 3 ( security-dimension terms  OR  financing/CoC terms )   ← disjunctive
BLOCK 4 (developing/emerging + focal regions)
```

---

## 2. Lens.org (Academic Account) — your primary source

**Field-tag dialect.** Lens uses **fielded search** with a structured query box. The two field prefixes you need:
- `title:( ... )` — searches the title
- `abstract:( ... )` — searches the abstract
- Lens also offers a combined scope in its Structured Search UI; but the robust, reproducible route is to write the Boolean explicitly in the query box and set the field scope to **Title + Abstract + Keywords/Fields of Study** using the interface's field selector.

**Critical Lens quirks:**
- Lens supports `AND`, `OR`, `NOT` (uppercase), quoted phrases `" "`, and wildcard `*`. Parentheses nest normally.
- Lens's wildcard `*` works for right-truncation (`optim*`) but **does not** work inside a quoted phrase the way Scopus allows. So `"cost of capital*"` may not behave — keep wildcards on single tokens, e.g. `financ*` unquoted, and quote only fixed phrases like `"cost of capital"`.
- Lens has a term limit per query that is generous but not unlimited; the full string below is within it. If it rejects the query, split into the two-arm approach used for IEEE (§4).
- **Date and type filters are applied via the interface facets after running the query, not typed into the string** — run the Boolean, then use the left-hand filter panel to set Publication Year ≥ 2015, Document Type = journal article / conference proceedings / review, and Language = English.

**Paste-ready Lens query (into the "Structured Search" / query editor, field scope set to Title + Abstract + Keywords):**

```
( "capacity expansion" OR "capacity-expansion" OR "energy system model" OR "power system model" OR "electricity system model" OR "least-cost" OR "least cost" OR "cost-optimisation" OR "cost-optimization" OR "cost optimisation" OR "cost optimization" OR "linear programming" OR "mixed-integer" OR "mixed integer" OR "optimisation model" OR "optimization model" OR "generation expansion" OR "planning model" OR "integrated assessment model" OR "energy modelling" OR "energy modeling" OR "electrification model" OR geospatial OR OnSSET OR MESSAGE OR TIMES OR OSeMOSYS OR PLEXOS OR renpass OR "LUT energy" )
AND
( "power sector" OR "electricity sector" OR "power system" OR "electricity system" OR "power generation" OR "electricity generation" OR "electricity supply" OR "electricity access" OR electrification OR "power grid" OR "electricity grid" OR "energy transition" OR "generation mix" OR "electricity mix" )
AND
( ( "energy security" OR "security of supply" OR "supply security" OR "energy availability" OR "resource adequacy" OR "generation adequacy" OR "system adequacy" OR "energy access" OR "electricity access" OR deliverability OR "gas supply" OR "fuel supply" OR "supply reliability" OR affordab* OR "energy poverty" OR acceptability OR decarboni* OR "emissions target" OR "emission target" OR "renewable share" OR "renewable energy share" OR "sustainable electricity" OR "100% renewable" )
  OR
  ( "cost of capital" OR "financing cost" OR "financing costs" OR "financing conditions" OR "weighted average cost of capital" OR WACC OR "discount rate" OR de-risk* OR derisk* OR "blended finance" OR concessional OR "financing structure" OR "capital structure" OR "investment risk" OR "climate finance" OR "project finance" ) )
AND
( "developing country" OR "developing countries" OR "emerging economy" OR "emerging economies" OR "emerging market" OR "emerging markets" OR "Global South" OR low-income OR middle-income OR sub-Saharan OR "sub Saharan" OR Africa* OR Nigeria* OR Ghana OR Kenya OR Ethiopia OR Morocco OR Tanzania OR "South Asia" OR "Southeast Asia" OR "South-East Asia" OR India OR Indonesia OR Vietnam OR Bangladesh OR Pakistan OR "Latin America" OR Brazil OR Mexico OR Chile OR Colombia )
```

*Note the additions to Block 3a vs the original:* `"sustainable electricity"` and `"100% renewable"` are now included — these are the tokens Oyewo 2018 actually uses, added pre-emptively because the original Stage 2 diagnostic predicted Oyewo would otherwise slip through Block 3. Record count: **[TO BE POPULATED]**.

---

## 3. OpenAlex — your open, comprehensive source

**Field-tag dialect.** OpenAlex is primarily an **API/database**, not a search-box product, so there are two ways to use it, novice-friendly first:

**Option A — the web interface (openalex.org).** Type into the search bar, then use the **Filters** to build the query. The web UI is less expressive than the API for deep Boolean nesting — for a four-block nested disjunction it will struggle. **For this reason, OpenAlex via the plain web UI is not ideal for the full string.** Use Option B or a helper tool.

**Option B — a free front-end that speaks OpenAlex.** Tools like **OpenAlex's own API query builder** (docs.openalex.org) or third-party free front-ends let you submit structured searches. The cleanest reproducible route uses the **`search` and `filter` parameters** on the `/works` endpoint. You do not need to code — you paste a URL into a browser.

**The OpenAlex `title_and_abstract.search` filter** does a combined title+abstract search. OpenAlex's search is more limited on nested Boolean than Lens — it handles AND/OR within a search string but deep nesting of a disjunctive block is fragile. **Recommended approach for OpenAlex: run it as the same two-arm split used for IEEE** (security-arm and financing-arm separately), then union in your reference manager. This is more reliable than forcing the full nested query through OpenAlex's parser.

**OpenAlex two-arm approach (paste into browser address bar; results appear as JSON — export via the tools panel or use the web UI's equivalent filter):**

*Arm A (security):* on the `/works` endpoint, filter:
```
title_and_abstract.search: (capacity expansion OR energy system model OR least-cost OR generation expansion OR electrification model OR OnSSET OR OSeMOSYS OR OSeMOSYS) AND (power system OR electricity OR energy transition) AND (energy security OR resource adequacy OR energy access OR gas supply OR decarbonisation OR sustainable electricity OR 100% renewable) AND (developing OR Africa OR Nigeria OR emerging OR Morocco)
```
*Arm B (financing):* replace the third clause with:
```
... AND (cost of capital OR WACC OR financing cost OR de-risking OR blended finance OR concessional OR financing structure) AND ...
```
Then apply filters: `from_publication_date:2015-01-01`, `type:article`, `language:en`. Deduplicate A∪B. Record counts A, B, A∪B: **[TO BE POPULATED]**.

*Practical note:* OpenAlex's free web interface (openalex.org) now supports much of this through its point-and-click filter builder — if the URL/JSON route feels too technical, build the two arms in the web UI's filter panel instead. The **logic** is what matters; the entry method is your choice. If you find the web UI can hold the full nested query reliably, use it as a single query and record one count — but verify it returns all three controls (§5) either way.

---

## 4. IEEE Xplore — engineering-venue depth

**Field-tag dialect.** IEEE Xplore's **Command Search** uses `("Field":term)` syntax. The relevant field bundle is `"All Metadata"` (title + abstract + index terms). IEEE's parser is the weakest of the three: it caps terms per query and handles deep nesting poorly. **Do not paste the full four-block string.** Run the two-arm split (unchanged from the original Stage 2):

**IEEE run A (security-arm) — Command Search:**
```
("All Metadata":"capacity expansion" OR "All Metadata":"energy system model" OR "All Metadata":"least-cost" OR "All Metadata":"generation expansion" OR "All Metadata":"electrification model" OR "All Metadata":"OnSSET" OR "All Metadata":"OSeMOSYS")
AND ("All Metadata":"power system" OR "All Metadata":"electricity" OR "All Metadata":"energy transition")
AND ("All Metadata":"energy security" OR "All Metadata":"resource adequacy" OR "All Metadata":"energy access" OR "All Metadata":"gas supply" OR "All Metadata":"decarbonization" OR "All Metadata":"sustainable electricity")
AND ("All Metadata":"developing" OR "All Metadata":"Africa" OR "All Metadata":"Nigeria" OR "All Metadata":"emerging" OR "All Metadata":"Morocco")
```

**IEEE run B (financing-arm)** — replace the third line with:
```
AND ("All Metadata":"cost of capital" OR "All Metadata":"WACC" OR "All Metadata":"financing cost" OR "All Metadata":"de-risking" OR "All Metadata":"blended finance" OR "All Metadata":"concessional")
```
Then filter both to Year ≥ 2015. Deduplicate A∪B in the reference manager. Counts A, B, A∪B: **[TO BE POPULATED]**.

*Note:* IEEE full-text is paywalled where the author lacks access, but **search and abstract-level screening are free** — sufficient for title/abstract screening. Full-text for included studies is obtained via the free author-accepted manuscripts (institutional repositories, ResearchGate, author sites) or interlibrary request.

---

## 5. Validation dry-run — the acceptance test (BEFORE screening)

*Unchanged in substance from the original Stage 2. This is the gate that tells you whether the free-source stack has adequate recall.*

**Procedure:** execute all three sources, deduplicate the combined pool, and confirm presence of **all three positive controls**:
- Agutu et al. (2022), *Nature Energy* 7:631–641 — DOI 10.1038/s41560-022-01041-6
- Oyewo et al. (2018), *Energy Conversion and Management* 178:44–64 — DOI 10.1016/j.enconman.2018.10.036
- Schinko et al. (2019), *Energy, Sustainability and Society* 9:1 — DOI 10.1186/s13705-018-0186-8

**Additional check specific to the free-source stack:** because you have swapped away from Scopus/WoS, the dry-run does double duty — it also tests whether the *stack itself* has adequate coverage. All three controls are indexed in Crossref (hence Lens and OpenAlex should both hold them; all three are journal articles with DOIs). If a control is present in Crossref but missing from your Lens/OpenAlex return, the failure is in your *string*, not the *source* — apply the diagnostic below. If a control is genuinely absent from a source's index (unlikely for these three, but check), that is a *coverage* finding to report.

**Pass criterion:** all three positive controls present in the combined return.

**If a control is missing — diagnostic (do NOT inject its DOI):**
- Test the missing paper's title against each block singly to find which block excluded it.
- Predicted weak point remains **Oyewo in Block 3** — now mitigated by the added `"sustainable electricity"` / `"100% renewable"` tokens. If Oyewo still misses, check its actual abstract/keywords for the exact token it uses and add it to Block 3a.
- Revise the offending block, re-run the *full* search, re-test. Log every revision (term added, reason, before/after count) in the search-revision log. Under the OSF registration, this log is how revisions stay transparent rather than concealed.

**Documented exclusions behave-check:** Calcaterra (2024) and Okorie (2025) may be *returned* by the search (on-topic) but *excluded at full-text* for pre-registered reasons — that is correct. The search must not be *designed* to drop them.

**Known-item supplementary check:** also test Dioha & Kumar (2020, *Energy Policy*, Nigeria NDC) and Schyska & Kies (2020, *Applied Energy*, regional CoC in power-system design). If the validated strings catch these too, recall confidence rises.

---

## 6. Corpus-size expectation

Low thousands combined, post-deduplication, is the target. Tens of thousands → Block 1/4 too broad (likely `Africa*` / `energy transition`); tighten. Low hundreds → Block 3 AND-collapse or the free sources under-indexing; inspect immediately, because an artificially small corpus manufactures a false gap. **Actual: [TO BE POPULATED].**

---

## 7. Reference-manager and screening pipeline (free tools)

- **Zotero** (free) — ingest the RIS/BibTeX/CSV exports from all three sources; deduplicate; store PDFs.
- **Rayyan** (free) — title/abstract screening; supports the second-screener + Cohen's κ workflow the protocol specifies.
- Export from Lens: use the workspace export (RIS/CSV). From OpenAlex: export via the web UI or API results. From IEEE: export citations (RIS).

---

## Open items owed
- **Author:** confirm Lens Academic workspace export actually functions before relying on the stack (verify, don't assume).
- **Author:** the Block 4 country list is the focal-sample-first version (agreed); expand only if validation/known-item recall fails.
- **Model path (not SLR):** NDC 3.0 baseline reconciliation (573.5 vs 614 Mt).
