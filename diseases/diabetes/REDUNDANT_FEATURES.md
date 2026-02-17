# Redundant feature groups (diabetes priority)

Variables are assigned to **groups** in `feature_priority.csv` (column `group`). The runner can pick features **by group** using `group_priority.csv`: it selects round-robin (1st from each group, then 2nd from each group, …) so you get one from each high-priority group first, then add more from those same groups.

Within each group, many variables are **redundant** (same construct). Below: which variables belong to the same construct and a suggested **canonical** (one or two) to prefer when you want a single representative.

---

## 1. Age (6 variables → keep 1)

- **Variables:** `AGEY_B`, `AGEY_E`, `AGEY_M`, `AGEM_B`, `AGEM_E`, `AGEM_M`
- **Why redundant:** Age in years vs months; baseline (B) vs interview (E/M). All measure the same thing; extremely high correlation.
- **Keep:** `AGEY_B` (or `AGEY_E`) — one age variable is enough.

---

## 2. Blood pressure (4 variables → keep 2)

- **Variables:** `BPSYS`, `BPDIA`, `BPPOS`, `BPPULS`
- **Why redundant:** BPSYS/BPDIA are the main clinical BP measures. BPPOS (position), BPPULS (pulse) are secondary and often correlated with the main BP.
- **Keep:** `BPSYS` and `BPDIA`; drop BPPOS/BPPULS unless you need position/pulse specifically.

---

## 3. Depression (CESD total + 16 items → keep 1)

- **Variables:** `CESD`, `CESDM`, and all `DEP*` (DEPCON, DEPDOWN, DEPEVD, DEPHUN, DEPLOS, DEPNIT, DEPNOAP, DEPPOR, DEPRECM, DEPRES, DEPREX, DEPSLE, DEPTHO, DEPTIR, DEPWKS, DEPYR)
- **Why redundant:** In RAND HRS, CESD is the **total score** of the depression scale; the DEP* variables are typically the **individual items** that sum to it. Using CESD plus all DEP* is redundant (the items are the components of CESD).
- **Keep:** `CESD` (or `CESDM` if that’s the version in your build). Use either the total or the items, not both.

---

## 4. Self-rated health (6+ variables → keep 1)

- **Variables:** `SHLT`, `SHLTC`, `SHLTCF`, `HLTHLM`, `LBFAMHLTHPRB`, `LBHLTHPRB`, `LBSATHLTH`
- **Why redundant:** All capture “self-rated health” or health perception; highly correlated.
- **Keep:** `SHLT` (main self-rated health in HRS).

---

## 5. Body weight / size (4–8 variables → keep 1 with BMI)

- **Variables:** `BWC20`, `BWC20P`, `BWC20W`, `BWC86` (and possibly `FBWC*`): body weight change or weight-related.
- **Why redundant:** Different definitions of weight change; often correlated with each other and with BMI.
- **Keep:** `BMI` is the primary obesity/body-size feature. Add at most one BWC* if you explicitly want weight change; otherwise skip BWC* when you already use BMI.

---

## 6. Physical activity (5 variables → keep 1–2)

- **Variables:** `CACT`, `CACTP`, `FCACT`, `FCACTP`, `VIGACT`, `MODACT`
- **Why redundant:** Overlapping measures of current/frequency of activity; vigorous vs moderate may be correlated.
- **Keep:** One of `VIGACT` or `MODACT`, or a single summary like `CACT`; avoid using all five.

---

## 7. ADL / IADL (40+ variables → keep 1–2 summaries)

- **Variables:** `ADLR10`, `ADL5A`, `ADL5H`, `ADL6A`, `ADL6H`, `ADLW`; plus individual items: `BATH`, `BED`, `DRESS`, `EAT`, `TOILT`, `WALK1`, `WALKR`, `WALKS` (and variants with suffixes _A, _H, _W), `IADL5A`, `IADL5H`, `SADLF`, etc.
- **Why redundant:** ADLR10 (and ADL5/6*) are **counts or summaries** of limitations; BATH, BED, DRESS, EAT, etc. are the **individual items** that go into those counts. Using ADLR10 plus all items is redundant. Suffixes (_A, _H, _W) often indicate “any help”, “hours”, “waves” — same underlying activity.
- **Keep:** `ADLR10` (or one ADL* summary) and optionally one IADL summary; drop the individual ADL/IADL items when using the summary.

---

## 8. Mobility / balance (20+ variables → keep 1–2)

- **Variables:** `MOBILA`, `MOBILW`, `BACK`, `CHAIR`, `CLIM1`, `CLIMS`, `BALFUL`, `BALFULC`, `BALFULT`, `BALSBS`, `BALSBSC`, `BALSEMI`, `BALSEMIC`, `DCBAL1`–`DCBAL4`, `FDCBAL1`–`FDCBAL4`
- **Why redundant:** All capture mobility, balance, or physical function; many are highly correlated.
- **Keep:** `MOBILA` (mobility limitation); optionally one balance measure if needed.

---

## 9. Smoking (2 variables → keep 1)

- **Variables:** `SMOKEV`, `SMOKEN`
- **Why redundant:** Ever vs never smoked; one binary or the other is usually enough.
- **Keep:** One of `SMOKEV` or `SMOKEN`.

---

## Summary for “non-redundant” top-K

When selecting top-K for diabetes, prefer **one representative per construct**:

| Construct      | Prefer (canonical)     | Avoid adding many from same row |
|----------------|------------------------|----------------------------------|
| Age            | AGEY_B                 | AGEY_E, AGEY_M, AGEM_*           |
| BP             | BPSYS, BPDIA           | BPPOS, BPPULS                    |
| Depression     | CESD                   | CESDM, all DEP*                  |
| Self-rated health | SHLT                | SHLTC, SHLTCF, HLTHLM, LB*       |
| Body size      | BMI                    | BWC20, BWC20P, BWC20W, BWC86     |
| Activity       | VIGACT or MODACT       | CACT, CACTP, FCACT, FCACTP       |
| ADL/IADL       | ADLR10                 | ADL5/6*, BATH, BED, DRESS, …     |
| Mobility       | MOBILA                 | MOBILW, BACK, CHAIR, CLIM*, BAL* |
| Smoking        | SMOKEV or SMOKEN       | both                             |

The priority file still lists all 800 so you can rank them; when building a small model, pick the **highest-scored variable in each group** rather than taking the top 20 and ending up with 6 age variables and 10 ADL items.
