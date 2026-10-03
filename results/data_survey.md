# measurement-db survey

Source: `aims-foundations/measurement-db` revision `bc8204d811823da849c6686bf124d4ca9f82e4de`

* benchmark directories on HF: 10; response_type {'mixed': 2, 'fraction': 2, 'binary': 6}; granularity {'item': 10}
* eligible (binary, item-level) loaded: **4**
* eligible with >= 80 items (evaluable): **4**
* responses (one per subject x item x interactors): **74,445**
* items: **8,877**; subject rows (configurations): **147**; distinct model identities: **60** (60 with a normalized name)
* identities appearing in >= 2 / >= 5 / >= 10 benchmarks: 17 / 0 / 0
* density (responses / subjects x items): median 0.744, overall 0.382
* base rate: pooled 0.206, per-benchmark median 0.364 (IQR 0.301-0.400)
* items per benchmark: median 1180, max 6,306; subjects per benchmark: median 32, max 82

## Subject attribute fill rates

| field | non-empty |
|---|---|
| normalized_name | 100.0% |
| provider | 100.0% |
| release_date | 93.9% |
| access_date | 55.8% |
| harness | 55.8% |
| reasoning_effort | 0.0% |
| harness_version | 0.7% |
| subject_features_extra | 0.0% |

## Item attributes

* item_content: median 931 chars, 90th pct 3557
* item_features non-empty in 2,571 items; most common keys: `lang` (2126), `website` (233), `paper` (212)
* distinct non-empty interactors values: 0

## Largest benchmarks

| benchmark         | domain                                   |   subjects |   items |   responses |   density |   base_rate |   item_features_nonempty |   interactors_nonempty |   median_content_chars |
|:------------------|:-----------------------------------------|-----------:|--------:|------------:|----------:|------------:|-------------------------:|-----------------------:|-----------------------:|
| multi_swebench    | software_engineering,agents_and_tool_use |         82 |    2126 |       57808 |     0.332 |       0.148 |                    1.000 |                  0.000 |               1147.000 |
| researchcodebench | software_engineering,ml_engineering      |         31 |     212 |        6572 |     1.000 |       0.352 |                    1.000 |                  0.000 |              96287.500 |
| swe_rebench       | software_engineering,agents_and_tool_use |          1 |    6306 |        6306 |     1.000 |       0.475 |                    0.000 |                  0.000 |                857.000 |
| real_webagents    | agents_and_tool_use                      |         33 |     233 |        3759 |     0.489 |       0.375 |                    1.000 |                  0.000 |                 92.000 |
