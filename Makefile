# Reproduces every figure in report/report.md. No data download is needed: the
# protocol experiments run on a synthetic replica, and the K derivation runs on
# the variance components logged in report/evidence.md.
PY = PYTHONPATH=$(CURDIR)/harness python3

.PHONY: all check protocol derivation report-check
all: check protocol derivation report-check

check:        ## harness self-tests; answers fixed by the competition rules
	cd experiments && $(PY) selftest.py

protocol:     ## why benchmark holdout, and what 3 folds can detect
	cd experiments && $(PY) leakage_test.py
	cd experiments && $(PY) power.py
	cd experiments && $(PY) heterogeneity.py

derivation:   ## K from the measured variance, and its regret profile
	cd experiments && $(PY) derive_k.py
	cd experiments && $(PY) sensitivity.py

report-check: ## every number in the report must trace to a logged measurement
	cd report && python3 crosscheck.py report.md evidence.md
