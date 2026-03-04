#!/bin/bash
set -e
#python3 ./command/00_alternate_file.py
#python3 ./command/01_folddisco_validation_query.py
#python3 ./command/01.5_folddisco_result_check.py
#python3 ./command/01.55_folddisco_nokey_adding_query.py
#python3 ./command/02_classify_folddisco_result_by_querylen.py
#python3 ./command/03_folddisco_gather_by_querylen.py
python3 ./command/04_fit_evalue.py
python3 ./command/05_test_and_plot_result.py