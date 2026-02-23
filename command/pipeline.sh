#!/bin/bash
set -e 
#python3 ./command/alternate_file.py
#python3 ./command/01_folddisco_validation_query.py
#python3 ./command/01.5_folddisco_result_check.py 
#python3 ./command/01.55_folddisco_nokey_adding_query.py 
#python3 ./command/02_classify_folddisco_result_by_querylen.py 
#bash ./command/03_folddisco_gather_by_querylen.sh 
#python3 ./command/04_folddisco_evalue_parameters_computing.py 
python3 ./command/05_folddisco_result_to_evalue_fitted.py
python3 ./command/06_folddisco_evalue_analysis.py
#python3 ./command/07_evalue_distribution_check.py 
python3 ./command/08_evalue_calibration.py
