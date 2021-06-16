# Tests for BOOP accuracy

This repository holds reference data and tests used to check the accuracy of bond-orientational order parameters in freud against reference data.

To generate text files for freud, run:

```bash
python calculate_boops.py Test_Configuration.dat
```

The file `test_values.py` includes tests written in pytest for checking the accuracy of freud's values compared to the reference codes "GC" and "RvD". To run tests:

```bash
pytest -v -s
```
