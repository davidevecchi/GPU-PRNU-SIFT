# Frames Selection Strategies for PRNU-Based Source Camera Verification of Stabilized Videos

This repository contains a revised implementation of the official code for the "ICIP 2022" paper ["GPU-accelerated SIFT-aided source identification of stabilized videos"](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=C0v9f-cAAAAJ&citation_for_view=C0v9f-cAAAAJ:UeHWp8X0CEIC)

The code has undergone a comprehensive restructuring, now organized into modules and classes for improved clarity and maintainability.

### Usage

To utilize the newly implemented methods and testing modalities described in the paper, execute the following commands:

```python
python -u main.py --hypothesis 0 --method {ICIP|RAFT|NEW} --mode {ALL|I0|GOP0} --videos <vision/dataset/> --fingerprint <PRNU_fingerprints/> --output 'results'
python -u main.py --hypothesis 1 --method {ICIP|RAFT|NEW} --mode {ALL|I0|GOP0} --videos <vision/dataset/> --fingerprint <PRNU_fingerprints/> --output 'results'
```

### Additional Information

For further details on the implementation, its functionalities and usage, please refer to the original repository: [GPU-PRNU-SIFT](https://github.com/AMontiB/GPU-PRNU-SIFT)
