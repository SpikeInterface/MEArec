.. _releasenotes:

=============
Release notes
=============


Version 1.9.1
=============

6th April 2024

Bug fixes
---------

- Fix n_burst_spikes param (when missing from default dict) (#156)
- Fix jitter generation in drifting (#153)

Improvements
------------

- Load templates as memmap rather than in memory (#167)
- Little clean of padding templates (#165)
- Do not remove resampled templates if not resampled  enhancement (#163)
- Fix numpy deprecations (#162)
- Improve documentation (#159)


Version 1.9.0
=============

23rd May 2023


Bug fixes
---------

- Propagate filter order and add filter mode (#145)
- Fix resampling timing (#141)
- Fix cell selection for unknown cell types (#138)
- Fix filter: use chunk replacement instead of addition (#134)
- Fix passing template_ids (#120)


New features
------------


- Add option to control the drift linear gradient (#129)
- Define external drifts (#128)
- Add check_eap_shape parameter for template-generation (#126)
- Add extract_units_drift_vector (#122)


Improvements
------------

- Convolution computed in float for more precision (#118)

Packaging
---------

- Packaging: move to pyproject.toml, src structure, and black (#146)


Version 1.8.0
=============

18th July 2022

New features
------------

- Refactored drift generation (#97)
- Add LSB, ADC_BIT_DEPTH, and GAIN concepts (#104, #105)
- Add smoothing step to templates to make sure they sart and end at 0 (#110)

Improvements
------------

- Pre-generate random positions at template generation to improve speed (#95)
- Improve random seed management (#94)
- Use `packaging` instead of `distutils` (#114)
