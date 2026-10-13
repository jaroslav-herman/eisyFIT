# Third-party notices

EIS Fitting is distributed under the GNU General Public License v3.0 or later.
It depends on the
open-source projects below; those projects remain under their own licenses.
When redistributing an installation or a derivative distribution, retain the
license and copyright notices supplied by the corresponding dependencies.

| Dependency | License | Project |
| --- | --- | --- |
| eisyfit-bayes-drt2 (imports as `bayes_drt2`) | BSD 3-Clause | <https://github.com/jaroslav-herman/bayes-drt2> |
| eisyfit-hybrid-drt (imports as `hybdrt`) | BSD 3-Clause | <https://github.com/jaroslav-herman/hybrid-drt> |
| eisyfit-galvani (imports as `galvani`) | GPLv3+ | <https://github.com/jaroslav-herman/galvani> |
| impedance | MIT | <https://github.com/ECSHackWeek/impedance.py> |
| Matplotlib | Matplotlib license | <https://matplotlib.org/stable/project/license.html> |
| natsort | MIT | <https://github.com/SethMMorton/natsort> |
| NumPy | BSD/MIT/Zlib/0BSD/CC0 components | <https://numpy.org/doc/stable/license.html> |
| pandas | BSD 3-Clause | <https://pandas.pydata.org/about.html#license> |
| SciPy | BSD 3-Clause | <https://scipy.org/scipylib/license.html> |
| PyArrow | Apache 2.0 | <https://arrow.apache.org/> |
| scikit-learn | BSD 3-Clause | <https://scikit-learn.org/stable/about.html#citing-scikit-learn> |
| PyGAD | Project license; verify bundled notices | <https://github.com/ahmedfgad/GeneticAlgorithmPython> |
| PySwarms | MIT | <https://github.com/ljvmiranda921/pyswarms> |
| eisyfit-wepy (imports as `wepy`) | GPLv3+; dependency notices apply | <https://github.com/jaroslav-herman/wepy> |

Some dependencies bring additional transitive dependencies. Their license
files and metadata are installed alongside those distributions and remain
authoritative for those components.

## Key scientific works behind the dependencies

The methods implemented by the analysis dependencies are described in the
following peer-reviewed publications:

- `impedance` — impedance.py: Murbach et al., *J. Open Source Software* 5(52),
  2349 (2020). <https://doi.org/10.21105/joss.02349>. Lin-KK validation:
  Schönleber et al., *Electrochimica Acta* 131, 20-27 (2014).
  <https://doi.org/10.1016/j.electacta.2014.01.034>
- `eisyfit-bayes-drt2` — hierarchical Bayesian EIS inversion: Huang, Papac &
  O’Hayre, *Electrochimica Acta* 367, 137493 (2021).
  <https://doi.org/10.1016/j.electacta.2020.137493>. Hyperparametric ridge
  regression: Ciucci & Chen, *Electrochimica Acta* 167, 439-454 (2015)
  <https://doi.org/10.1016/j.electacta.2015.03.123> and Effat & Ciucci,
  *Electrochimica Acta* 247, 1117-1129 (2017)
  <https://doi.org/10.1016/j.electacta.2017.07.050>
- `eisyfit-hybrid-drt` — probabilistic DRT analysis: Huang, Sullivan, Zakutayev
  & O’Hayre, *Electrochimica Acta* 443, 141879 (2023).
  <https://doi.org/10.1016/j.electacta.2023.141879>. Joint-domain impedance:
  Huang et al., *Joule* 8(7), 2049-2072 (2024); Huang & Zeier, *ACS Energy
  Letters* 11(2), 986-994 (2026)
- `scikit-learn` — Pedregosa et al., *J. Mach. Learn. Res.* 12, 2825-2830
  (2011). <https://doi.org/10.48550/arXiv.1201.0490>
