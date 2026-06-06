# National Impact

## The reliability gap in rural America

The United States has approximately 900 electric cooperatives serving **42 million people** across 56% of the nation's land area, primarily rural, agricultural, and remote communities. These cooperatives collectively operate roughly 2.5 million miles of distribution lines, the majority of which traverse terrain with elevated exposure to:

- Extreme heat events (transformer thermal overload)
- Wildfire and smoke (insulation degradation, conductor faults)
- Ice storms and freeze-thaw cycles (mechanical failure of poles and hardware)
- Flooding and soil saturation (underground cable damage, substation flooding)

Despite this elevated risk exposure, most rural cooperatives lack the data infrastructure and engineering resources to run predictive maintenance programs. The investor-owned utilities that serve urban markets can afford dedicated data science teams; a 12-employee cooperative in rural Texas typically cannot.

---

## The data gap climagrid fills

The environmental data needed for grid resilience modeling already exists and is freely available from federal agencies:

| Agency | Dataset | Relevance |
|--------|---------|-----------|
| NASA | POWER (MERRA-2 surface met) | Temperature, wind, solar, humidity |
| NOAA | HRRR 3 km NWP | High-resolution weather forecasts |
| NOAA | NCEI CDO station records | Ground truth observations |
| USDA NRCS | SCAN/SNOTEL sensors | Soil moisture, snowpack |
| USFS NIFC | WFIGS wildfire perimeters | Active fire proximity |

The barrier is not data availability. It is the translation layer. Converting these five disparate APIs into IEEE-standard engineering stress features that a cooperative's asset management system can consume requires expertise in meteorology, power systems engineering, geospatial processing, and software engineering simultaneously. Very few rural cooperatives have all four.

climagrid is that translation layer, openly licensed and requiring no API keys for the core functionality.

---

## Downstream applications

Organizations using climagrid's output can build or improve:

**Predictive maintenance models**
: Combine the feature matrix with a utility's own historical failure records to train a model that ranks assets by environmental stress. The strength of any link between stress and failure must be validated on the utility's own data; climagrid supplies the standards-based environmental inputs, not a validated failure prediction.

**Dynamic line rating (DLR)**
: Use `feat_conductor_sag_index` (a simplified IEEE 738 based index) as a screening indicator of when cool, windy conditions lower thermal sag risk. Full dynamic line rating requires a calibrated IEEE 738 conductor-temperature calculation with electrical load, which is beyond this index.

**Insurance and regulatory reporting**
: Document environmental stress exposure history for FERC, state PUCs, or insurance carriers. Apache 2.0 license means regulators can reproduce calculations independently.

**Climate adaptation planning**
: Project equipment life expectancy under CMIP6 future climate scenarios by feeding climate model output through the same IEEE engineering models.

---

## Geographic coverage

climagrid's default source (NASA POWER) provides global coverage, making the toolkit applicable to:

- **U.S. rural cooperatives:** the primary design target
- **Municipal utilities** in small cities and towns
- **Tribal utilities** and remote community microgrids
- **International electric utilities** in developing countries where grid resilience data is even scarcer

The NOAA HRRR adapter provides higher resolution (3 km) for the continental United States, and the USDA NRCS and USFS WFIGS adapters provide soil and wildfire data specifically for western U.S. cooperative service territories.

---

## Connection to federal grid resilience policy

climagrid's design is directly aligned with:

- **DOE Grid Resilience and Innovation Partnerships (GRIP):** funds rural cooperative grid modernization
- **USDA ReConnect Program:** rural broadband and infrastructure
- **NERC Reliability Standards:** FAC-002 and TPL-001 (transmission planning), where environmental stress is a relevant input to reliability assessments
- **FERC Order 881:** requires utilities to implement ambient-adjusted ratings (AAR) for transmission lines, a use case the IEEE 738 conductor thermal modeling behind `feat_conductor_sag_index` is relevant to

By releasing under Apache 2.0, climagrid ensures that federally-funded cooperative modernization programs can adopt it without license barriers, and that FERC and NERC staff can audit the underlying calculations.
