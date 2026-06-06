# climagrid Governance

## Project status

climagrid is an open-source project released under the Apache 2.0 License with the explicit goal of being freely usable by any electric utility, rural cooperative, researcher, or regulator in the United States and worldwide.

The Apache 2.0 license will never change. Patent grants and freedom from royalties are permanent.

## Current model: Benevolent Dictator For Now (BDFN)

The project is currently maintained by its author. Major decisions (API changes, new data source additions, schema column additions or renames) are made by the maintainer with input from the community via GitHub issues and discussions.

## Commit rights

Commit rights are granted to contributors who have:

1. Submitted at least two accepted pull requests
2. Demonstrated familiarity with the column schema conventions in `schema.py`
3. Written tests that pass in CI without live API calls

Request commit rights by opening a GitHub issue titled "Commit access request".

## Decision process

For non-breaking changes (new source adapters, new feature modules, documentation): a pull request with a passing CI and one maintainer approval is sufficient.

For breaking changes (schema column renames, removed public API): a 14-day comment period on the relevant GitHub issue is required before merging.

## Roadmap

The roadmap is discussed in GitHub Discussions. Community members are encouraged to propose new data sources and feature modules through the issue tracker.

## Evolution

As the contributor base grows, governance will evolve toward a Steering Committee model with representatives from rural cooperatives and utility research organizations. Governance changes themselves require a 30-day comment period and consensus among existing committers.

## Security

Report security issues privately to temidireadesiji@gmail.com. Do not open public issues for security vulnerabilities. We aim to acknowledge reports within 48 hours and provide a resolution timeline within 5 business days. See [SECURITY.md](SECURITY.md) for the full policy.
