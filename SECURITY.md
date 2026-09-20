# Security Policy

## Current support status

OpenJev is in active development. There are no released model or software versions with a defined security support window yet. Release-specific support information will be added when artifacts are published.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** option if private vulnerability reporting is available on the hosting platform. Repository files alone do not enable that feature.

If that option is unavailable, use a private contact method explicitly listed by a maintainer. If no such method exists, open an issue titled “Request for a private security contact” with no vulnerability details, secrets, personal data, or exploit payload, and ask for a private channel.

Include the affected commit or model version, environment, reproduction steps, expected impact, and any proposed mitigation once a private channel is established. Please do not publish sensitive details in a normal bug report.

No response-time guarantee or bug bounty is currently offered. Maintainers will coordinate fixes and disclosure as the project develops.

## Model behavior and security

Incorrect classifications and poorly calibrated probabilities are model-quality issues unless they expose a specific security boundary failure. Report ordinary quality issues through the research template using synthetic or appropriately redacted examples.

Cross-request data leakage, unsafe artifact loading, credential exposure, and input compilation that permits structural injection are examples of security-relevant failures. A model's typed output does not make an application safe without validation and appropriate workflow controls.
