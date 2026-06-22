# Security Policy

## Reporting a vulnerability

Please do not open public issues for security vulnerabilities.

Report privately using one of these methods:

- GitHub Security Advisories (preferred)
- Email the maintainer directly with subject: `NEXUS-ATMS Security Report`

Include:

- Affected file(s) and component(s)
- Reproduction steps
- Impact assessment
- Suggested fix (if available)

## Response targets

- Initial acknowledgement: within 72 hours
- Triage/update: within 7 days
- Fix timeline: depends on severity and reproducibility

## Supported versions

This project is under active development. Security fixes are applied to `main`.

## Security best practices for contributors

- Never commit secrets, tokens, or credentials.
- Keep dependencies updated.
- Validate and sanitize external inputs in API handlers.
- Avoid logging sensitive payloads.
