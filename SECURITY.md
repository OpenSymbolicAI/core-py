# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public GitHub issue.

**Preferred:** Use GitHub's private vulnerability reporting by opening a [new Security Advisory](https://github.com/OpenSymbolicAI/core-py/security/advisories/new). This is the channel we monitor.

**Fallback:** `rajkumar42@users.noreply.github.com` (not actively monitored — please prefer the advisory link above).

Please include a description of the issue and steps to reproduce if possible. We aim to acknowledge reports within 72 hours and will provide a timeline for a fix after triage.

## Security Best Practices

When using this library:

- Keep dependencies up to date
- Never commit API keys or secrets to the repository
- Use environment variables for sensitive configuration
- Review the permissions granted to any LLM integrations
