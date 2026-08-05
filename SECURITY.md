# Security Policy

## Supported code

Security fixes are applied to the current default branch and, when practical, the
most recent release line. Frozen scientific tags and milestone artifacts are not
rewritten: a security advisory or successor release must identify the affected
historical revision and the corrected replacement.

## Reporting a vulnerability

Please do not disclose an exploitable vulnerability, secret, private dataset path,
or unsafe artifact publicly before it has been assessed.

Use GitHub's private vulnerability-reporting facility for this repository when it
is available. Otherwise, contact the repository owner privately through the
institutional contact listed on the owner's GitHub profile. Include:

- the affected revision, package version, and execution environment;
- the smallest reproducible example;
- the expected and observed behavior;
- the security impact and required preconditions;
- whether untrusted files, checkpoints, repositories, or network services are
  involved;
- any proposed mitigation, without including unrelated private data.

The maintainers will assess scope, coordinate a fix or advisory, and preserve the
scientific provenance of any affected release or evidence artifact.

## Important trust boundaries

### Legacy pickle files are trusted-code inputs

Some compatibility paths load legacy Python pickle artifacts through a
Bayesian-PhysTwin provider boundary. Unpickling can execute code. A SHA-256 check
proves that the bytes match an expected digest; it does not make arbitrary pickle
content safe or sandbox its execution.

For registered physical-counterfactual evaluation, prefer the one-time
`causal4d evidence physical-target import-legacy` conversion. It requires
explicit unsafe-pickle consent and an expected digest, then emits a strict
non-pickled artifact used by the stable evaluator.

Only load a legacy pickle when all of the following hold:

1. the expected digest was obtained through a trusted, independently identified
   manifest or release channel;
2. the file came from the corresponding trusted producer;
3. the digest is verified before loading and rechecked where the provider contract
   requires it;
4. the process and host are appropriate for trusted research artifacts.

Do not use Causal4D's legacy loaders as a general-purpose pickle inspection tool.

### Digests provide identity, not authenticity by themselves

A digest supplied beside an artifact by the same untrusted source does not establish
trust. Claim-bearing manifests must bind expected identities, producer revisions,
provider versions, protocol files, and relevant environment information through a
trusted chain.

### Optional model code and checkpoints execute with process privileges

MolmoMotion, Warp/PhysTwin, GPU libraries, externally installed providers, and model
checkpoints may load native code or Python code and can access the permissions of the
running process. Use pinned sources and isolated environments, and do not expose
secrets or unrelated private datasets to exploratory model environments.

### NumPy, NPZ, JSON, and image inputs still require validation

Use `allow_pickle=False` for NumPy archives unless a narrowly documented trusted
legacy boundary requires otherwise. Validate paths, shapes, dtypes, finite values,
units, coordinate frames, timing, and content inventories before claim-bearing use.
Reject path traversal and unexpected files rather than guessing intent.

### Evidence and provenance data may be sensitive

Manifests, logs, exception traces, absolute paths, device identifiers, timestamps,
and raw acquisition metadata can reveal private infrastructure or participant data.
Do not commit credentials, tokens, private URLs, personal data, or unrestricted raw
acquisition artifacts. Redact only copies intended for sharing; never silently alter
the sealed original evidence record.

## Out of scope for public issue reports

General model accuracy, a failed scientific hypothesis, expected floating-point
variation within a registered tolerance, and missing optional research hardware are
not security vulnerabilities unless they create a concrete confidentiality,
integrity, availability, or code-execution risk.
