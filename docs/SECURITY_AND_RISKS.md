# Security, Deployment, and Project Risks

MiniGPT is an educational language-model project. It is suitable for demonstrations and portfolio use, but it has not been hardened as a production service.

## Untrusted PyTorch checkpoints

PyTorch checkpoints are loaded with `torch.load()`. Traditional PyTorch checkpoint files use Python pickle internally, so a malicious or untrusted checkpoint may execute arbitrary code during loading.

Mitigation:

- Load only checkpoints created by this project or obtained from a trusted source.
- Do not allow public users to upload checkpoint files.
- Keep checkpoint paths controlled by the application owner.
- Consider a safer weight-only serialization format before accepting external models.

## Public Streamlit deployment

The Streamlit interface does not implement authentication, authorization, rate limiting, request quotas, or abuse prevention. A public instance could receive repeated or expensive generation requests.

Mitigation:

- Keep maximum generation length limited.
- Use platform access controls when available.
- Monitor CPU and memory consumption.
- Add rate limiting and authentication before treating the application as a public service.

## Resource exhaustion

Long prompts, large token limits, repeated requests, and larger checkpoints can consume significant CPU time and memory. Free hosting tiers may restart, sleep, or terminate the application when limits are exceeded.

Mitigation:

- Use a small model for public demonstrations.
- Restrict prompt length and generated-token count.
- Avoid loading multiple model checkpoints simultaneously.
- Test the deployment within the host's memory and execution limits.

## Dependency and supply-chain risk

`requirements.txt` lists dependencies without pinned versions. Future package releases could introduce incompatible behavior, security problems, or different results.

Mitigation:

- Pin tested dependency versions for releases and deployments.
- Review dependency security advisories.
- Rebuild and test the environment before publishing a new deployment.
- Do not commit the local `.venv/` directory.

## Dataset licensing and redistribution

The repository includes a Quran-derived text corpus and Juz metadata. The legal right to redistribute a dataset depends on its original source and license, which must be verified independently of the code.

Mitigation:

- Identify and document the original dataset source.
- Confirm its license permits redistribution and derived-model use.
- Add attribution and license notices where required.
- Exclude the dataset from the public repository if redistribution rights are unclear.

## Model-output risk

Generated text may be repetitive, malformed, inaccurate, or presented in a style resembling the training corpus. It must not be represented as authentic Quranic text, interpretation, translation, or religious guidance.

Mitigation:

- Display a clear educational-use disclaimer.
- Label outputs as model-generated text.
- Avoid claims of factual or theological accuracy.
- Do not use generated output in safety-critical, religious-authority, or factual-reference contexts.

## Artifact and repository size

Model checkpoints and experiment artifacts can make Git repositories large. GitHub rejects individual files larger than 100 MB, and repositories containing many binary checkpoints can become slow to clone and maintain.

Mitigation:

- Keep generated checkpoints out of normal Git history.
- Publish selected models through Git LFS, GitHub Releases, or model-hosting storage.
- Keep `.venv/`, caches, and temporary experiment files ignored.
- Review staged files before pushing.

## Secrets and configuration

The current project does not require API keys, but deployment changes may introduce credentials or platform tokens. Accidentally committing them would expose account access.

Mitigation:

- Store secrets in environment variables or the deployment platform's secret manager.
- Never place credentials directly in source files or committed configuration.
- Inspect commits and staged files before pushing to GitHub.
- Rotate any secret immediately if it is exposed.

## Reproducibility limitations

Fixed seeds improve repeatability, but exact results may still differ across Python, PyTorch, operating-system, CPU, or GPU versions. Free deployment environments can also change without notice.

Mitigation:

- Record the seed, configuration, dataset version, dependency versions, and environment.
- Keep experiment artifacts and commands together.
- Treat small numerical differences across platforms as expected unless deterministic execution is explicitly enforced.

## Current security posture

The project is appropriate for local use, controlled demonstrations, and portfolio presentation when checkpoints and inputs are trusted. Before production use, it would need stronger artifact validation, dependency pinning, authentication, rate limiting, monitoring, resource controls, and a confirmed dataset license.
